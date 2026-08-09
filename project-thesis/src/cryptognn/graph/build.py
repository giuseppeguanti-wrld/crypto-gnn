"""Construction of the dynamic graph from rolling correlation matrices.

Turns each correlation matrix C_t into the three objects the rest of the study
consumes, following the chain of thesis Section 4.3 (from correlation matrix to
graph):

  C_t --Mantegna--> W_full --threshold--> W_thresh --renormalize--> A_hat

The Mantegna transform maps a correlation into a metric distance
d_ij = sqrt(2(1 - rho_ij)) and then into a weight w_ij = 1 - d_ij/2 in [0, 1].
Non-negative weights are not a convenience: they are what guarantees the graph
Laplacian is positive semidefinite (thesis prop:psd), which every spectral
quantity computed downstream relies on.

Which graph feeds which consumer (PLANNING.md Section 1.2):
  - W_full   -> topological metrics (lambda_2, MST, spectral entropy). No
                threshold, so lambda_2 never collapses to 0 on a disconnected
                graph during calm periods.
  - W_thresh -> graph density, the one metric that depends on tau by definition.
  - A_hat    -> the GCN substrate, with self-loops and the renormalization trick.

Exports:
  - mantegna_distance(), mantegna_weights(): correlation -> distance -> weight
  - apply_threshold(): zero out edges below tau, on signed rho
  - normalized_adjacency(): D^-1/2 (W + I) D^-1/2, i.e. eq:renormalization
  - edge_density(): fraction of surviving edges among the N(N-1)/2 pairs
  - validate_weights(), validate_adjacency(): raise on violated invariants

Integration: called by scripts/02_build_graphs.py on the (K, N, N) correlation
  tensor from graph.correlation, with tau from graph.threshold; the saved
  W_full.npy / W_thresh.npy / A_hat.npy feed graph.metrics (Sprint 2) and
  models.gcn (Sprint 4).
Why every function is shape-agnostic: they operate on the last two axes only,
  so a single (N, N) matrix and a batched (K, N, N) tensor take the same path.
  All ~1947 windows are transformed in one vectorized pass, with no Python loop
  and no separate single-window code path that could drift out of agreement.
"""
from __future__ import annotations

import numpy as np


def mantegna_distance(corr: np.ndarray) -> np.ndarray:
    """Mantegna distance d_ij = sqrt(2 (1 - rho_ij)), in float64.

    Maps perfect correlation to distance 0, independence to sqrt(2), and perfect
    anticorrelation to 2. Unlike 1 - rho it satisfies the triangle inequality,
    which is what makes the minimum spanning tree of Section 6.6 well defined.

    `corr` is clipped to [-1, 1] first. The rolling correlation tensor is stored
    in float32 and validate_correlation() deliberately tolerates values outside
    the range by up to 1e-6; without the clip, a rho of 1 + eps would put a
    negative number under the square root and silently produce NaN.
    """
    clipped = np.clip(np.asarray(corr, dtype=np.float64), -1.0, 1.0)
    return np.sqrt(2.0 * (1.0 - clipped))


def mantegna_weights(corr: np.ndarray) -> np.ndarray:
    """Edge weights w_ij = 1 - d_ij / 2 in [0, 1], with a zeroed diagonal.

    w is increasing in rho: 1 at rho = 1, 1 - sqrt(2)/2 ~ 0.293 at rho = 0, and 0
    at rho = -1. Being non-negative by construction, it keeps the Laplacian
    positive semidefinite (thesis prop:psd).

    The diagonal is set to 0 rather than left at its natural w_ii = 1. A weight
    of 1 on the diagonal would be a self-loop already baked into W, so
    normalized_adjacency()'s A_tilde = W + I would carry a diagonal of 2 and
    count each self-loop twice relative to eq:renormalization. Keeping W free of
    self-loops means they enter exactly once, in normalized_adjacency(), under
    the control of config.graph.self_loops -- and it also lets edge_density()
    and the MST of graph.metrics treat every stored entry as a real edge.
    """
    weights = 1.0 - mantegna_distance(corr) / 2.0
    weights = np.clip(weights, 0.0, 1.0)  # guard the [0, 1] contract against float error
    np.einsum("...ii->...i", weights)[...] = 0.0
    return weights


def apply_threshold(corr: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
    """Keep the weight of every pair correlated above `tau`, zero the rest.

    The mask is taken from `corr` and applied to `weights`: the decision is made
    on the correlation, the surviving value is the Mantegna weight, unrescaled.

    Thresholding is on **signed** rho, not |rho|, so a strongly anticorrelated
    pair is removed rather than kept as a strong edge. This is a modelling
    choice to declare in Section 6.2, and it is not a no-op: at the lowest
    correlation observed in this study (rho = -0.551) the Mantegna weight is
    still 0.119, and at rho = tau it is 0.373, so the threshold does remove
    edges of genuinely non-zero weight rather than merely dropping near-zeros.

    The boundary is exclusive (`rho > tau` survives), matching the calibration
    in graph.threshold, where tau is the (1 - alpha) quantile of the null: a
    correlation exactly at the quantile is not evidence against the null.
    """
    mask = np.asarray(corr, dtype=np.float64) > tau
    return np.where(mask, np.asarray(weights, dtype=np.float64), 0.0)


def normalized_adjacency(weights: np.ndarray, *, self_loops: bool = True) -> np.ndarray:
    """Symmetrically normalized adjacency A_hat = D^-1/2 (W + I) D^-1/2.

    This is eq:renormalization, the renormalization trick of Kipf & Welling:
    adding self-loops before normalizing keeps a node's own features in its
    representation and confines the spectrum of A_hat to [-1, 1], so stacking
    layers neither explodes nor vanishes the signal.

    `self_loops` comes from config.graph.self_loops rather than being assumed.
    With self_loops=False an isolated node has degree 0 and the normalization
    divides by zero; this is not hypothetical on real data (62 of the 1947
    windows of this study contain at least one isolated node at the calibrated
    tau), so the degenerate case raises a ValueError naming the offending
    windows instead of propagating inf and NaN into the GCN.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n_nodes = weights.shape[-1]

    adjacency = weights + np.eye(n_nodes) if self_loops else weights.copy()
    degree = adjacency.sum(axis=-1)

    if (degree <= 0).any():
        isolated = np.argwhere(degree <= 0)
        raise ValueError(
            f"{len(isolated)} node(s) with zero degree at indices {isolated[:10].tolist()} "
            f"(self_loops={self_loops}): normalization would divide by zero. "
            "Enable self-loops or lower the threshold."
        )

    d_inv_sqrt = degree ** -0.5
    return d_inv_sqrt[..., :, None] * adjacency * d_inv_sqrt[..., None, :]


def edge_density(weights: np.ndarray) -> np.ndarray | float:
    """Fraction of the N(N-1)/2 possible pairs that carry a non-zero weight.

    Counts the upper triangle only, so the zeroed diagonal of mantegna_weights()
    is excluded and the denominator is 105 for the 15 assets of this study, not
    120. Returns a scalar for a single (N, N) matrix, or one value per window
    for a batched (K, N, N) tensor.

    This is the canonical density computation of the study; graph.metrics
    delegates to it so the figures of Section 6.6 and the diagnostics written
    into tau_calibration.json can never disagree.
    """
    weights = np.asarray(weights, dtype=np.float64)
    rows, cols = np.triu_indices(weights.shape[-1], k=1)
    pairs = weights[..., rows, cols]
    return (pairs != 0.0).mean(axis=-1)


def validate_weights(weights: np.ndarray, *, atol: float = 1e-6) -> None:
    """Assert the invariants of a Mantegna weight matrix, raising a ValueError on
    the first violation. A failure here is a bug in this module, never a
    property of the data.

      1. No NaN (the signature of a negative argument reaching the square root).
      2. Symmetry: w_ij == w_ji.
      3. Every weight in [0, 1] -- non-negativity is what keeps the Laplacian
         positive semidefinite (prop:psd).
      4. Zero diagonal, so self-loops are added exactly once downstream.
    """
    weights = np.asarray(weights)

    if np.isnan(weights).any():
        bad = np.argwhere(np.isnan(weights))
        raise ValueError(f"NaN weights at indices: {bad[:10].tolist()}")

    transposed = np.swapaxes(weights, -1, -2)
    if not np.allclose(weights, transposed, atol=atol):
        bad = np.argwhere(~np.isclose(weights, transposed, atol=atol))
        raise ValueError(f"Weights not symmetric within {atol} at indices: {bad[:10].tolist()}")

    out_of_range = (weights < -atol) | (weights > 1.0 + atol)
    if out_of_range.any():
        bad = np.argwhere(out_of_range)
        raise ValueError(f"Weights outside [0, 1] at indices: {bad[:10].tolist()}")

    diagonal = np.einsum("...ii->...i", weights)
    if not np.allclose(diagonal, 0.0, atol=atol):
        bad = np.argwhere(~np.isclose(diagonal, 0.0, atol=atol))
        raise ValueError(f"Non-zero diagonal (unexpected self-loops) at indices: {bad[:10].tolist()}")


def validate_adjacency(a_hat: np.ndarray, *, atol: float = 1e-6) -> None:
    """Assert the invariants of a renormalized adjacency, raising a ValueError on
    the first violation.

      1. No NaN or inf (the signature of a zero-degree node slipping through).
      2. Symmetry, which is what makes the eigenvalues real to begin with.
      3. Spectrum contained in [-1, 1]: the property the renormalization trick
         exists to provide, and the reason stacked layers stay numerically
         stable. tests/test_gcn.py re-checks it as test_renormalized_spectrum;
         here it is a cheap runtime guard on the actual artifact.
    """
    a_hat = np.asarray(a_hat, dtype=np.float64)

    if not np.isfinite(a_hat).all():
        bad = np.argwhere(~np.isfinite(a_hat))
        raise ValueError(f"Non-finite adjacency entries at indices: {bad[:10].tolist()}")

    transposed = np.swapaxes(a_hat, -1, -2)
    if not np.allclose(a_hat, transposed, atol=atol):
        bad = np.argwhere(~np.isclose(a_hat, transposed, atol=atol))
        raise ValueError(f"Adjacency not symmetric within {atol} at indices: {bad[:10].tolist()}")

    eigenvalues = np.linalg.eigvalsh(a_hat)
    if (eigenvalues < -1.0 - atol).any() or (eigenvalues > 1.0 + atol).any():
        raise ValueError(
            f"Adjacency spectrum outside [-1, 1]: min {eigenvalues.min():.6f}, "
            f"max {eigenvalues.max():.6f} -- the renormalization trick should prevent this"
        )
