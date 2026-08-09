"""Topological metrics of the dynamic correlation graph (thesis Section 6.6).

Reduces each window's 15x15 matrix to a handful of scalars that can be plotted
against time, so the question of Section 6.6 -- does the market's network
structure compact during a crisis? -- becomes a series to read rather than 1947
matrices to inspect.

Which graph feeds which metric is not interchangeable (PLANNING.md Section 1.2):

  - Spectral metrics (lambda_2, spectral entropy, market mode share, eigenvalues
    outside the Marchenko-Pastur bulk) are computed on the **complete** graph or
    on C directly, with no threshold. A thresholded graph can be disconnected --
    62 of this study's 1947 windows have an isolated node at the calibrated tau
    -- and lambda_2 of a disconnected graph is identically 0, which would report
    "no connectivity" for what is really "the threshold cut a node loose".
  - Density is the one metric that depends on the threshold by definition, and
    is therefore the only one computed on the thresholded graph.

Exports:
  - mean_correlation(), graph_density()
  - algebraic_connectivity(), algebraic_connectivity_combinatorial()
  - mst_length(), spectral_entropy(), market_mode_share(), eigs_outside_mp()

Integration: called by scripts/03_topology_analysis.py over the (K, N, N)
  tensors from graph.build; output is results/metrics/topology.parquet, which
  feeds the event study (S2.4), the figures of Section 6.6 (S2.5), and the
  Streamlit explorer (Sprint 6).
Why two algebraic connectivities: L_sym is invariant to a uniform rescaling of
  the weights (W -> cW leaves D^-1/2 W D^-1/2 unchanged), so on a complete graph
  it measures how *uneven* the weights are, not how strong they are. On this
  data that makes it nearly flat (cv 0.021, range [0.923, 1.058]) while the
  combinatorial Laplacian L = D - W, which has no such invariance, moves with
  the market (cv 0.178). Both are reported so Section 6.6 can state the
  distinction rather than silently pick the one that looks better.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from cryptognn.graph.build import edge_density, normalized_adjacency


def mean_correlation(corr: np.ndarray) -> np.ndarray | float:
    """Average correlation over the N(N-1)/2 distinct pairs.

    The simplest possible summary of market co-movement, and the reference the
    other metrics are judged against: a topological metric that does not track
    it at all is either measuring something genuinely different or is broken.
    The diagonal is excluded, so the constant 1s do not inflate the mean.
    """
    corr = np.asarray(corr, dtype=np.float64)
    rows, cols = np.triu_indices(corr.shape[-1], k=1)
    return corr[..., rows, cols].mean(axis=-1)


def graph_density(weights: np.ndarray) -> np.ndarray | float:
    """Fraction of possible edges present in the thresholded graph.

    Delegates to graph.build.edge_density(), which is the single implementation
    of this count in the study: the diagnostics written into
    tau_calibration.json and the series plotted in Section 6.6 must never be
    able to disagree. Exists under this name because PLANNING.md and the thesis
    refer to it as graph density among the topological metrics.
    """
    return edge_density(weights)


def algebraic_connectivity(weights: np.ndarray) -> np.ndarray | float:
    """lambda_2 of the symmetric normalized Laplacian L_sym = I - D^-1/2 W D^-1/2.

    The second-smallest eigenvalue is 0 exactly when the graph is disconnected,
    and grows as the graph becomes better connected -- which is why Section
    sec:network-structure-crisis-regimes favours it: unlike density, it needs no
    threshold to be defined.

    Computed on the **complete** Mantegna graph, where every weight is positive
    and the graph is therefore always connected. That is what makes it safe to
    read lambda_2 off index 1 of the ascending spectrum: lambda_1 is 0 by
    construction (verified to 8.9e-16 on this study's data), so index 1 is the
    genuine second eigenvalue rather than a second zero from a second connected
    component.

    Note the scale invariance: L_sym is unchanged by W -> cW, so on a complete
    graph this responds to weight *heterogeneity*, not to the overall level of
    correlation. See algebraic_connectivity_combinatorial() for the variant that
    does track the level.

    Reuses build.normalized_adjacency(self_loops=False) rather than repeating
    the D^-1/2 (.) D^-1/2 construction; self-loops must be off here because
    L_sym is defined on the graph itself, not on the GCN's renormalized
    substrate.
    """
    weights = np.asarray(weights, dtype=np.float64)
    laplacian = np.eye(weights.shape[-1]) - normalized_adjacency(weights, self_loops=False)
    return np.linalg.eigvalsh(laplacian)[..., 1]


def algebraic_connectivity_combinatorial(weights: np.ndarray) -> np.ndarray | float:
    """lambda_2 of the combinatorial Laplacian L = D - W (the Fiedler value).

    Same interpretation as algebraic_connectivity() -- 0 iff disconnected,
    larger when better connected -- but without the normalization, and therefore
    without its scale invariance: doubling every weight doubles this value. On
    the complete Mantegna graph that difference is decisive, since a market-wide
    rise in correlation raises all weights together and is invisible to L_sym.
    """
    weights = np.asarray(weights, dtype=np.float64)
    degree = weights.sum(axis=-1)
    laplacian = degree[..., :, None] * np.eye(weights.shape[-1]) - weights
    return np.linalg.eigvalsh(laplacian)[..., 1]


def mst_length(distance: np.ndarray) -> np.ndarray | float:
    """Normalized length of the minimum spanning tree: sum of its edges / (N-1).

    This is the metric of onnela2003dynamics. The MST keeps the N-1 edges that
    connect every asset at minimum total Mantegna distance, so its length is a
    threshold-free summary of how tightly the market is bound: distances shrink
    as correlations rise, and the tree shortens as the market compacts. Dividing
    by N-1 turns it into a mean edge length, comparable across universe sizes.

    `distance` is the Mantegna distance from build.mantegna_distance(), not a
    weight matrix: the MST minimizes cost, so it must be fed distances, and
    passing weights would return the *maximum* spanning tree of the market.

    Unlike the other metrics this loops over windows -- networkx builds one
    graph object per matrix. Measured at 0.42 s for the study's 1947 windows,
    which is immaterial against the script's runtime.
    """
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim == 2:
        return _single_mst_length(distance)
    return np.array([_single_mst_length(matrix) for matrix in distance])


def _single_mst_length(distance: np.ndarray) -> float:
    n_nodes = distance.shape[-1]
    graph = nx.from_numpy_array(distance)
    tree = nx.minimum_spanning_tree(graph)
    return sum(data["weight"] for _, _, data in tree.edges(data=True)) / (n_nodes - 1)


def spectral_entropy(corr: np.ndarray) -> np.ndarray | float:
    """Shannon entropy of the normalized eigenvalue spectrum of C.

    With lambda~_k = lambda_k / sum(lambda), the entropy -sum(lambda~ log lambda~)
    measures how evenly variance is spread across the principal components. It
    is maximal at log(N) = 2.708 for N = 15 when C = I (every direction equally
    important, no common structure) and falls towards 0 as a single mode takes
    over -- so a *drop* signals the market collapsing onto one factor.

    The x log x -> 0 limit is applied explicitly: C is positive semidefinite, so
    a numerically zero eigenvalue is possible in principle and would otherwise
    produce -inf. (On this study's data the smallest observed eigenvalue is
    6.5e-3, since T_w = 60 > N = 15 keeps C full rank.)
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(corr, dtype=np.float64))
    normalized = eigenvalues / eigenvalues.sum(axis=-1, keepdims=True)
    normalized = np.clip(normalized, 0.0, None)  # guard float error on near-zero eigenvalues
    terms = np.where(normalized > 0.0, normalized * np.log(np.where(normalized > 0.0, normalized, 1.0)), 0.0)
    return -terms.sum(axis=-1)


def market_mode_share(corr: np.ndarray) -> np.ndarray | float:
    """Share of total variance carried by the largest eigenvalue: lambda_max / sum(lambda).

    The leading eigenvector of a financial correlation matrix is the market mode
    -- the direction in which all assets move together -- and its eigenvalue is
    how much of the total variation that single direction explains. Since
    sum(lambda) = trace(C) = N for a correlation matrix, this is lambda_max / N.
    Bounded in [1/N, 1]: 1/N when C = I, 1 when every asset moves identically.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(corr, dtype=np.float64))
    return eigenvalues[..., -1] / eigenvalues.sum(axis=-1)


def eigs_outside_mp(corr: np.ndarray, q: float) -> np.ndarray | float:
    """Number of eigenvalues of C above the Marchenko-Pastur upper edge (1 + sqrt(q))^2.

    With q = N / T_w, random-matrix theory says the spectrum of a correlation
    matrix estimated from *independent* series fills a bulk bounded above by
    (1 + sqrt(q))^2 -- 2.25 for this study's q = 15/60 = 0.25. Eigenvalues past
    that edge cannot be explained by estimation noise alone and are the
    candidates for genuine collective structure. This is the analysis of
    laloux1999noise, cited in the thesis.

    `q` is supplied by the caller rather than derived here: the function sees a
    matrix, not the sample length that produced it, and silently assuming a
    window would put a magic number in the wrong place.

    Note this is the count only, and on this study's data it is almost constant
    (exactly 1 in 1939 of 1947 windows). That stability is itself the Laloux
    result -- only the market mode escapes the bulk -- and belongs in Section
    6.6 as a distributional statement, not as a time series.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(corr, dtype=np.float64))
    upper_edge = (1.0 + np.sqrt(q)) ** 2
    return (eigenvalues > upper_edge).sum(axis=-1)
