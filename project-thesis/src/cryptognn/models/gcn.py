"""The graph model of the study: a two-layer GCN over the dynamic correlation graph.

The architecture of Section 6.3, in plain PyTorch. With N = 15 nodes there is
nothing PyTorch Geometric could contribute: a layer is A_hat @ (H W) + b, which
is eq:gcn written out, and every quantity in it is a dense tensor small enough
that the whole walk-forward fits in seconds on a CPU. Writing it directly keeps
the code and the equation in the thesis the same object, and removes the one
installation risk of this project that had no upside.

Exports:
  - GCNLayer: one propagation step, A_hat @ (H W) + b
  - GCN2: the two layers of the study, dropout -> layer -> ReLU -> dropout -> layer
  - GCNForecaster: one trained model, conforming to the walk-forward protocol
  - GCNGridForecaster: the frozen grid and the seed average, as one forecaster
  - gcn_factories(): the two arms of the ablation, one factory each
  - seed_everything(): the determinism discipline, in one place

Integration: both forecasters implement cryptognn.evaluation.protocols.Forecaster
  and SupportsDiagnostics, so they run through run_walkforward() on the identical
  loop as the five baselines; scripts/05_run_gcn.py runs the grid one over both
  arms of the ablation. The tensors they read are the ones the harness already
  guarantees are causal -- Segment.a_hat and Segment.features -- so this module
  has no view of the panel and cannot reach past a prediction origin.
Why the grid runner lives here rather than in cryptognn.evaluation: it is
  experimental protocol and would sit naturally beside the harness, except that
  building a GCNForecaster imports torch and evaluation/ is deliberately kept
  free of it (see the note in evaluation/protocols.py). It stays next to the
  model it orchestrates, and the dependency still runs models -> evaluation.
Why it is not re-exported from cryptognn.models: that package's __init__ is
  imported by scripts/04_run_baselines.py, and listing GCN2 there would make
  running the baselines import torch. Callers write
  `from cryptognn.models.gcn import GCN2`, for the same reason the Forecaster
  protocol lives beside the harness rather than beside the models.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from cryptognn.config import Config
from cryptognn.features import FoldStandardizer

if TYPE_CHECKING:
    from cryptognn.evaluation.walkforward import Segment

# The task is one scalar per asset per prediction origin: r_hat_{t+1}. Not a
# hyperparameter -- it is the dimension of the question, so it is fixed here
# rather than read from the config.
OUT_FEATURES = 1

# Below this, a training block's returns are treated as having no spread at all
# and the target is left unscaled, rather than divided by something close to
# zero. It cannot happen on real returns; it can on a degenerate test fixture.
_TARGET_SCALE_FLOOR = 1e-12


def seed_everything(seed: int) -> np.random.Generator:
    """Pin every source of randomness this study uses, and return the NumPy one.

    Torch's global seed governs weight initialization and dropout masks; the
    returned Generator is for whatever the caller randomizes itself. Both are
    needed: a run reproducible in one and not the other is not reproducible.

    `use_deterministic_algorithms(True)` forbids the nondeterministic CUDA
    kernels torch would otherwise be free to pick. On this CPU-only study it
    changes nothing measurable, which is exactly why it is affordable to set --
    the cost of leaving it off is a result nobody can reproduce months later.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


class GCNLayer(nn.Module):
    """One graph convolution: H' = A_hat (H W) + b.

    The weight matrix is (F_in, F_out) and is **shared by every node**. That is
    the whole content of the architecture: a node's representation is built from
    its neighbours' features through the same transformation applied everywhere,
    which is what makes the layer permutation equivariant (Chapter 2) and what
    keeps the parameter count independent of N. A per-node weight would fit this
    data better and stop being a graph convolution.

    `A_hat` is the renormalized adjacency of graph.build.normalized_adjacency():
    self-loops added, symmetrically normalized, spectrum in [-1, 1]. This module
    takes it as given and never builds one, so the substrate the GCN trains on
    is the same artifact the topology of Section 6.6 is measured on.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Glorot uniform on the weight, zeros on the bias, as in Kipf & Welling.

        Kept as a method rather than inlined in __init__ so a caller can re-seed
        and re-initialize an existing module in place -- what the five-seed
        averaging of S4.3 repeats four times per configuration.
        """
        limit = math.sqrt(6.0 / (self.in_features + self.out_features))
        nn.init.uniform_(self.weight, -limit, limit)
        nn.init.zeros_(self.bias)

    def forward(self, a_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Propagate `h` (B, N, F_in) over `a_hat` (B, N, N) -> (B, N, F_out).

        The feature projection comes first and the propagation second. The two
        orders are algebraically identical -- matrix multiplication associates --
        but this one costs B N F_in F_out + B N^2 F_out instead of
        B N^2 F_in + B N F_in F_out, which is the cheaper way round whenever the
        hidden width is below the feature count, and never the more expensive.
        """
        if a_hat.ndim != 3 or h.ndim != 3:
            raise ValueError(f"Expected batched (B, N, N) and (B, N, F), got {tuple(a_hat.shape)} and {tuple(h.shape)}")
        if a_hat.shape[0] != h.shape[0] or a_hat.shape[-1] != h.shape[1]:
            raise ValueError(f"Adjacency {tuple(a_hat.shape)} does not match features {tuple(h.shape)}")
        if h.shape[-1] != self.in_features:
            raise ValueError(f"Layer takes {self.in_features} input features, got {h.shape[-1]}")

        return torch.bmm(a_hat, h @ self.weight) + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class GCN2(nn.Module):
    """The two-layer GCN of Section 6.3, and its no-graph ablation.

    Dropout -> GCNLayer(F, hidden) -> ReLU -> Dropout -> GCNLayer(hidden, 1),
    which is eq:gcn applied twice: two propagation steps, so a node's forecast
    draws on its neighbours and on its neighbours' neighbours, and no further.

    `hidden` and `dropout` have no defaults on purpose. They are the two axes of
    the grid frozen in Sprint 1 (config.model.gcn.hidden x dropout, four
    configurations), and a default here would be a magic number quietly
    competing with the config for the role of source of truth.

    **use_graph=False is the ablation of the first research question.** It
    replaces A_hat with the identity inside forward(), which turns the model into
    a per-node MLP with the *same parameters and the same features* -- the only
    difference between the two runs is whether information crosses between
    assets. Substituting inside rather than asking the caller to pass an identity
    keeps both arms of the ablation on one call site, so they cannot drift apart.

    Output is (B, N): one forecast per asset per prediction origin, the shape the
    walk-forward harness scores.
    """

    def __init__(self, in_features: int, hidden: int, dropout: float, use_graph: bool = True) -> None:
        super().__init__()
        if in_features < 1 or hidden < 1:
            raise ValueError(f"in_features and hidden must be positive, got {in_features} and {hidden}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")

        self.use_graph = use_graph
        self.dropout = nn.Dropout(dropout)
        self.layer1 = GCNLayer(in_features, hidden)
        self.layer2 = GCNLayer(hidden, OUT_FEATURES)

    def forward(self, a_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forecast (B, N) from the graph (B, N, N) and node features (B, N, F).

        `a_hat` is still required with use_graph=False, and still has to match the
        shape of the features. The ablation is meant to answer "does the graph
        help?", and a signature that let the caller omit it would let the two arms
        be fed different data without anyone noticing.
        """
        if not self.use_graph:
            a_hat = torch.eye(h.shape[1], dtype=h.dtype, device=h.device).expand_as(a_hat)

        hidden = torch.relu(self.layer1(a_hat, self.dropout(h)))
        return self.layer2(a_hat, self.dropout(hidden)).squeeze(-1)

    def n_parameters(self) -> int:
        """Total trainable parameters -- identical with and without the graph.

        Equal capacity is what makes the ablation an isolation of the graph
        rather than a comparison of two model sizes, so the number is exposed
        for the run diagnostics of S4.2 to record.
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def extra_repr(self) -> str:
        return f"use_graph={self.use_graph}"


class GCNForecaster:
    """A GCN2 trained on one walk-forward fold, under the harness's contract.

    Adam on the mean squared error, early stopping on the validation block, and
    the best weights restored before fit() returns. Everything the training loop
    needs beyond the two grid axes comes from config.model.gcn, so a run is
    described by the config hash plus (hidden, dropout, use_graph, seed).

    Three decisions worth stating, because each has a plausible alternative:

      - **The target is divided by one scalar**, the pooled standard deviation of
        the training block's returns, and the forecast is multiplied back by it.
        A single number rather than one per asset: dividing by a scalar rescales
        the whole loss by 1/s^2 and leaves the objective otherwise untouched, so
        the model is still optimizing the pooled RMSE that Section 6.4 reports,
        whereas per-asset standardization would silently reweight the assets and
        optimize something else. It is not cosmetic. Daily returns are O(0.04)
        while a Glorot-initialized network outputs O(0.5), so on raw targets the
        first hundred epochs are spent shrinking the output rather than fitting
        it: at the config's 300-epoch budget the training error had only just
        come back down to the level of forecasting zero. Every MSE reported by
        diagnostics() is converted back to return units.
      - **Training is full batch.** The frozen config has no batch size, and a
        fold is a (365, 15, 15) adjacency with a (365, 15, 8) feature block --
        a few hundred kB. One epoch is one Adam step, and dropout is the only
        stochasticity, which the seed pins.
      - **The seed is set inside fit(), not in the constructor**, so the model
        fitted on fold f does not depend on how many folds were fitted before
        it. Re-running a single fold reproduces the number the full run gave.

    The features are standardized by a FoldStandardizer fitted on the training
    split alone -- the look-ahead Section 5.4 calls the most insidious, since
    nothing fails when it is present.
    """

    def __init__(
        self,
        config: Config,
        hidden: int,
        dropout: float,
        *,
        use_graph: bool = True,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        settings = config.model.gcn
        self.hidden = hidden
        self.dropout = dropout
        self.use_graph = use_graph
        self.seed = config.seed if seed is None else seed
        self.lr = settings.lr
        self.weight_decay = settings.weight_decay
        self.epochs = settings.epochs
        self.patience = settings.patience
        self.name = name if name is not None else ("gcn" if use_graph else "gcn-nograph")

        self.model_: GCN2 | None = None
        self.standardizer_: FoldStandardizer | None = None
        self.target_scale_: float = 1.0
        self.history_: list[dict[str, float]] = []
        self.epochs_run_: int = 0
        self.best_epoch_: int = 0
        self.best_train_mse_: float = float("nan")
        self.best_val_mse_: float = float("nan")

    # ----------------------------------------------------------------------
    # Fitting
    # ----------------------------------------------------------------------

    @staticmethod
    def _require_inputs(segment: Segment, split: str) -> None:
        """Reject a segment the GCN cannot read, naming what is missing.

        This is the one model of the study that needs both the graph and the node
        features; the baselines run on returns alone. A caller who assembled the
        container without them gets told which, instead of an AttributeError on
        None several frames further in.
        """
        missing = [name for name in ("features", "a_hat") if getattr(segment, name) is None]
        if missing:
            raise ValueError(f"GCNForecaster needs {' and '.join(missing)} on the {split} segment, which carries none")

    def _tensors(self, segment: Segment) -> tuple[torch.Tensor, torch.Tensor]:
        """The segment's graph and standardized features as float32 tensors.

        float32 throughout: the graph tensors are stored float32 on disk anyway,
        and the extra precision buys nothing a gradient step would notice. One
        conversion path for train, validation and test, so the three cannot be
        preprocessed differently.
        """
        if self.standardizer_ is None:
            raise RuntimeError("GCNForecaster used before fit()")
        a_hat = torch.from_numpy(np.ascontiguousarray(segment.a_hat, dtype=np.float32))
        features = torch.from_numpy(self.standardizer_.transform(segment.features).astype(np.float32))
        return a_hat, features

    def fit(self, train: Segment, val: Segment) -> None:
        self._require_inputs(train, "train")
        self._require_inputs(val, "validation")

        seed_everything(self.seed)
        self.standardizer_ = FoldStandardizer().fit(train.features)

        # One scalar for the whole panel, from the training block only. Estimated
        # here rather than taken from FoldStandardizer, which works per asset and
        # per channel and would reweight the loss across assets.
        scale = float(np.std(train.y))
        self.target_scale_ = scale if scale > _TARGET_SCALE_FLOOR else 1.0

        train_a, train_x = self._tensors(train)
        val_a, val_x = self._tensors(val)
        train_y = torch.from_numpy((train.y / self.target_scale_).astype(np.float32))
        val_y = torch.from_numpy((val.y / self.target_scale_).astype(np.float32))

        model = GCN2(train_x.shape[-1], hidden=self.hidden, dropout=self.dropout, use_graph=self.use_graph)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        best_train = float("nan")
        best_epoch = 0
        waited = 0
        self.history_ = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            criterion(model(train_a, train_x), train_y).backward()
            optimizer.step()

            # Both splits are scored in eval mode: the training loss measured
            # through dropout is a noisier quantity than the validation loss, and
            # the two would not be comparable in the per-fold log.
            # Reported in return units, not in the scaled ones the loss uses, so
            # the per-fold log is directly comparable with the RMSE of Section 6.4
            # and with the other models' errors.
            in_returns = self.target_scale_**2
            model.eval()
            with torch.no_grad():
                train_mse = float(criterion(model(train_a, train_x), train_y)) * in_returns
                val_mse = float(criterion(model(val_a, val_x), val_y)) * in_returns
            self.history_.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

            self.epochs_run_ = epoch
            if val_mse < best_val:
                best_val, best_train, best_epoch = val_mse, train_mse, epoch
                best_state = copy.deepcopy(model.state_dict())
                waited = 0
            else:
                waited += 1
                if waited >= self.patience:
                    break

        # The restore is the whole point of early stopping: without it the model
        # kept is the one from `patience` epochs past its best, which is strictly
        # worse and indistinguishable from the correct behaviour in any log.
        model.load_state_dict(best_state)
        self.model_ = model
        self.best_epoch_, self.best_train_mse_, self.best_val_mse_ = best_epoch, best_train, best_val

    # ----------------------------------------------------------------------
    # Prediction and reporting
    # ----------------------------------------------------------------------

    def predict(self, segment: Segment) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("GCNForecaster.predict() called before fit()")
        self._require_inputs(segment, "prediction")

        a_hat, features = self._tensors(segment)
        self.model_.eval()
        with torch.no_grad():
            scaled = self.model_(a_hat, features).numpy().astype(np.float64)
        return scaled * self.target_scale_

    def diagnostics(self) -> dict[str, float | int | str]:
        """The per-fold training record of S4.2: how long it trained, and to what.

        Reported through the harness's diagnostics hook rather than written to a
        file of its own, so it lands in the same per-fold frame as the VAR's lag
        order and reaches disk through artifacts.save_run_diagnostics().
        """
        if self.model_ is None:
            raise RuntimeError("GCNForecaster.diagnostics() called before fit()")
        return {
            "epochs_run": self.epochs_run_,
            "best_epoch": self.best_epoch_,
            "train_mse": self.best_train_mse_,
            "val_mse": self.best_val_mse_,
            "early_stopped": int(self.epochs_run_ < self.epochs),
            "n_params": self.model_.n_parameters(),
            "hidden": self.hidden,
            "gcn_dropout": self.dropout,
            "seed": self.seed,
            "use_graph": int(self.use_graph),
        }


class GCNGridForecaster:
    """The frozen grid and the seed average, wrapped as a single forecaster.

    What Section 6.5 actually runs. Per fold: every (hidden, dropout) of the grid
    frozen in Sprint 1, each fitted from every seed of config.model.gcn.seeds;
    the configuration is chosen on the fold's validation block, and the test
    forecast is the average of that configuration's seeds.

    **The validation block does two jobs and the test block does neither.** It
    early-stops each individual fit, and it chooses between configurations. That
    is what a three-way split is for: the test block is untouched by both
    decisions, so it remains an estimate of out-of-sample error rather than of
    how well the selection was tuned.

    **The selection criterion is the validation error of the seed average**, not
    the average of the seeds' individual validation errors. The quantity selected
    on is then exactly the quantity used on test, so the procedure describes
    itself in one sentence; scoring the seeds separately would rank
    configurations by how a single training run behaves and then deploy something
    else. Averaging seeds at all is a variance reduction over the initialization,
    and is declared as part of the protocol rather than treated as an
    implementation detail.

    The grid is read from the config and is not a constructor argument: it was
    frozen before any result was seen, and an argument here would be a way to
    reopen it from the outside without touching config/default.yaml.
    """

    def __init__(self, config: Config, *, use_graph: bool = True, name: str | None = None) -> None:
        settings = config.model.gcn
        self.config = config
        self.use_graph = use_graph
        self.hidden_grid = list(settings.hidden)
        self.dropout_grid = list(settings.dropout)
        self.seeds = list(settings.seeds)
        self.name = name if name is not None else ("gcn" if use_graph else "gcn-nograph")

        self.grid_: list[dict[str, float]] = []
        self.selected_: dict[str, float] | None = None
        self.selected_models_: list[GCNForecaster] = []

    @property
    def n_fits(self) -> int:
        return len(self.hidden_grid) * len(self.dropout_grid) * len(self.seeds)

    def fit(self, train: Segment, val: Segment) -> None:
        self.grid_ = []
        best: tuple[float, dict[str, float], list[GCNForecaster]] | None = None
        blind_val = val.without_target()

        for hidden in self.hidden_grid:
            for dropout in self.dropout_grid:
                models = []
                for seed in self.seeds:
                    model = GCNForecaster(
                        self.config, hidden=hidden, dropout=dropout, use_graph=self.use_graph, seed=seed
                    )
                    model.fit(train, val)
                    models.append(model)

                # Scored on the ensemble, because the ensemble is what test gets.
                # The target is withheld from predict() here exactly as the
                # harness withholds it, so this path cannot read what the
                # equivalent path on the test block could not.
                ensemble = np.mean([model.predict(blind_val) for model in models], axis=0)
                record = {
                    "hidden": hidden,
                    "dropout": dropout,
                    "val_mse": float(np.mean((val.y - ensemble) ** 2)),
                    "epochs_mean": float(np.mean([model.epochs_run_ for model in models])),
                    "early_stopped_share": float(
                        np.mean([model.diagnostics()["early_stopped"] for model in models])
                    ),
                    "n_seeds": len(models),
                }
                self.grid_.append(record)

                if best is None or record["val_mse"] < best[0]:
                    best = (record["val_mse"], record, models)

        if best is None:
            raise ValueError(
                f"Empty grid: config.model.gcn has {len(self.hidden_grid)} hidden sizes, "
                f"{len(self.dropout_grid)} dropout rates and {len(self.seeds)} seeds"
            )
        _, self.selected_, self.selected_models_ = best

    def predict(self, segment: Segment) -> np.ndarray:
        if not self.selected_models_:
            raise RuntimeError("GCNGridForecaster.predict() called before fit()")
        return np.mean([model.predict(segment) for model in self.selected_models_], axis=0)

    def diagnostics(self) -> dict[str, float | int | str]:
        """What the grid decided on this fold, and how close the runners-up were.

        The per-configuration validation errors are reported alongside the winner
        because "which configuration was selected" is only half the finding: if
        the four are indistinguishable, the grid did nothing and Section 6.5 has
        to say so. That is not recoverable after the fact from the winner alone.
        """
        if self.selected_ is None:
            raise RuntimeError("GCNGridForecaster.diagnostics() called before fit()")

        row: dict[str, float | int | str] = {
            "selected_hidden": int(self.selected_["hidden"]),
            "selected_dropout": float(self.selected_["dropout"]),
            "val_mse": self.selected_["val_mse"],
            "epochs_mean": self.selected_["epochs_mean"],
            "early_stopped_share": self.selected_["early_stopped_share"],
            "n_params": self.selected_models_[0].model_.n_parameters(),
            "n_fits": self.n_fits,
            "use_graph": int(self.use_graph),
        }
        for record in self.grid_:
            row[f"val_mse_h{int(record['hidden'])}_d{record['dropout']}"] = record["val_mse"]
        return row


def gcn_factories(config: Config) -> dict[str, Callable[[], GCNGridForecaster]]:
    """The two arms of Section 6.5, in the order they are reported.

    The counterpart of models.baseline_factories(), and it lives here rather than
    beside it for the reason stated at the top of this module: models/__init__.py
    is imported by scripts/04_run_baselines.py, and a GCN entry there would make
    running the baselines import torch.

    `use_graph` is bound as a default argument rather than captured by the
    closure. The difference is invisible today, because run_walkforward() calls
    each factory within the iteration that built it -- but a caller who collected
    these factories and ran them later would get the last arm twice, and the two
    arms of an ablation coming out identical is the one failure this comparison
    cannot afford to produce quietly.
    """
    return {
        name: lambda use_graph=use_graph: GCNGridForecaster(config, use_graph=use_graph)
        for name, use_graph in (("gcn", True), ("gcn-nograph", False))
    }
