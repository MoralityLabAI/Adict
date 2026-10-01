from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass
class RidgeProbe:
    """Ridge regression on standardised inputs; targets may have several columns."""

    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray  # [features, targets], in standardised input space
    b: np.ndarray
    lam: float

    @staticmethod
    def fit_path(x: np.ndarray, y: np.ndarray, grid: Sequence[float]) -> List["RidgeProbe"]:
        """One probe per penalty in `grid`, sharing the Gram matrix."""
        x = x.astype(np.float64)
        y = y.astype(np.float64).reshape(x.shape[0], -1)
        mean = x.mean(axis=0)
        scale = x.std(axis=0) + 1e-8
        xs = (x - mean) / scale
        b = y.mean(axis=0)
        gram = xs.T @ xs
        xty = xs.T @ (y - b)
        eye = np.eye(x.shape[1])
        return [
            RidgeProbe(mean=mean, scale=scale, w=np.linalg.solve(gram + lam * x.shape[0] * eye, xty), b=b, lam=lam)
            for lam in grid
        ]

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, lam: float) -> "RidgeProbe":
        return RidgeProbe.fit_path(x, y, [lam])[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float64) - self.mean) / self.scale) @ self.w + self.b

    def directions(self) -> np.ndarray:
        """Probe weights expressed in raw activation space, [features, targets]."""
        return self.w / self.scale[:, None]


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    """Variance-weighted R^2 over target columns."""
    y = y.astype(np.float64).reshape(pred.shape)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_selected(
    x: Dict[str, np.ndarray], y: Dict[str, np.ndarray], grid: Sequence[float] = RIDGE_GRID
) -> Tuple[RidgeProbe, Dict[str, float]]:
    """Fit on train, choose the penalty on val, report R^2 on all three splits."""
    path = RidgeProbe.fit_path(x["train"], y["train"], grid)
    probe = max(path, key=lambda p: r2_score(y["val"], p.predict(x["val"])))
    return probe, {name: r2_score(y[name], probe.predict(x[name])) for name in ("train", "val", "test")}


def fit_label_indexed(
    x: Dict[str, np.ndarray],
    token: Dict[str, np.ndarray],
    value: Dict[str, np.ndarray],
    grid: Sequence[float] = RIDGE_GRID,
) -> Tuple[Dict[int, RidgeProbe], Dict[str, float]]:
    """One probe per option label: reads that label's value wherever the label is on display.

    `token[split]` is [rows, slots] label token ids and `value[split]` the
    matching per-slot values. This tests whether a per-option variable is bound
    to the option's token identity instead of its displayed slot. One penalty is
    chosen for all labels on pooled val R^2; R^2 is pooled over (row, slot).
    """
    labels = [int(t) for t in np.unique(token["train"])]
    where = {s: {t: np.nonzero(token[s] == t) for t in labels} for s in token}
    paths = {}
    for t in labels:
        r, j = where["train"][t]
        paths[t] = RidgeProbe.fit_path(x["train"][r], value["train"][r, j], grid)

    def pooled(split: str, i: int) -> float:
        pred = np.full(value[split].shape, np.nan, dtype=np.float64)
        for t in labels:
            r, j = where[split][t]
            if r.size:
                pred[r, j] = paths[t][i].predict(x[split][r])[:, 0]
        seen = ~np.isnan(pred)
        return r2_score(value[split][seen], pred[seen])

    best = max(range(len(grid)), key=lambda i: pooled("val", i))
    probes = {t: paths[t][best] for t in labels}
    return probes, {s: pooled(s, best) for s in ("train", "val", "test")}


def covariance_directions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross-covariance of activations with each target column, [features, targets].

    The continuous analogue of a difference of class means: it points along
    high-variance activation directions that covary with the target, where
    regression weights are additionally whitened.
    """
    xc = x.astype(np.float64) - x.mean(axis=0)
    yc = y.astype(np.float64).reshape(x.shape[0], -1)
    yc = yc - yc.mean(axis=0)
    return xc.T @ yc / x.shape[0]


def permute_within_step(y: np.ndarray, episode: np.ndarray, step: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Control targets: each episode receives another episode's target trajectory.

    Step index is preserved, so a probe trained on these can still learn what
    position alone predicts; the R^2 it reaches on real held-out targets is the
    share of decodability owed to position.
    """
    out = y.copy()
    ids = np.unique(episode)
    donor = dict(zip(ids, rng.permutation(ids)))
    lookup = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(episode, step))}
    for i, (e, s) in enumerate(zip(episode, step)):
        j = lookup.get((int(donor[e]), int(s)))
        if j is not None:
            out[i] = y[j]
    return out


def orthonormal_basis(directions: np.ndarray) -> np.ndarray:
    """Orthonormal basis [features, k] for the span of the given direction columns."""
    q, _ = np.linalg.qr(directions.reshape(directions.shape[0], -1))
    return q
