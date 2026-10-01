from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .model import ResidualEdit


def interchange_factory(
    site: int,
    basis: np.ndarray,
    positions: np.ndarray,
    source_resid: np.ndarray,
    device: str | torch.device,
) -> Callable[[np.ndarray], ResidualEdit]:
    """Subspace interchange at one residual site and one position per episode.

    For each episode row, the component of the residual inside span(`basis`)
    at `positions[row]` is replaced by that of `source_resid[row]`; everything
    orthogonal to the subspace is left as it was. `basis` is orthonormal
    [d_model, k]. The result plugs into `model_outputs(edit_factory=...)`.
    """
    q = torch.as_tensor(basis, dtype=torch.float32, device=device)

    def factory(rows: np.ndarray) -> ResidualEdit:
        pos = torch.as_tensor(positions[rows].astype(np.int64), device=device)
        src = torch.as_tensor(source_resid[rows], dtype=torch.float32, device=device)

        def edit(s: int, x: torch.Tensor) -> torch.Tensor:
            if s != site:
                return x
            b = torch.arange(x.shape[0], device=x.device)
            base = x[b, pos].float()
            swapped = base + ((src - base) @ q) @ q.T
            x = x.clone()
            x[b, pos] = swapped.to(x.dtype)
            return x

        return edit

    return factory


def pca_basis(acts: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal directions of the activations, [d_model, k]."""
    centred = acts.astype(np.float64) - acts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:k].T


def random_subspace(within: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random k-dimensional orthonormal basis inside span(`within`)."""
    q, _ = np.linalg.qr(rng.normal(size=(within.shape[1], k)))
    return within @ q
