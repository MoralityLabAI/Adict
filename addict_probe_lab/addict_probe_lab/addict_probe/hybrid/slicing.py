from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .types import LayerSliceBatch


def _as_vector(row: dict, dim: int = 32) -> np.ndarray:
    h = row.get("hidden_state_summary", []) or []
    out = np.zeros((dim,), dtype=np.float32)
    if isinstance(h, list):
        vals = np.asarray(h[:dim], dtype=np.float32)
        out[: vals.shape[0]] = vals
    return out


def build_layer_slice_batch(
    rows: Iterable[dict],
    tap_layers: List[int],
    channel_ranges: Dict[int, tuple[int, int]],
    time_steps: int = 1,
    base_dim: int = 32,
) -> LayerSliceBatch:
    """
    Build fixed-shape slice tensors from trace rows.
    This is a detached proxy for real layer taps, suitable for sidecar scaffolding.
    """
    xs = [_as_vector(r, dim=base_dim) for r in rows]
    if not xs:
        return LayerSliceBatch(batch_size=0, slices={})
    x = np.stack(xs, axis=0)  # [batch, dim]
    slices: Dict[str, np.ndarray] = {}
    for layer in tap_layers:
        lo, hi = channel_ranges.get(layer, (0, base_dim))
        lo = max(0, min(lo, base_dim))
        hi = max(lo + 1, min(hi, base_dim))
        key = f"layer_{layer}"
        v = x[:, lo:hi]  # [batch, dim_slice]
        # Repeat across time dimension to keep fixed [batch,time,dim].
        vt = np.repeat(v[:, None, :], repeats=max(1, time_steps), axis=1)
        slices[key] = vt.astype(np.float32)
    return LayerSliceBatch(batch_size=int(x.shape[0]), slices=slices)

