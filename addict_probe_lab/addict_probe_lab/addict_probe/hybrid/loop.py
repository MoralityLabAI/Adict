from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..trace_schema import read_trace_jsonl
from .adapter_bank import TinyLoraAdapterBank
from .losses import HybridLossWeights, compute_hybrid_loss
from .routing import RoutingMLP
from .sidecar import ANESidecarProbeRunner
from .slicing import build_layer_slice_batch


@dataclass
class HybridLoopConfig:
    trace_path: str
    batch_size: int = 32
    steps: int = 50
    tap_layers: List[int] = None  # type: ignore[assignment]
    channel_ranges: Dict[int, tuple[int, int]] = None  # type: ignore[assignment]
    adapter_names: List[str] = None  # type: ignore[assignment]
    route_hidden: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        if self.tap_layers is None:
            self.tap_layers = [16, 24]
        if self.channel_ranges is None:
            self.channel_ranges = {16: (0, 16), 24: (8, 32)}
        if self.adapter_names is None:
            self.adapter_names = ["general_storyworld", "moral_conflict", "powret_helper"]


def _batch_indices(n: int, bsz: int, step: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    if bsz <= 0:
        return np.zeros((0,), dtype=np.int32)
    start = (step * bsz) % n
    # Modulo-wrapped gather handles bsz > n without producing out-of-range indices.
    return ((start + np.arange(bsz, dtype=np.int32)) % n).astype(np.int32)


def _task_loss_proxy(adapter_mix: np.ndarray, sidecar_probe: np.ndarray) -> float:
    if adapter_mix.size == 0 or sidecar_probe.size == 0:
        return 0.0
    # Proxy objective: mixed adapter should correlate with constitutional probe mean.
    target = np.mean(sidecar_probe, axis=1, keepdims=True)
    pred = np.mean(adapter_mix, axis=1, keepdims=True)
    return float(np.mean((pred - target) ** 2))


def run_hybrid_calibration_loop(cfg: HybridLoopConfig) -> dict:
    rows = read_trace_jsonl(cfg.trace_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"trace has no rows: {cfg.trace_path}")
    rng = np.random.default_rng(cfg.seed)

    sidecar = ANESidecarProbeRunner(seed=cfg.seed)
    bank = TinyLoraAdapterBank(adapter_names=cfg.adapter_names, adapter_dim=8, seed=cfg.seed)
    # Input is probe + conflict + capability summaries.
    router_input_dim = sidecar.probe_dim + 1 + sidecar.capability_dim
    router = RoutingMLP(
        input_dim=router_input_dim,
        adapter_count=bank.count,
        hidden_dim=cfg.route_hidden,
        seed=cfg.seed,
    )
    w = HybridLossWeights()

    step_rows = []
    for step in range(cfg.steps):
        idx = _batch_indices(n=n, bsz=cfg.batch_size, step=step)
        batch_rows = [rows[int(i)] for i in idx.tolist()]
        slice_batch = build_layer_slice_batch(
            batch_rows,
            tap_layers=cfg.tap_layers,
            channel_ranges=cfg.channel_ranges,
            time_steps=1,
            base_dim=32,
        )
        side = sidecar.score(slice_batch)
        rv = np.concatenate([side.probe_scores, side.conflict_score, side.capability_scores], axis=1)
        route = router.route(rv)
        mix = bank.compose(route.route_weights)

        # Sample target constitutional profiles for scaffold runs.
        const_targets = rng.uniform(0.2, 0.85, size=side.probe_scores.shape).astype(np.float32)
        task_loss = _task_loss_proxy(mix, side.probe_scores)
        loss = compute_hybrid_loss(
            task_loss=task_loss,
            sidecar=side,
            route=route,
            const_targets=const_targets,
            adapter_count=bank.count,
            w=w,
        )
        step_rows.append(
            {
                "step": int(step),
                **loss,
                "route_entropy_mean": float(route.entropy_mean),
                "route_top1_usage_rate": float(route.top1_usage_rate),
                "route_mode_usage_rate": float(route.mode_usage_rate),
                "conflict_score_mean": float(np.mean(side.conflict_score)) if side.conflict_score.size else 0.0,
                "probe_mean": float(np.mean(side.probe_scores)) if side.probe_scores.size else 0.0,
            }
        )

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in step_rows] if step_rows else [0.0]
        return float(np.mean(vals))

    return {
        "trace_path": cfg.trace_path,
        "rows_in_trace": int(n),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "tap_layers": cfg.tap_layers,
        "adapter_names": cfg.adapter_names,
        "summary": {
            "total_loss_mean": _mean("total_loss"),
            "task_loss_mean": _mean("task_loss"),
            "const_loss_mean": _mean("const_loss"),
            "route_loss_mean": _mean("route_loss"),
            "sae_loss_mean": _mean("sae_loss"),
            "route_entropy_mean": _mean("route_entropy_mean"),
            "route_mode_usage_rate_mean": _mean("route_mode_usage_rate"),
            "probe_mean": _mean("probe_mean"),
        },
        "steps_detail": step_rows,
    }
