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
    target_feature_ids: List[int] = None  # type: ignore[assignment]
    control_feature_ids: List[int] = None  # type: ignore[assignment]
    late_layer_floor: int = 20
    seed: int = 7

    def __post_init__(self) -> None:
        if self.tap_layers is None:
            self.tap_layers = [16, 24]
        if self.channel_ranges is None:
            self.channel_ranges = {16: (0, 16), 24: (8, 32)}
        if self.adapter_names is None:
            self.adapter_names = ["general_storyworld", "moral_conflict", "powret_helper"]
        if self.target_feature_ids is None:
            self.target_feature_ids = [5, 12]
        if self.control_feature_ids is None:
            self.control_feature_ids = [18, 25]


def _batch_indices(n: int, bsz: int, step: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    if bsz <= 0:
        return np.zeros((0,), dtype=np.int32)
    start = (step * bsz) % n
    # Modulo-wrapped gather handles bsz > n without producing out-of-range indices.
    return ((start + np.arange(bsz, dtype=np.int32)) % n).astype(np.int32)


def _task_loss_proxy(adapter_mix: np.ndarray, target_signal: np.ndarray, target_outcome: np.ndarray) -> float:
    if adapter_mix.size == 0 or target_signal.size == 0 or target_outcome.size == 0:
        return 0.0
    pred = np.mean(adapter_mix, axis=1, keepdims=True)
    desired = 0.6 * target_signal + 0.4 * target_outcome
    return float(np.mean((pred - desired) ** 2))


def _row_targets(batch_rows: List[dict], probe_dim: int) -> tuple[np.ndarray, np.ndarray]:
    const_targets = []
    outcome_targets = []
    for row in batch_rows:
        moral = float(row.get("moral_score_proxy", 0.0))
        useful = float(row.get("usefulness_score_proxy", 0.0))
        consistent = float(row.get("consistency_score_proxy", 0.0))
        manip = float(row.get("manipulation_risk_proxy", 0.0))
        progress = float(row.get("story_goal_progress_proxy", 0.0))
        intervention = float(bool(row.get("intervention_needed", False)))
        reasoning_text = str(row.get("reasoning_summary", "")).lower()
        think_bias = 1.0 if "moral_logic" in reasoning_text or "principle" in reasoning_text else 0.0
        row_target = np.array(
            [
                moral,
                consistent,
                useful,
                1.0 - manip,
                progress,
                0.5 * moral + 0.25 * consistent + 0.25 * think_bias,
            ],
            dtype=np.float32,
        )
        if probe_dim > row_target.shape[0]:
            pad = np.full((probe_dim - row_target.shape[0],), row_target[-1], dtype=np.float32)
            row_target = np.concatenate([row_target, pad], axis=0)
        const_targets.append(row_target[:probe_dim])
        outcome_targets.append([0.5 * moral + 0.3 * useful + 0.2 * consistent - 0.25 * manip - 0.15 * intervention])
    return np.asarray(const_targets, dtype=np.float32), np.asarray(outcome_targets, dtype=np.float32)


def run_hybrid_calibration_loop(cfg: HybridLoopConfig) -> dict:
    rows = read_trace_jsonl(cfg.trace_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"trace has no rows: {cfg.trace_path}")
    layer_weight_by_key = {f"layer_{layer}": (1.5 if layer >= cfg.late_layer_floor else 0.75) for layer in cfg.tap_layers}
    sidecar = ANESidecarProbeRunner(
        seed=cfg.seed,
        target_feature_ids=tuple(cfg.target_feature_ids),
        control_feature_ids=tuple(cfg.control_feature_ids),
        late_layer_floor=cfg.late_layer_floor,
        layer_weight_by_key=layer_weight_by_key,
    )
    bank = TinyLoraAdapterBank(adapter_names=cfg.adapter_names, adapter_dim=8, seed=cfg.seed)
    # Input emphasizes late-layer reweighting evidence, then probe/capability summaries.
    router_input_dim = 1 + 1 + 1 + sidecar.probe_dim + 1 + sidecar.capability_dim
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
        rv = np.concatenate(
            [
                side.target_feature_strength,
                side.control_feature_strength,
                side.target_control_gap,
                side.probe_scores,
                side.late_layer_drift,
                side.capability_scores,
            ],
            axis=1,
        )
        route = router.route(rv)
        mix = bank.compose(route.route_weights)

        const_targets, outcome_targets = _row_targets(batch_rows, probe_dim=side.probe_scores.shape[1])
        task_loss = _task_loss_proxy(mix, side.target_feature_strength + side.target_control_gap, outcome_targets)
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
                "steering_strength_mean": float(route.steering_strength_mean),
                "conflict_score_mean": float(np.mean(side.conflict_score)) if side.conflict_score.size else 0.0,
                "probe_mean": float(np.mean(side.probe_scores)) if side.probe_scores.size else 0.0,
                "target_feature_mean": float(np.mean(side.target_feature_strength)) if side.target_feature_strength.size else 0.0,
                "control_feature_mean": float(np.mean(side.control_feature_strength)) if side.control_feature_strength.size else 0.0,
                "target_control_gap_mean": float(np.mean(side.target_control_gap)) if side.target_control_gap.size else 0.0,
                "late_layer_drift_mean": float(np.mean(side.late_layer_drift)) if side.late_layer_drift.size else 0.0,
                "constitutional_target_mean": float(np.mean(const_targets)) if const_targets.size else 0.0,
                "outcome_target_mean": float(np.mean(outcome_targets)) if outcome_targets.size else 0.0,
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
        "target_feature_ids": cfg.target_feature_ids,
        "control_feature_ids": cfg.control_feature_ids,
        "summary": {
            "total_loss_mean": _mean("total_loss"),
            "task_loss_mean": _mean("task_loss"),
            "const_loss_mean": _mean("const_loss"),
            "route_loss_mean": _mean("route_loss"),
            "sae_loss_mean": _mean("sae_loss"),
            "feature_gap_loss_mean": _mean("feature_gap_loss"),
            "drift_loss_mean": _mean("drift_loss"),
            "route_entropy_mean": _mean("route_entropy_mean"),
            "route_mode_usage_rate_mean": _mean("route_mode_usage_rate"),
            "steering_strength_mean": _mean("steering_strength_mean"),
            "probe_mean": _mean("probe_mean"),
            "target_feature_mean": _mean("target_feature_mean"),
            "control_feature_mean": _mean("control_feature_mean"),
            "target_control_gap_mean": _mean("target_control_gap_mean"),
            "late_layer_drift_mean": _mean("late_layer_drift_mean"),
            "constitutional_target_mean": _mean("constitutional_target_mean"),
            "outcome_target_mean": _mean("outcome_target_mean"),
        },
        "steps_detail": step_rows,
    }
