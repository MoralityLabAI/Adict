from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import RoutingOutput, SidecarScores


@dataclass
class HybridLossWeights:
    const_weight: float = 0.2
    route_weight: float = 0.05
    sae_weight: float = 0.02
    usage_balance_weight: float = 0.02
    feature_gap_weight: float = 0.12
    drift_weight: float = 0.05


def compute_constitutional_loss(sidecar: SidecarScores, const_targets: np.ndarray) -> float:
    if sidecar.probe_scores.size == 0 or const_targets.size == 0:
        return 0.0
    d = sidecar.probe_scores - const_targets
    return float(np.mean(np.sum(d * d, axis=1)))


def compute_route_loss(route: RoutingOutput, adapter_count: int) -> float:
    if adapter_count <= 1:
        return 0.0
    # Encourage non-collapsed usage.
    norm_entropy = route.entropy_mean / np.log(max(2, adapter_count))
    return float(max(0.0, 1.0 - norm_entropy))


def compute_usage_balance_loss(route: RoutingOutput, target_mode_max: float = 0.7) -> float:
    return float(max(0.0, route.mode_usage_rate - target_mode_max))


def compute_sae_loss(sidecar: SidecarScores, target_sparsity: float = 0.1) -> float:
    if sidecar.sae_salience.size == 0:
        return 0.0
    act = sidecar.sae_salience
    active_frac = float(np.mean(act > 1e-3))
    return float(abs(active_frac - target_sparsity))


def compute_feature_gap_loss(sidecar: SidecarScores, target_margin: float = 0.2) -> float:
    if sidecar.target_control_gap.size == 0:
        return 0.0
    margin = sidecar.target_control_gap - target_margin
    return float(np.mean(np.square(np.minimum(margin, 0.0))))


def compute_drift_regularization(sidecar: SidecarScores, max_drift: float = 0.35) -> float:
    if sidecar.late_layer_drift.size == 0:
        return 0.0
    excess = np.maximum(sidecar.late_layer_drift - max_drift, 0.0)
    return float(np.mean(excess * excess))


def compute_hybrid_loss(
    task_loss: float,
    sidecar: SidecarScores,
    route: RoutingOutput,
    const_targets: np.ndarray,
    adapter_count: int,
    w: HybridLossWeights,
) -> dict:
    l_const = compute_constitutional_loss(sidecar, const_targets)
    l_route = compute_route_loss(route, adapter_count=adapter_count)
    l_sae = compute_sae_loss(sidecar)
    l_balance = compute_usage_balance_loss(route)
    l_gap = compute_feature_gap_loss(sidecar)
    l_drift = compute_drift_regularization(sidecar)
    total = (
        float(task_loss)
        + w.const_weight * l_const
        + w.route_weight * l_route
        + w.sae_weight * l_sae
        + w.usage_balance_weight * l_balance
        + w.feature_gap_weight * l_gap
        + w.drift_weight * l_drift
    )
    return {
        "task_loss": float(task_loss),
        "const_loss": float(l_const),
        "route_loss": float(l_route),
        "sae_loss": float(l_sae),
        "usage_balance_loss": float(l_balance),
        "feature_gap_loss": float(l_gap),
        "drift_loss": float(l_drift),
        "total_loss": float(total),
    }
