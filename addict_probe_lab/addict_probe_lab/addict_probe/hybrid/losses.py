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
    total = (
        float(task_loss)
        + w.const_weight * l_const
        + w.route_weight * l_route
        + w.sae_weight * l_sae
        + w.usage_balance_weight * l_balance
    )
    return {
        "task_loss": float(task_loss),
        "const_loss": float(l_const),
        "route_loss": float(l_route),
        "sae_loss": float(l_sae),
        "usage_balance_loss": float(l_balance),
        "total_loss": float(total),
    }

