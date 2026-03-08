from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LayerSliceBatch:
    """Fixed-shape detached layer slices sent to the sidecar."""

    batch_size: int
    slices: Dict[str, np.ndarray]
    # Shape convention for each value: [batch, time, dim]


@dataclass
class SidecarScores:
    """Adict/ANE sidecar outputs for routing and auxiliary losses."""

    probe_scores: np.ndarray
    conflict_score: np.ndarray
    capability_scores: np.ndarray
    sae_salience: np.ndarray


@dataclass
class RoutingOutput:
    """TinyLora routing weights and regularization stats."""

    route_weights: np.ndarray
    entropy_mean: float
    top1_usage_rate: float
    mode_usage_rate: float

