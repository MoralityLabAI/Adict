from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import numpy as np

from .types import LayerSliceBatch, SidecarScores


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _layer_num(key: str) -> int:
    try:
        return int(key.split("_")[-1])
    except (ValueError, IndexError):
        return 0


@dataclass
class ANESidecarProbeRunner:
    """
    Fixed-shape sidecar that approximates late-layer constitutional reweighting.

    The sidecar keeps a deterministic fixed-base dictionary per layer, projects pooled
    slice activations into feature space, and reports target/control feature strengths.
    """

    probe_dim: int = 6
    capability_dim: int = 3
    sae_dim: int = 32
    layer_feature_dim: int = 32
    late_layer_floor: int = 20
    target_feature_ids: tuple[int, ...] = (5, 12)
    control_feature_ids: tuple[int, ...] = (18, 25)
    seed: int = 7
    layer_weight_by_key: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._weights: Dict[str, Dict[str, np.ndarray]] = {}

    def _init_layer(self, key: str, in_dim: int) -> None:
        feature_dim = min(self.layer_feature_dim, in_dim)
        feature_w = self.rng.normal(0.0, 0.12, size=(in_dim, feature_dim)).astype(np.float32)
        q, _ = np.linalg.qr(feature_w)
        feature_w = q[:, :feature_dim].astype(np.float32)
        self._weights[key] = {
            "feature_w": feature_w,
            "probe_w": self.rng.normal(0.0, 0.08, size=(self.layer_feature_dim, self.probe_dim)).astype(np.float32),
            "probe_b": np.zeros((self.probe_dim,), dtype=np.float32),
            "cap_w": self.rng.normal(0.0, 0.08, size=(self.layer_feature_dim, self.capability_dim)).astype(np.float32),
            "cap_b": np.zeros((self.capability_dim,), dtype=np.float32),
            "conf_w": self.rng.normal(0.0, 0.08, size=(self.layer_feature_dim, 1)).astype(np.float32),
            "conf_b": np.zeros((1,), dtype=np.float32),
        }

    def _layer_weight(self, key: str) -> float:
        if key in self.layer_weight_by_key:
            return float(self.layer_weight_by_key[key])
        layer_num = _layer_num(key)
        return 1.5 if layer_num >= self.late_layer_floor else 0.75

    def _feature_summary(self, feature_space: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = feature_space.shape[1]
        target_idx = [int(i) % width for i in self.target_feature_ids]
        control_idx = [int(i) % width for i in self.control_feature_ids]
        target = np.mean(feature_space[:, target_idx], axis=1, keepdims=True).astype(np.float32)
        control = np.mean(feature_space[:, control_idx], axis=1, keepdims=True).astype(np.float32)
        gap = (target - control).astype(np.float32)
        return target, control, gap

    def _late_layer_baseline(self, weighted_features: Iterable[np.ndarray], weighted_values: Iterable[float]) -> np.ndarray:
        features = list(weighted_features)
        weights = list(weighted_values)
        if not features:
            return np.zeros((0, self.layer_feature_dim), dtype=np.float32)
        stacked = np.stack(features, axis=0)
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
        return (np.sum(stacked * w, axis=0) / np.clip(np.sum(w), 1e-6, None)).astype(np.float32)

    def score(self, batch: LayerSliceBatch) -> SidecarScores:
        if batch.batch_size == 0 or not batch.slices:
            z_probe = np.zeros((0, self.probe_dim), dtype=np.float32)
            z_cap = np.zeros((0, self.capability_dim), dtype=np.float32)
            z_sae = np.zeros((0, self.sae_dim), dtype=np.float32)
            z_scalar = np.zeros((0, 1), dtype=np.float32)
            return SidecarScores(
                probe_scores=z_probe,
                conflict_score=z_scalar,
                capability_scores=z_cap,
                sae_salience=z_sae,
                target_feature_strength=z_scalar,
                control_feature_strength=z_scalar,
                target_control_gap=z_scalar,
                late_layer_drift=z_scalar,
                layer_feature_strengths={},
            )

        probe_acc = []
        cap_acc = []
        conf_acc = []
        layer_feature_strengths: Dict[str, np.ndarray] = {}
        target_acc = []
        control_acc = []
        gap_acc = []
        weighted_features = []
        weighted_values = []

        for key, x in batch.slices.items():
            pooled = np.mean(x, axis=1).astype(np.float32)
            in_dim = int(pooled.shape[1])
            if key not in self._weights:
                self._init_layer(key, in_dim)
            w = self._weights[key]
            feature_space_raw = np.abs(pooled @ w["feature_w"]).astype(np.float32)
            target, control, gap = self._feature_summary(feature_space_raw)
            if feature_space_raw.shape[1] < self.layer_feature_dim:
                pad = np.zeros((feature_space_raw.shape[0], self.layer_feature_dim - feature_space_raw.shape[1]), dtype=np.float32)
                feature_space = np.concatenate([feature_space_raw, pad], axis=1)
            else:
                feature_space = feature_space_raw
            layer_feature_strengths[key] = feature_space
            probe_acc.append(_sigmoid(feature_space @ w["probe_w"] + w["probe_b"]))
            cap_acc.append(_sigmoid(feature_space @ w["cap_w"] + w["cap_b"]))
            conf_acc.append(_sigmoid(feature_space @ w["conf_w"] + w["conf_b"]))
            target_acc.append(target)
            control_acc.append(control)
            gap_acc.append(gap)
            weighted_features.append(feature_space)
            weighted_values.append(self._layer_weight(key))

        late_baseline = self._late_layer_baseline(weighted_features, weighted_values)
        drift_acc = []
        sae_acc = []
        for key, feature_space in layer_feature_strengths.items():
            weight = self._layer_weight(key)
            drift_acc.append(weight * np.mean(np.abs(feature_space - late_baseline), axis=1, keepdims=True))
            sae_acc.append(weight * np.abs(feature_space - np.mean(feature_space, axis=0, keepdims=True)))

        probe = np.mean(np.stack(probe_acc, axis=0), axis=0).astype(np.float32)
        cap = np.mean(np.stack(cap_acc, axis=0), axis=0).astype(np.float32)
        conf = np.mean(np.stack(conf_acc, axis=0), axis=0).astype(np.float32)
        target = np.mean(np.stack(target_acc, axis=0), axis=0).astype(np.float32)
        control = np.mean(np.stack(control_acc, axis=0), axis=0).astype(np.float32)
        gap = np.mean(np.stack(gap_acc, axis=0), axis=0).astype(np.float32)
        drift = np.mean(np.stack(drift_acc, axis=0), axis=0).astype(np.float32)
        sae = np.mean(np.stack(sae_acc, axis=0), axis=0).astype(np.float32)
        if sae.shape[1] > self.sae_dim:
            sae = sae[:, : self.sae_dim]
        elif sae.shape[1] < self.sae_dim:
            pad = np.zeros((sae.shape[0], self.sae_dim - sae.shape[1]), dtype=np.float32)
            sae = np.concatenate([sae, pad], axis=1)

        return SidecarScores(
            probe_scores=probe,
            conflict_score=conf,
            capability_scores=cap,
            sae_salience=sae,
            target_feature_strength=target,
            control_feature_strength=control,
            target_control_gap=gap,
            late_layer_drift=drift,
            layer_feature_strengths=layer_feature_strengths,
        )
