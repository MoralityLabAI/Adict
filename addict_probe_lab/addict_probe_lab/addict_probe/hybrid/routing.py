from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import RoutingOutput


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-12, None)


@dataclass
class RoutingMLP:
    input_dim: int
    adapter_count: int
    hidden_dim: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 0.08, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.08, size=(self.hidden_dim, self.adapter_count)).astype(np.float32)
        self.b2 = np.zeros((self.adapter_count,), dtype=np.float32)

    def route(self, sidecar_vector: np.ndarray) -> RoutingOutput:
        """
        sidecar_vector: [batch, input_dim]
        """
        h = np.tanh(sidecar_vector @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        w = _softmax(logits, axis=1).astype(np.float32)
        entropy = -np.sum(w * np.log(np.clip(w, 1e-12, 1.0)), axis=1)
        top1 = np.argmax(w, axis=1)
        counts = np.bincount(top1, minlength=self.adapter_count)
        mode_rate = float(np.max(counts) / max(1, int(np.sum(counts))))
        top1_rate = float(np.mean(np.max(w, axis=1))) if w.size else 0.0
        return RoutingOutput(
            route_weights=w,
            entropy_mean=float(np.mean(entropy)) if entropy.size else 0.0,
            top1_usage_rate=top1_rate,
            mode_usage_rate=mode_rate,
        )

