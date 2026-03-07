from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class TinyMLPScorer:
    """Small fixed-shape scorer suitable for Core ML export."""

    input_dim: int = 72
    hidden_dim: int = 64
    output_dim: int = 4
    seed: int = 7

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = (0.02 * rng.normal(size=(self.input_dim, self.hidden_dim))).astype(np.float32)
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.w2 = (0.02 * rng.normal(size=(self.hidden_dim, self.output_dim))).astype(np.float32)
        self.b2 = np.zeros((self.output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        y = h @ self.w2 + self.b2
        return y

    def score_dict(self, x: np.ndarray) -> Dict[str, float]:
        y = self.forward(x)[0]
        sig = 1.0 / (1.0 + np.exp(-y))
        return {
            "moral_reasoning_score": float(sig[0]),
            "consistency_score": float(sig[1]),
            "manipulation_risk": float(sig[2]),
            "story_goal_progress": float(sig[3]),
        }

    def save_npz(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            input_dim=np.array([self.input_dim], dtype=np.int32),
            hidden_dim=np.array([self.hidden_dim], dtype=np.int32),
            output_dim=np.array([self.output_dim], dtype=np.int32),
        )

    @staticmethod
    def load_npz(path: str | Path) -> "TinyMLPScorer":
        data = np.load(path)
        m = TinyMLPScorer(
            input_dim=int(data["input_dim"][0]),
            hidden_dim=int(data["hidden_dim"][0]),
            output_dim=int(data["output_dim"][0]),
        )
        m.w1 = data["w1"].astype(np.float32)
        m.b1 = data["b1"].astype(np.float32)
        m.w2 = data["w2"].astype(np.float32)
        m.b2 = data["b2"].astype(np.float32)
        return m
