from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LinearProbe:
    w: np.ndarray
    b: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> "LinearProbe":
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        a = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1], dtype=np.float32)
        w_all = np.linalg.solve(a, x_aug.T @ y)
        return LinearProbe(w=w_all[:-1], b=w_all[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32) @ self.w + self.b
