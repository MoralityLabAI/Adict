from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .linear_probe import LinearProbe


@dataclass
class TemporalProbe:
    """Flattened-window temporal probe baseline."""

    base: LinearProbe
    window: int
    feat_dim: int

    @staticmethod
    def fit(x_windows: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> "TemporalProbe":
        n, w, f = x_windows.shape
        x_flat = x_windows.reshape(n, w * f)
        base = LinearProbe.fit(x_flat, y, ridge=ridge)
        return TemporalProbe(base=base, window=w, feat_dim=f)

    def predict(self, x_windows: np.ndarray) -> np.ndarray:
        x_flat = x_windows.reshape(x_windows.shape[0], self.window * self.feat_dim)
        return self.base.predict(x_flat)
