from __future__ import annotations

from typing import Dict

import numpy as np


def kmeans_numpy(x: np.ndarray, k: int = 6, iters: int = 20, seed: int = 7) -> Dict[str, np.ndarray]:
    if x.shape[0] == 0:
        return {"centroids": np.zeros((0, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int32)}
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(k, x.shape[0]), replace=False)
    c = x[idx].copy()
    for _ in range(iters):
        d = np.sum((x[:, None, :] - c[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d, axis=1)
        new_c = c.copy()
        for j in range(c.shape[0]):
            pts = x[labels == j]
            if len(pts) > 0:
                new_c[j] = pts.mean(axis=0)
        if np.allclose(c, new_c):
            break
        c = new_c
    return {"centroids": c.astype(np.float32), "labels": labels.astype(np.int32)}
