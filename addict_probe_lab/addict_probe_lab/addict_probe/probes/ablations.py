from __future__ import annotations

from typing import Dict, List

import numpy as np


def ablate_feature_spans(x: np.ndarray, spans: Dict[str, slice]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, sp in spans.items():
        x2 = x.copy()
        x2[:, sp] = 0.0
        out[name] = x2
    return out


def topk_feature_importance(weights: np.ndarray, k: int = 16) -> List[int]:
    w = np.asarray(weights)
    if w.ndim > 1:
        w = np.mean(np.abs(w), axis=1)
    return np.argsort(np.abs(w))[-k:][::-1].astype(int).tolist()
