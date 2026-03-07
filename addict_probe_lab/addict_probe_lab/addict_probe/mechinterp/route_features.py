from __future__ import annotations

from typing import Dict

import numpy as np

from ..trace_schema import DEFAULT_LAYOUT, read_trace_jsonl, row_to_feature


def feature_correlation_with_outcome(trace_jsonl: str, outcome_key: str = "moral_score_proxy") -> Dict[str, float]:
    rows = read_trace_jsonl(trace_jsonl)
    if not rows:
        return {}
    x = np.stack([row_to_feature(r, layout=DEFAULT_LAYOUT) for r in rows], axis=0)
    y = np.asarray([float(r.get(outcome_key, 0.0)) for r in rows], dtype=np.float32)
    y = (y - y.mean()) / (y.std() + 1e-6)
    xz = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    corr = (xz * y[:, None]).mean(axis=0)
    top_idx = np.argsort(np.abs(corr))[-16:][::-1]
    return {f"feat_{int(i)}": float(corr[i]) for i in top_idx}
