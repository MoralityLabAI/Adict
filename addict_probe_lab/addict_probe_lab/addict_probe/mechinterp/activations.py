from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..trace_schema import read_trace_jsonl


def extract_activation_matrix(trace_jsonl: str, key: str = "hidden_state_summary", dim: int = 32) -> np.ndarray:
    rows = read_trace_jsonl(trace_jsonl)
    out: List[np.ndarray] = []
    for r in rows:
        raw = r.get(key, []) or []
        arr = np.zeros((dim,), dtype=np.float32)
        v = np.asarray(raw, dtype=np.float32).reshape(-1) if len(raw) else np.zeros((0,), dtype=np.float32)
        n = min(dim, v.shape[0])
        arr[:n] = v[:n]
        out.append(arr)
    if not out:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(out, axis=0)
