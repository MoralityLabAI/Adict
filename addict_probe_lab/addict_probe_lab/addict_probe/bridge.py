from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RUNTIME_FIELDS = ['ctx', 'commit_state', 'route_mask', 'score']


def load_runtime_trace_jsonl(path: str):
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def rows_to_feature_matrix(rows, pad_to: int = 128):
    feats = []
    for row in rows:
        flat = []
        for field in RUNTIME_FIELDS:
            val = row[field]
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                val = val[0]
            flat.extend(val)
        arr = np.array(flat, dtype=np.float32)
        if arr.shape[0] < pad_to:
            arr = np.pad(arr, (0, pad_to - arr.shape[0]))
        feats.append(arr[:pad_to])
    return np.stack(feats)
