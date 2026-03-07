from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .trace_schema import DEFAULT_LAYOUT, group_rows_by_episode, read_trace_jsonl, row_to_feature


def traces_to_training_tensors(
    trace_jsonl: str | Path, window: int = 8, stride: int = 1
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rows = read_trace_jsonl(trace_jsonl)
    by_ep = group_rows_by_episode(rows)

    x_list: List[np.ndarray] = []
    y_moral_next: List[float] = []
    y_consistency_delta: List[float] = []
    y_intervention_needed: List[float] = []
    y_usefulness_delta: List[float] = []

    for ep_rows in by_ep.values():
        feats = np.stack([row_to_feature(r, layout=DEFAULT_LAYOUT) for r in ep_rows], axis=0)
        moral = np.asarray([float(r.get("moral_score_proxy", 0.0)) for r in ep_rows], dtype=np.float32)
        consistency = np.asarray([float(r.get("consistency_score_proxy", 0.0)) for r in ep_rows], dtype=np.float32)
        usefulness = np.asarray([float(r.get("usefulness_score_proxy", 0.0)) for r in ep_rows], dtype=np.float32)
        intervene = np.asarray([float(bool(r.get("intervention_needed", False))) for r in ep_rows], dtype=np.float32)
        for i in range(0, len(ep_rows) - window, stride):
            j = i + window
            k = min(len(ep_rows) - 1, j + 2)
            x_list.append(feats[i:j])
            y_moral_next.append(float(moral[j]))
            y_consistency_delta.append(float(consistency[k] - consistency[j]))
            y_intervention_needed.append(float(intervene[j]))
            y_usefulness_delta.append(float(usefulness[k] - usefulness[j]))

    if not x_list:
        x = np.zeros((0, window, DEFAULT_LAYOUT.total_dim), dtype=np.float32)
        labels = {
            "next_moral_score": np.zeros((0,), dtype=np.float32),
            "future_consistency_delta": np.zeros((0,), dtype=np.float32),
            "intervention_needed": np.zeros((0,), dtype=np.float32),
            "future_usefulness_delta": np.zeros((0,), dtype=np.float32),
        }
        return x, labels

    x = np.asarray(x_list, dtype=np.float32)
    labels = {
        "next_moral_score": np.asarray(y_moral_next, dtype=np.float32),
        "future_consistency_delta": np.asarray(y_consistency_delta, dtype=np.float32),
        "intervention_needed": np.asarray(y_intervention_needed, dtype=np.float32),
        "future_usefulness_delta": np.asarray(y_usefulness_delta, dtype=np.float32),
    }
    return x, labels


def save_training_tensors(out_npz: str | Path, x: np.ndarray, labels: Dict[str, np.ndarray]) -> None:
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, x=x, **labels)
