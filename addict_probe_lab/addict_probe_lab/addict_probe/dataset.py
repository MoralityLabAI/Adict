from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .trace_schema import DEFAULT_LAYOUT, FeatureLayout, jsonl_to_btf, sliding_windows, split_indices


@dataclass
class TraceDataset:
    x_windows: np.ndarray
    y_next: np.ndarray
    splits: Dict[str, np.ndarray]


def build_trace_dataset(
    trace_jsonl: str | Path,
    layout: FeatureLayout = DEFAULT_LAYOUT,
    window: int = 8,
    stride: int = 1,
    seed: int = 7,
) -> TraceDataset:
    btf, _ = jsonl_to_btf(trace_jsonl, layout=layout)
    x, y = sliding_windows(btf, window=window, stride=stride)
    splits = split_indices(len(x), seed=seed)
    return TraceDataset(x_windows=x, y_next=y, splits=splits)


def traces_to_btf(
    trace_jsonl: str | Path, layout: FeatureLayout = DEFAULT_LAYOUT
) -> Tuple[np.ndarray, list[str]]:
    return jsonl_to_btf(trace_jsonl, layout=layout)
