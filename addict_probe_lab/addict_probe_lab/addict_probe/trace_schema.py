from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureLayout:
    """Fixed per-step feature layout for Addict Probe training."""

    prompt_summary: int = 16
    output_summary: int = 16
    hidden_summary: int = 32
    score_summary: int = 8
    history_summary: int = 16
    metadata_summary: int = 8
    reserved: int = 32

    @property
    def total_dim(self) -> int:
        return (
            self.prompt_summary
            + self.output_summary
            + self.hidden_summary
            + self.score_summary
            + self.history_summary
            + self.metadata_summary
            + self.reserved
        )


DEFAULT_LAYOUT = FeatureLayout()


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Small deterministic text embedding for fixed-shape summaries."""
    out = np.zeros((dim,), dtype=np.float32)
    if not text:
        return out
    tokens = text.lower().split()
    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] % 2 == 0) else -1.0
        out[idx] += sign
    norm = float(np.linalg.norm(out))
    if norm > 1e-6:
        out /= norm
    return out


def _float_list(values: Sequence[float], dim: int) -> np.ndarray:
    arr = np.zeros((dim,), dtype=np.float32)
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    n = min(dim, v.shape[0])
    arr[:n] = v[:n]
    return arr


def row_to_feature(row: Dict[str, Any], layout: FeatureLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """Convert one trace row into a fixed-size feature vector."""
    prompt_vec = _hash_vector(str(row.get("prompt_summary", "")), layout.prompt_summary)
    output_text = f"{row.get('reasoning_summary', '')} {row.get('model_output_summary', '')}"
    output_vec = _hash_vector(output_text.strip(), layout.output_summary)

    hidden_raw = row.get("hidden_state_summary", []) or row.get("activation_summary", []) or []
    hidden_vec = _float_list(hidden_raw, layout.hidden_summary)

    score_raw = [
        float(row.get("moral_score_proxy", 0.0)),
        float(row.get("usefulness_score_proxy", 0.0)),
        float(row.get("consistency_score_proxy", 0.0)),
        float(row.get("manipulation_risk_proxy", 0.0)),
        float(row.get("story_goal_progress_proxy", 0.0)),
        float(row.get("latency_ms", 0.0) / 1000.0),
        float(bool(row.get("intervention_needed", False))),
        float(row.get("step_idx", 0)),
    ]
    score_vec = _float_list(score_raw, layout.score_summary)

    hist_vec = _hash_vector(str(row.get("bounded_history_summary", "")), layout.history_summary)
    meta_text = "|".join(
        [
            str(row.get("scenario_id", "")),
            str(row.get("backend", "")),
            str(row.get("oracle", "")),
            str(row.get("episode_id", "")),
        ]
    )
    meta_vec = _hash_vector(meta_text, layout.metadata_summary)
    reserve_vec = np.zeros((layout.reserved,), dtype=np.float32)

    return np.concatenate(
        [prompt_vec, output_vec, hidden_vec, score_vec, hist_vec, meta_vec, reserve_vec], axis=0
    ).astype(np.float32)


def read_trace_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_rows_by_episode(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        ep = str(row.get("episode_id", ""))
        by_ep.setdefault(ep, []).append(row)
    for ep_rows in by_ep.values():
        ep_rows.sort(key=lambda r: int(r.get("step_idx", 0)))
    return by_ep


def rows_to_time_features(
    rows: Sequence[Dict[str, Any]], layout: FeatureLayout = DEFAULT_LAYOUT
) -> np.ndarray:
    return np.stack([row_to_feature(r, layout=layout) for r in rows], axis=0).astype(np.float32)


def jsonl_to_btf(
    path: str | Path, layout: FeatureLayout = DEFAULT_LAYOUT, pad_time_to: int | None = None
) -> Tuple[np.ndarray, List[str]]:
    """Convert trace JSONL to [batch, time, feat] using episode grouping."""
    rows = read_trace_jsonl(path)
    by_ep = group_rows_by_episode(rows)
    episode_ids = sorted(by_ep.keys())
    if not episode_ids:
        return np.zeros((0, 0, layout.total_dim), dtype=np.float32), []
    time_lengths = [len(by_ep[e]) for e in episode_ids]
    t_max = max(time_lengths) if pad_time_to is None else max(max(time_lengths), int(pad_time_to))

    out = np.zeros((len(episode_ids), t_max, layout.total_dim), dtype=np.float32)
    for i, ep in enumerate(episode_ids):
        feats = rows_to_time_features(by_ep[ep], layout=layout)
        out[i, : feats.shape[0], :] = feats
    return out, episode_ids


def sliding_windows(
    btf: np.ndarray, window: int = 8, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce [n, window, feat] windows and aligned next-step targets."""
    if btf.ndim != 3 or btf.shape[1] < window + 1:
        return (
            np.zeros((0, window, btf.shape[-1] if btf.ndim == 3 else 0), dtype=np.float32),
            np.zeros((0, btf.shape[-1] if btf.ndim == 3 else 0), dtype=np.float32),
        )
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for b in range(btf.shape[0]):
        seq = btf[b]
        for i in range(0, seq.shape[0] - window, stride):
            xs.append(seq[i : i + window])
            ys.append(seq[i + window])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_indices(n: int, seed: int = 7, train_frac: float = 0.8, val_frac: float = 0.1) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}
