from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..probes.lexica_runtime import LexicaRuntime
from ..trace_schema import DEFAULT_LAYOUT, group_rows_by_episode, read_trace_jsonl, row_to_feature


@dataclass
class AtomTransitionStat:
    """Weighted effect summary for one lexica atom across moral trajectories."""

    atom_idx: int
    support_steps: int
    activation_mass: float
    mean_activation: float
    moral_lift: float
    consistency_lift: float
    usefulness_lift: float
    intervention_lift: float
    composite_gain: float
    top_scenarios: List[str]


def _episode_targets(rows: List[Dict[str, object]], horizon: int) -> Dict[str, np.ndarray]:
    moral = np.asarray([float(r.get("moral_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    consistency = np.asarray([float(r.get("consistency_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    usefulness = np.asarray([float(r.get("usefulness_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    intervene = np.asarray([float(bool(r.get("intervention_needed", False))) for r in rows], dtype=np.float32)

    moral_delta = np.zeros_like(moral)
    consistency_delta = np.zeros_like(consistency)
    usefulness_delta = np.zeros_like(usefulness)
    intervention_delta = np.zeros_like(intervene)

    for idx in range(len(rows)):
        j = min(len(rows) - 1, idx + max(1, horizon))
        moral_delta[idx] = moral[j] - moral[idx]
        consistency_delta[idx] = consistency[j] - consistency[idx]
        usefulness_delta[idx] = usefulness[j] - usefulness[idx]
        intervention_delta[idx] = intervene[j] - intervene[idx]

    return {
        "moral_delta": moral_delta,
        "consistency_delta": consistency_delta,
        "usefulness_delta": usefulness_delta,
        "intervention_delta": intervention_delta,
    }


def analyze_transition_lens(
    trace_path: str | Path,
    lexica_npz: str | Path,
    horizon: int = 2,
    activation_floor: float = 0.05,
    top_atoms: int = 12,
) -> Dict[str, object]:
    """Rank lexica atoms by weighted future-score shifts instead of per-step fit only."""

    rows = read_trace_jsonl(trace_path)
    if not rows:
        return {
            "trace_path": str(trace_path),
            "lexica_npz": str(lexica_npz),
            "horizon": int(horizon),
            "row_count": 0,
            "atom_count": 0,
            "top_atoms": [],
        }

    runtime = LexicaRuntime.from_npz(lexica_npz)
    features = np.stack([row_to_feature(row, layout=DEFAULT_LAYOUT) for row in rows], axis=0).astype(np.float32)
    atom_acts = np.maximum(features @ runtime.dictionary.T, 0.0)
    atom_count = int(atom_acts.shape[1])

    by_ep = group_rows_by_episode(rows)
    targets = {
        "moral_delta": np.zeros((len(rows),), dtype=np.float32),
        "consistency_delta": np.zeros((len(rows),), dtype=np.float32),
        "usefulness_delta": np.zeros((len(rows),), dtype=np.float32),
        "intervention_delta": np.zeros((len(rows),), dtype=np.float32),
    }

    cursor = 0
    for ep_rows in by_ep.values():
        ep_targets = _episode_targets(ep_rows, horizon=horizon)
        n = len(ep_rows)
        for key, arr in ep_targets.items():
            targets[key][cursor : cursor + n] = arr
        cursor += n

    global_means = {key: float(np.mean(val)) for key, val in targets.items()}
    stats: List[AtomTransitionStat] = []

    for atom_idx in range(atom_count):
        acts = atom_acts[:, atom_idx]
        mask = acts > activation_floor
        support = int(np.sum(mask))
        if support == 0:
            continue

        weights = acts[mask]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            continue

        def weighted_avg(name: str) -> float:
            vals = targets[name][mask]
            return float(np.sum(weights * vals) / weight_sum)

        scenario_mass: Dict[str, float] = {}
        filtered_rows = [row for row, keep in zip(rows, mask) if keep]
        for row, w in zip(filtered_rows, weights):
            sid = str(row.get("scenario_id", ""))
            scenario_mass[sid] = scenario_mass.get(sid, 0.0) + float(w)
        top_scenarios = [sid for sid, _ in sorted(scenario_mass.items(), key=lambda item: item[1], reverse=True)[:3]]

        moral_lift = weighted_avg("moral_delta") - global_means["moral_delta"]
        consistency_lift = weighted_avg("consistency_delta") - global_means["consistency_delta"]
        usefulness_lift = weighted_avg("usefulness_delta") - global_means["usefulness_delta"]
        intervention_lift = weighted_avg("intervention_delta") - global_means["intervention_delta"]
        composite_gain = moral_lift + consistency_lift + 0.5 * usefulness_lift - 0.5 * intervention_lift

        stats.append(
            AtomTransitionStat(
                atom_idx=atom_idx,
                support_steps=support,
                activation_mass=weight_sum,
                mean_activation=float(np.mean(weights)),
                moral_lift=float(moral_lift),
                consistency_lift=float(consistency_lift),
                usefulness_lift=float(usefulness_lift),
                intervention_lift=float(intervention_lift),
                composite_gain=float(composite_gain),
                top_scenarios=top_scenarios,
            )
        )

    stats.sort(key=lambda item: item.composite_gain, reverse=True)
    top = stats[:top_atoms]
    bottom = list(reversed(stats[-top_atoms:])) if stats else []

    return {
        "trace_path": str(trace_path),
        "lexica_npz": str(lexica_npz),
        "horizon": int(horizon),
        "row_count": int(len(rows)),
        "atom_count": atom_count,
        "global_future_means": global_means,
        "top_atoms": [asdict(item) for item in top],
        "bottom_atoms": [asdict(item) for item in bottom],
        "guardrail_atoms": [int(item.atom_idx) for item in top[: min(6, len(top))]],
        "suppress_atoms": [int(item.atom_idx) for item in bottom[: min(6, len(bottom))]],
    }


def write_transition_report(
    trace_path: str | Path,
    lexica_npz: str | Path,
    out_json: str | Path,
    horizon: int = 2,
    activation_floor: float = 0.05,
    top_atoms: int = 12,
) -> Dict[str, object]:
    report = analyze_transition_lens(
        trace_path=trace_path,
        lexica_npz=lexica_npz,
        horizon=horizon,
        activation_floor=activation_floor,
        top_atoms=top_atoms,
    )
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["AtomTransitionStat", "analyze_transition_lens", "write_transition_report"]
