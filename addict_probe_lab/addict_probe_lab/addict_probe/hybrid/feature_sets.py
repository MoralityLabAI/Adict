from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class HybridFeatureSet:
    name: str
    target_feature_ids: tuple[int, ...]
    control_feature_ids: tuple[int, ...]
    late_layer_floor: int = 20
    tap_layers: tuple[int, ...] = (16, 24)


DEFAULT_FEATURE_SET = HybridFeatureSet(
    name="late_layer_constitution_v1",
    target_feature_ids=(5, 12),
    control_feature_ids=(18, 25),
    late_layer_floor=20,
    tap_layers=(16, 24),
)


def _coerce_int_tuple(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def load_feature_sets(path: str | Path) -> Dict[str, HybridFeatureSet]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw_sets: Mapping[str, Mapping[str, Any]]
    raw_sets = payload.get("feature_sets", payload)
    out: Dict[str, HybridFeatureSet] = {}
    for name, spec in raw_sets.items():
        out[str(name)] = HybridFeatureSet(
            name=str(name),
            target_feature_ids=_coerce_int_tuple(spec.get("target_feature_ids", DEFAULT_FEATURE_SET.target_feature_ids)),
            control_feature_ids=_coerce_int_tuple(spec.get("control_feature_ids", DEFAULT_FEATURE_SET.control_feature_ids)),
            late_layer_floor=int(spec.get("late_layer_floor", DEFAULT_FEATURE_SET.late_layer_floor)),
            tap_layers=_coerce_int_tuple(spec.get("tap_layers", DEFAULT_FEATURE_SET.tap_layers)),
        )
    return out


def resolve_feature_set(
    path: str | Path | None = None,
    name: str | None = None,
) -> HybridFeatureSet:
    if not path:
        return DEFAULT_FEATURE_SET
    feature_sets = load_feature_sets(path)
    if not feature_sets:
        return DEFAULT_FEATURE_SET
    if name and name in feature_sets:
        return feature_sets[name]
    if name:
        raise KeyError(f"feature set not found: {name}")
    first_key = next(iter(feature_sets.keys()))
    return feature_sets[first_key]
