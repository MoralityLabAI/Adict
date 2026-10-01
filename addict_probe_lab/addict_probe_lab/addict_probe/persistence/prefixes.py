from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .behaviour_model import PREFIX_THETA, BehaviourPolicy
from .env import ChoiceEnv, EnvSpec
from .grammar import render_line
from .indices import PREFIX_EPSILON
from .rollout import StepRecord, run_acquisition, run_scripted


def build_prefix(spec: EnvSpec) -> Dict[str, Any]:
    """Generate one shared acquisition prefix and return its dataset record.

    `tstar_char` is the exclusive end offset of the last acquisition outcome
    line in `text`; token offsets are derived per tokenizer at collection time.
    """
    env = ChoiceEnv(spec)
    rng = np.random.default_rng([spec.seed, 1])
    records = run_acquisition(env, BehaviourPolicy(PREFIX_THETA, spec.n_options), rng, epsilon=PREFIX_EPSILON)
    step_end_chars: List[int] = []
    offset = 0
    for line in env.lines:
        offset += len(render_line(line))
        if line.kind == "outcome":
            step_end_chars.append(offset)
        offset += 1  # newline
    return {
        "prefix_id": f"{spec.family}-{spec.vocab_split}-k{spec.n_options}-t{spec.t_acq}-s{spec.seed}",
        "env_seed": spec.seed,
        "family": spec.family,
        "vocab_split": spec.vocab_split,
        "n_options": spec.n_options,
        "t_acq": spec.t_acq,
        "t_test": spec.t_test,
        "target_points": spec.target_points,
        "alt_points": spec.alt_points,
        "label_pool": list(spec.label_pool),
        "labels": env.labels,
        "target": env.target,
        "alt": env.alt,
        "rstar_name": env.rstar_name,
        "actions": [rec.action for rec in records],
        "target_selected": [int(rec.action == env.target) for rec in records],
        "points": [rec.outcome.points for rec in records],
        "text": env.text(),
        "step_end_chars": step_end_chars,
        "tstar_char": step_end_chars[-1],
    }


def spec_of(record: Dict[str, Any]) -> EnvSpec:
    return EnvSpec(
        family=record["family"],
        n_options=record["n_options"],
        t_acq=record["t_acq"],
        t_test=record["t_test"],
        vocab_split=record["vocab_split"],
        seed=record["env_seed"],
        target_points=record["target_points"],
        alt_points=record["alt_points"],
        label_pool=tuple(record.get("label_pool", ())),
    )


def restore_prefix(record: Dict[str, Any]) -> Tuple[ChoiceEnv, List[StepRecord]]:
    """Rebuild the environment at the end of acquisition from a stored record."""
    env = ChoiceEnv(spec_of(record))
    records = run_scripted(env, record["actions"])
    if env.text() != record["text"]:
        raise RuntimeError(f"prefix {record['prefix_id']} does not reproduce its stored transcript")
    return env, records
