from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from ..persistence.env import M0, M1, M2, M3, M4R, M5, M6, M7OFF, M7ON, MEQ, MV
from ..persistence.prefixes import restore_prefix
from ..persistence.rollout import run_test
from .lm import HookedLM, ResidualEdit
from .policy import LMPolicy

PI_CODES = ("M1", "M2", "M3")
# Manipulations whose first-step target probability is recorded exactly (no rollouts).
FIRST_STEP = (M0, M1, MV(1), MV(2), MEQ, M4R, M5, M6, M7ON, M7OFF)


def prefix_labels(
    lm: HookedLM,
    record: Dict[str, Any],
    n_rollouts: int = 4,
    seed: int = 0,
    edit: ResidualEdit | None = None,
    rollout_codes: Sequence[str] = PI_CODES,
) -> Dict[str, Any]:
    """Behavioural labels of one acquisition prefix for one model variant.

    PI is the mean fraction of test steps selecting the target under M1, M2 and
    M3, estimated from `n_rollouts` sampled continuations each. First-step
    target probabilities are exact. `edit` applies an intervention to every
    forward pass, so the same function measures intervened behaviour.
    """
    env, _ = restore_prefix(record)
    policy = LMPolicy(lm, edit)
    tau = env.target
    out: Dict[str, Any] = {"prefix_id": record["prefix_id"]}
    by_code = {"M1": M1, "M2": M2, "M3": M3}
    persist = {}
    for i, code in enumerate(rollout_codes):
        rng = np.random.default_rng([seed, record["env_seed"], i])
        res = run_test(env, policy, by_code[code], rng, n_rollouts=n_rollouts, stream=i)
        persist[code] = float((res.actions == tau).mean())
        out[f"first_p_{code}"] = float(res.first_probs[tau])
    out.update({f"persist_{c}": v for c, v in persist.items()})
    out["PI"] = float(np.mean([persist[c] for c in PI_CODES if c in persist])) if persist else float("nan")
    for j, manip in enumerate(FIRST_STEP):
        name = manip.code if manip.value is None else f"{manip.code}{manip.value}"
        key = f"first_p_{name}"
        if key in out:
            continue
        rng = np.random.default_rng([seed, record["env_seed"], 100 + j])
        res = run_test(env, policy, manip, rng, n_rollouts=0, stream=100 + j)
        out[key] = float(res.first_probs[tau])
    out["valid_mass_min"] = float(min(policy.valid_mass)) if policy.valid_mass else float("nan")
    out["valid_mass_mean"] = float(np.mean(policy.valid_mass)) if policy.valid_mass else float("nan")
    return out
