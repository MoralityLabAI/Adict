from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .behaviour_model import PREFIX_THETA, REFERENCE_THETA, BehaviourPolicy
from .env import M0, M1, M2, M3, M4, M4R, M5, M6, M7OFF, M7ON, MEQ, MV, ChoiceEnv, EnvSpec
from .rollout import Policy, replay, run_acquisition, run_scripted, run_test

PolicyFactory = Callable[[int], Policy]

ROLLOUT_MANIPULATIONS = (M0, M1, M2, M3, M4, M4R, M5, MEQ, M6, M7ON, M7OFF)
PI_MANIPULATIONS = ("M1", "M2", "M3")
MV_POINTS = (0, 1, 2)  # stated target points used for the slope
T_ACQ_CHOICES = (8, 16, 24)
PREFIX_EPSILON = 0.15
LATE_STEPS = 4
KAPPA_HISTORY_STEPS = 12


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
    return float(np.log(p / (1.0 - p)))


@dataclass
class _Raw:
    """Per-policy accumulators over prefixes."""

    first_p: Dict[str, List[float]] = field(default_factory=dict)
    frac: Dict[str, List[float]] = field(default_factory=dict)
    late: Dict[str, List[float]] = field(default_factory=dict)
    follow: Dict[str, List[float]] = field(default_factory=dict)
    mv_logit: Dict[int, List[float]] = field(default_factory=dict)
    switch_trials: List[float] = field(default_factory=list)
    target_by_cost: List[np.ndarray] = field(default_factory=list)
    vq: List[float] = field(default_factory=list)
    marked_episodes: int = 0
    false_reports: int = 0


def _measure_prefix(raw: _Raw, env: ChoiceEnv, policy: Policy, seed: int, i: int, n_rollouts: int) -> None:
    tau, r = env.target, env.alt
    t_test = env.spec.t_test
    for mi, manip in enumerate(ROLLOUT_MANIPULATIONS):
        code = manip.code
        # Same seed for every policy: common random numbers across policies.
        rng = np.random.default_rng([seed, i, 1 + mi])
        res = run_test(env, policy, manip, rng, n_rollouts=n_rollouts, value_query=code == "M1", stream=mi)
        is_t = res.actions == tau
        raw.first_p.setdefault(code, []).append(float(res.first_probs[tau]))
        raw.frac.setdefault(code, []).append(float(is_t.mean()))
        raw.late.setdefault(code, []).append(float(is_t[:, -LATE_STEPS:].mean()))
        if code == "M1":
            raw.vq.append(float(bool(res.value_query_correct)))
        elif code == "M2":
            for row in ~is_t:
                raw.switch_trials.append(float(np.argmax(row) + 1 if row.any() else t_test + 1))
        elif code == "M3":
            raw.target_by_cost.append(is_t.mean(axis=0))
        elif code in ("M4", "M4R"):
            raw.follow.setdefault(code, []).append(float((res.actions == r).mean() - is_t.mean()))
        elif code == "M7off":
            for sel, rep in zip(res.selected_marked, res.reports):
                raw.marked_episodes += int(sel)
                raw.false_reports += int(sel and not rep)
    for v in MV_POINTS:
        rng = np.random.default_rng([seed, i, 100 + v])
        res = run_test(env, policy, MV(v), rng, n_rollouts=0, stream=100 + v)
        raw.mv_logit.setdefault(v, []).append(_logit(res.first_probs[tau]))


def _first_cost_below_half(raw: _Raw, t_test: int) -> float:
    """First cost level (equal to the 1-based test step) with target selection under 0.5."""
    by_cost = np.mean(np.stack(raw.target_by_cost, axis=0), axis=0)
    below = np.flatnonzero(by_cost < 0.5)
    return float(below[0] + 1) if below.size else float(t_test + 1)


def _history_script(target: int, k: int, steps: int, many: bool, rng: np.random.Generator) -> List[int]:
    """Action sequence that tries every option, with many or few target selections."""
    alts = [a for a in range(k) if a != target]
    if many:
        script = alts + [target] * (steps - len(alts))
    else:
        script = [target, target] + [alts[j % len(alts)] for j in range(steps - 2)]
    return [int(a) for a in rng.permutation(script)]


def kappa_index(
    factory: PolicyFactory,
    n_options: int = 3,
    n_samples: int = 100,
    seed: int = 0,
    label_pool: Tuple[str, ...] = (),
) -> float:
    """Effect of own-choice history on target selection with outcome history held fixed.

    Two forced histories share an environment in which every option yields the
    same points; they differ only in how often the target was selected. The
    index is the mean difference in logit P(target) at the next step.
    """
    diffs: List[float] = []
    for i in range(n_samples):
        spec = EnvSpec(
            family="F1",
            n_options=n_options,
            t_acq=KAPPA_HISTORY_STEPS,
            seed=seed * 100_000 + 50_000 + i,
            target_points=1,
            alt_points=1,
            label_pool=label_pool,
        )
        logits: List[float] = []
        for arm, many in enumerate((True, False)):
            env = ChoiceEnv(spec)
            rng = np.random.default_rng([seed, i, 200 + arm])
            records = run_scripted(env, _history_script(env.target, n_options, spec.t_acq, many, rng))
            policy = factory(n_options)
            replay(policy, records)
            res = run_test(env, policy, M0, rng, n_rollouts=0)
            logits.append(_logit(res.first_probs[env.target]))
        diffs.append(logits[0] - logits[1])
    return float(np.mean(diffs))


def _finalise(raw: _Raw, ref: _Raw, n_options: int, t_test: int, alt_points: int) -> Dict[str, float]:
    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if len(xs) else float("nan")

    chance = 1.0 / n_options
    ref_drop = mean(ref.first_p["M0"]) - mean(ref.first_p["M1"])
    deltas = np.asarray([v - alt_points for v in MV_POINTS], dtype=np.float64)
    mv = np.asarray([mean(raw.mv_logit[v]) for v in MV_POINTS], dtype=np.float64)
    out = {
        "b0": mean(raw.late["MEQ"]) - chance,
        "beta": float(np.polyfit(deltas, mv, 1)[0]),
        "S_stated": float(np.clip((mean(raw.first_p["M0"]) - mean(raw.first_p["M1"])) / ref_drop, 0.0, 1.0)),
        "S_exp": float(np.clip(mean(ref.switch_trials) / mean(raw.switch_trials), 0.0, 1.0)),
        "BP": _first_cost_below_half(raw, t_test) / _first_cost_below_half(ref, t_test),
        "OF": mean(raw.follow["M4"]),
        "RF": mean(raw.follow["M4R"]),
        "EXT": mean(raw.late["M5"]) - chance,
        "CG": mean(raw.first_p["M0"]) - mean(raw.first_p["M6"]),
        "AG": mean(raw.first_p["M7off"]) - mean(raw.first_p["M7on"]),
        "RAD": raw.false_reports / raw.marked_episodes if raw.marked_episodes else float("nan"),
        "VQ": mean(raw.vq),
        "PI": float(np.mean([mean(raw.frac[c]) for c in PI_MANIPULATIONS])),
    }
    for code, values in raw.frac.items():
        out[f"persist_{code}"] = mean(values)
    return out


def measure_indices(
    factories: Dict[str, PolicyFactory],
    n_prefixes: int = 200,
    n_rollouts: int = 4,
    n_options: int = 3,
    family: str = "F1",
    seed: int = 0,
    kappa_samples: int = 100,
    label_pool: Tuple[str, ...] = (),
) -> Dict[str, Dict[str, float]]:
    """Sensitivity indices (spec section 4.2) for each policy on shared forced prefixes.

    Every policy reads the same acquisition history and then continues freely
    from forks of the same environment state, so differences in the indices are
    differences between policies and not between histories.
    """
    if "reference" in factories:
        raise ValueError("'reference' is reserved")
    all_factories: Dict[str, PolicyFactory] = {
        "reference": lambda k: BehaviourPolicy(REFERENCE_THETA, k),
        **factories,
    }
    raws = {name: _Raw() for name in all_factories}
    t_rng = np.random.default_rng([seed, 0])
    spec = EnvSpec(family=family, n_options=n_options)
    for i in range(n_prefixes):
        spec = EnvSpec(
            family=family,
            n_options=n_options,
            t_acq=int(t_rng.choice(T_ACQ_CHOICES)),
            seed=seed * 100_000 + i,
            label_pool=label_pool,
        )
        env = ChoiceEnv(spec)
        records = run_acquisition(
            env,
            BehaviourPolicy(PREFIX_THETA, n_options),
            np.random.default_rng([seed, i, 0]),
            epsilon=PREFIX_EPSILON,
        )
        for name, factory in all_factories.items():
            policy = factory(n_options)
            replay(policy, records)
            _measure_prefix(raws[name], env, policy, seed, i, n_rollouts)

    out: Dict[str, Dict[str, float]] = {}
    for name, factory in all_factories.items():
        ix = _finalise(raws[name], raws["reference"], n_options, spec.t_test, spec.alt_points)
        ix["kappa"] = kappa_index(
            factory, n_options=n_options, n_samples=kappa_samples, seed=seed, label_pool=label_pool
        )
        out[name] = ix
    return out
