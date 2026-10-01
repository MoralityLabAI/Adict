from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .behaviour_model import BehaviourPolicy, Theta
from .env import MV, ChoiceEnv, EnvSpec
from .rollout import replay, run_acquisition, run_scripted
from .teacher_data import ACQ_EPSILON, sample_theta
from .tokens import PAD_ID, TOKEN_ID, encode_lines

# The second factor of a yoked set, crossed with two action scripts.
#   "payoff": the points on every outcome line, as (target points, points of every other option).
#   "notice": a notice after the history that states the target's points.
FACTORS = ("payoff", "notice")
SCRIPT_SOURCES = ("teacher", "random", "altswap")
MIN_ALT_COUNT_GAP = 3
ALTSWAP_ATTEMPTS = 40
HISTORY_STEPS = {"payoff": 12, "notice": 16}
PAYOFFS: Tuple[Tuple[int, int], ...] = ((3, 1), (1, 1))
NOTICE_POINTS: Tuple[int, int] = (3, 0)  # restated unchanged, or devalued
# Corner index = 2 * action script + level of the second factor.
CORNERS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
MIN_TARGET_COUNT_GAP = 3


def _script(target: int, k: int, n_target: int, steps: int, rng: np.random.Generator) -> List[int]:
    """Random-order action sequence that tries every option and selects the target `n_target` times."""
    alts = [a for a in range(k) if a != target]
    fill = [int(a) for a in rng.choice(alts, size=steps - n_target - len(alts))]
    return [int(a) for a in rng.permutation(alts + [target] * n_target + fill)]


def _teacher_script(seed: int, k: int, steps: int, rng: np.random.Generator) -> List[int]:
    """Actions of a freshly sampled teacher in this environment, generated as in training."""
    env = ChoiceEnv(EnvSpec(n_options=k, t_acq=steps, seed=seed))
    _, theta = sample_theta(rng)
    records = run_acquisition(env, BehaviourPolicy(theta, k), rng, epsilon=ACQ_EPSILON)
    return [rec.action for rec in records]


def _altswap_pair(seed: int, target: int, k: int, steps: int, rng: np.random.Generator) -> List[List[int]]:
    """A script and the same script with the two non-target options exchanged.

    The two non-target options have identical payoffs, so the exchanged history
    has the same outcome sequence, the same points record and, by symmetry, the
    same evidence about the regime. Only the assignment of own-choice history to
    option labels differs: the choice kernel, isolated from everything else.
    """
    if k != 3:
        raise ValueError("the exchange of non-target options is defined for three options")
    a, b = [o for o in range(k) if o != target]
    base = None
    for _ in range(ALTSWAP_ATTEMPTS):
        candidate = _teacher_script(seed, k, steps, rng)
        if min(candidate.count(a), candidate.count(b)) >= 1 and (
            abs(candidate.count(a) - candidate.count(b)) >= MIN_ALT_COUNT_GAP
        ):
            base = candidate
            break
    if base is None:  # no teacher history was lopsided enough; fall back to a random one
        n_a = int(rng.integers(MIN_ALT_COUNT_GAP + 1, steps - 1))
        n_b = int(rng.integers(1, n_a - MIN_ALT_COUNT_GAP + 1))
        n_b = min(n_b, steps - n_a - 1)
        base = [int(x) for x in rng.permutation([a] * n_a + [b] * n_b + [target] * (steps - n_a - n_b))]
    exchange = {a: b, b: a, target: target}
    return [base, [exchange[o] for o in base]]


def generate_yoked(
    seed: int, n_options: int = 3, factor: str = "payoff", scripts: str = "teacher"
) -> Dict[str, np.ndarray]:
    """Four forced histories in one environment: two action scripts crossed with two levels of `factor`.

    All four share labels, display orders, tags and token positions. They differ
    only in the chosen labels (action script) and in the points digits that the
    second factor controls, so own-choice history and value information vary
    independently. Each sequence ends at the choice marker of the step that
    follows the forced history.

    `scripts` is "teacher" (two independently sampled teachers acting in the
    environment, as in training), "random" (random-order sequences that try
    every option and differ in how often the target is selected) or "altswap"
    (one teacher history and its copy with the two non-target options exchanged).
    """
    if factor not in FACTORS:
        raise ValueError(f"unknown factor: {factor}")
    if scripts not in SCRIPT_SOURCES:
        raise ValueError(f"unknown script source: {scripts}")
    rng = np.random.default_rng([seed, 11])
    k = n_options
    steps = HISTORY_STEPS[factor]
    # A notice is a manipulation, so it sits at the end of the acquisition phase.
    t_acq = steps if factor == "notice" else steps + 4
    target = ChoiceEnv(EnvSpec(n_options=k, t_acq=t_acq, seed=seed)).target
    if scripts == "random":
        max_target = steps - (k - 1)
        while True:
            n_target = rng.integers(1, max_target + 1, size=2)
            if abs(int(n_target[0]) - int(n_target[1])) >= MIN_TARGET_COUNT_GAP:
                break
        script_pair = [_script(target, k, int(n), steps, rng) for n in n_target]
    elif scripts == "altswap":
        script_pair = _altswap_pair(seed, target, k, steps, rng)
        n_target = np.asarray([s.count(target) for s in script_pair])
    else:
        script_pair = [_teacher_script(seed, k, steps, rng)]
        while len(script_pair) < 2:
            other = _teacher_script(seed, k, steps, rng)
            if other != script_pair[0]:
                script_pair.append(other)
        n_target = np.asarray([s.count(target) for s in script_pair])

    tokens: List[List[int]] = []
    h_slot = np.zeros((4, k), dtype=np.float32)
    v_slot = np.zeros((4, k), dtype=np.float32)
    order = slot_ids = None
    for c, (a, level) in enumerate(CORNERS):
        target_points, alt_points = PAYOFFS[level] if factor == "payoff" else PAYOFFS[0]
        env = ChoiceEnv(
            EnvSpec(n_options=k, t_acq=t_acq, seed=seed, target_points=target_points, alt_points=alt_points)
        )
        records = run_scripted(env, script_pair[a])
        # Latents that do not depend on the regime: choice kernel and the accurate points record.
        policy = BehaviourPolicy(Theta(), k)
        replay(policy, records)
        if factor == "notice":
            for n in env.apply(MV(NOTICE_POINTS[level])):
                policy.observe_notice(n.option, n.points, n.rstar)
        obs = env.begin_step()
        lat = policy.latents(obs.cue)
        h_slot[c], v_slot[c] = lat["h"][obs.order], lat["v"][obs.order]
        tokens.append(encode_lines(env.lines) + [TOKEN_ID["CH"]])
        corner_order = np.asarray(obs.order, dtype=np.int8)
        corner_ids = np.asarray([TOKEN_ID[env.labels[o]] for o in obs.order], dtype=np.int16)
        if order is None:
            order, slot_ids = corner_order, corner_ids
        elif not (np.array_equal(order, corner_order) and np.array_equal(slot_ids, corner_ids)):
            raise RuntimeError("corners of a yoked set must share the display order")

    length = len(tokens[0])
    if any(len(t) != length for t in tokens):
        raise RuntimeError("corners of a yoked set must have the same length")
    arr = np.asarray(tokens, dtype=np.int16)
    is_label = np.isin(arr[0], slot_ids)
    note = np.flatnonzero(arr[0] == TOKEN_ID["NOTE"])
    return {
        "tokens": arr,  # [4, length]
        "ch_pos": np.int16(length - 1),
        "opt_pos": np.flatnonzero(is_label)[-k:].astype(np.int16),  # option tokens of the final options line
        # Points digit of the notice, where the stated value enters; -1 without a notice.
        "note_pos": np.int16(note[-1] + 2 if note.size else -1),
        "slot_ids": slot_ids,
        "target_slot": np.int8(list(order).index(target)),
        "n_target": n_target.astype(np.int8),  # per action script
        "h_slot": h_slot,
        "v_slot": v_slot,
        "env_seed": np.int64(seed),
    }


def generate_yoked_dataset(
    seed_start: int, n: int, factor: str = "payoff", scripts: str = "teacher"
) -> Dict[str, np.ndarray]:
    """Stack `n` yoked sets with consecutive environment seeds; tokens are right-padded."""
    rows = [generate_yoked(seed_start + i, factor=factor, scripts=scripts) for i in range(n)]
    length = max(r["tokens"].shape[1] for r in rows)
    out = {key: np.stack([r[key] for r in rows], axis=0) for key in rows[0] if key != "tokens"}
    tokens = np.full((n, 4, length), PAD_ID, dtype=np.int16)
    for i, r in enumerate(rows):
        tokens[i, :, : r["tokens"].shape[1]] = r["tokens"]
    out["tokens"] = tokens
    return out
