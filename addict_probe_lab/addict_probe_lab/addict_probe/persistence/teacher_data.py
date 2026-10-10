from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import numpy as np

from .behaviour_model import REGIME_THETAS, BehaviourPolicy, Theta
from .env import M0, M1, M2, M3, M4, M4R, M5, M6, M7OFF, M7ON, MA, MEQ, MV, ChoiceEnv, EnvSpec
from .grammar import TAGS
from .indices import T_ACQ_CHOICES
from .tokens import BOS_ID, PAD_ID, TOKEN_ID, line_tokens

REGIME_ORDER: Tuple[str, ...] = tuple(REGIME_THETAS)
MANIPULATION_ORDER = (M0, M1, M2, M3, M4, M4R, M5, MEQ, M6, M7ON, M7OFF)
# Codes by the index stored in an episode's `manipulation` field; the last two exist only in variant "v2".
MANIPULATION_CODES = tuple(m.code for m in MANIPULATION_ORDER) + ("MV", "MA")
DATA_VARIANTS = ("v1", "v2")
# Share of "v2" episodes whose manipulation is a stated value change with a sampled value.
STATED_VALUE_WEIGHT = 0.4
MAX_STEPS = max(T_ACQ_CHOICES) + EnvSpec().t_test
MAX_TOKENS = 560
ACQ_EPSILON = 0.1
# The choice kernel only enters selection under HAB, so HAB is oversampled to
# make the kernel worth tracking in every episode.
HAB_WEIGHT = 0.25


def sample_theta(rng: np.random.Generator) -> Tuple[str, Theta]:
    """Draw a regime and jitter its canonical parameters."""
    codes = list(REGIME_ORDER)
    w = np.full((len(codes),), (1.0 - HAB_WEIGHT) / (len(codes) - 1))
    w[codes.index("HAB")] = HAB_WEIGHT
    code = codes[int(rng.choice(len(codes), p=w))]
    return code, jitter_theta(code, rng)


def jitter_theta(code: str, rng: np.random.Generator) -> Theta:
    """Canonical parameters of one regime with its defining parameters jittered."""
    base = REGIME_THETAS[code]
    changes: Dict[str, float] = {}
    if base.beta > 1.0:
        changes["beta"] = float(rng.uniform(2.0, 4.0))
    elif base.beta > 0.0:
        changes["beta"] = float(rng.uniform(0.3, 0.8))
    if code == "BIAS":
        changes["b"] = float(rng.uniform(1.0, 2.0))
    elif code in ("OCP_O", "OCP_A"):
        changes["b"] = float(rng.uniform(4.0, 7.0))
    elif code == "HAB":
        changes["kappa"] = float(rng.uniform(3.5, 6.0))
    elif code == "PERS":
        changes["alpha_neg"] = float(rng.uniform(0.02, 0.15))
    return dataclasses.replace(base, **changes)


def generate_episode(seed: int, n_options: int = 3, variant: str = "v1") -> Dict[str, np.ndarray]:
    """One full teacher episode with tokens and per-step ground-truth latents.

    Per-option quantities are stored in displayed-slot order. All latents
    describe the state before the step's choice.

    Variant "v1" uses fixed payoffs and the fixed manipulations. In it a notice
    about the target alone always means devaluation, so the stated number
    carries no information. Variant "v2" samples payoffs per episode and adds
    stated value changes with a sampled value for the target or another option,
    so that selection has to depend on the numbers.
    """
    if variant not in DATA_VARIANTS:
        raise ValueError(f"unknown data variant: {variant}")
    rng = np.random.default_rng([seed, 7])
    code, theta = sample_theta(rng)
    manip_idx = int(rng.integers(len(MANIPULATION_ORDER)))
    t_acq = int(rng.choice(T_ACQ_CHOICES))
    if variant == "v2":
        if rng.random() < STATED_VALUE_WEIGHT:
            manip_idx = len(MANIPULATION_ORDER) + int(rng.integers(2))
        target_points = int(rng.integers(2, 6))
        alt_points = int(rng.integers(0, min(target_points, 3)))
        stated = int(rng.integers(0, 6))
        spec = EnvSpec(
            family="F1", n_options=n_options, t_acq=t_acq, seed=seed, target_points=target_points, alt_points=alt_points
        )
    else:
        stated = 0
        spec = EnvSpec(family="F1", n_options=n_options, t_acq=t_acq, seed=seed)
    if manip_idx < len(MANIPULATION_ORDER):
        manipulation = MANIPULATION_ORDER[manip_idx]
    else:
        manipulation = (MV, MA)[manip_idx - len(MANIPULATION_ORDER)](stated)
    env = ChoiceEnv(spec)
    policy = BehaviourPolicy(theta, n_options)
    n_steps = spec.t_acq + spec.t_test
    k = n_options

    order = np.zeros((MAX_STEPS, k), dtype=np.int8)
    slot = {name: np.zeros((MAX_STEPS, k), dtype=np.float32) for name in ("v", "q", "h", "z", "probs")}
    chosen_slot = np.full((MAX_STEPS,), -1, dtype=np.int8)
    target_slot = np.full((MAX_STEPS,), -1, dtype=np.int8)
    d = np.zeros((MAX_STEPS,), dtype=np.float32)
    tag = np.full((MAX_STEPS,), -1, dtype=np.int8)
    points = np.full((MAX_STEPS,), -1, dtype=np.int8)

    n_first_tag = 0
    for t in range(n_steps):
        if t == spec.t_acq:
            for n in env.apply(manipulation):
                policy.observe_notice(n.option, n.points, n.rstar)
        obs = env.begin_step()
        probs = policy.action_probs(obs)
        lat = policy.latents(obs.cue)
        if t < spec.t_acq:
            probs = (1.0 - ACQ_EPSILON) * probs + ACQ_EPSILON / k
        action = int(rng.choice(k, p=probs))
        outcome = env.act(action)
        policy.observe_outcome(obs, action, outcome)

        order[t] = obs.order
        for name in ("v", "q", "h", "z"):
            slot[name][t] = lat[name][obs.order]
        slot["probs"][t] = probs[obs.order]
        chosen_slot[t] = obs.order.index(action)
        target_slot[t] = obs.order.index(env.target)
        d[t] = (n_first_tag + 0.5) / (t + 1.0)  # posterior mean of the tag rate
        tag[t] = TAGS.index(outcome.tag)
        n_first_tag += int(outcome.tag == TAGS[0])
        points[t] = outcome.points
    if manipulation.code.startswith("M7"):
        env.record_report(policy.report(rng))

    ids: List[int] = [BOS_ID]
    ch_pos = np.full((MAX_STEPS,), -1, dtype=np.int16)  # position whose next token is the chosen label
    out_pos = np.full((MAX_STEPS,), -1, dtype=np.int16)  # position whose next token is the points digit
    tag_pos = np.full((MAX_STEPS,), -1, dtype=np.int16)  # position whose next token is the tag
    step = 0
    for line in env.lines:
        if line.kind == "choice":
            ch_pos[step] = len(ids)
        elif line.kind == "outcome":
            out_pos[step] = len(ids)
            tag_pos[step] = len(ids) + 2
            step += 1
        ids += [TOKEN_ID[tok] for tok in line_tokens(line)]
    if len(ids) > MAX_TOKENS:
        raise RuntimeError(f"episode needs {len(ids)} tokens; MAX_TOKENS is {MAX_TOKENS}")
    tokens = np.full((MAX_TOKENS,), PAD_ID, dtype=np.int16)
    tokens[: len(ids)] = ids

    return {
        "tokens": tokens,
        "n_tokens": np.int16(len(ids)),
        "n_steps": np.int8(n_steps),
        "t_acq": np.int8(spec.t_acq),
        "regime": np.int8(REGIME_ORDER.index(code)),
        "manipulation": np.int8(manip_idx),
        "label_ids": np.asarray([TOKEN_ID[label] for label in env.labels], dtype=np.int16),
        "order": order,
        "chosen_slot": chosen_slot,
        "target_slot": target_slot,
        "v_slot": slot["v"],
        "q_slot": slot["q"],
        "h_slot": slot["h"],
        "z_slot": slot["z"],
        "probs_slot": slot["probs"],
        "d": d,
        "tag": tag,
        "points": points,
        "ch_pos": ch_pos,
        "out_pos": out_pos,
        "tag_pos": tag_pos,
    }


def generate_dataset(
    seed_start: int, n: int, tokens_only: bool = False, variant: str = "v1"
) -> Dict[str, np.ndarray]:
    """Stack `n` episodes with consecutive seeds starting at `seed_start`."""
    keys = ("tokens", "n_tokens") if tokens_only else None
    rows: Dict[str, List[np.ndarray]] = {}
    for i in range(n):
        ep = generate_episode(seed_start + i, variant=variant)
        for key in keys or ep.keys():
            rows.setdefault(key, []).append(ep[key])
    return {key: np.stack(values, axis=0) for key, values in rows.items()}
