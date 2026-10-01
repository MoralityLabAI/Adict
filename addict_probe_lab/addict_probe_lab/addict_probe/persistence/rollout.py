from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .env import ChoiceEnv, Manipulation, Outcome, StepObs


class Policy(Protocol):
    """Anything that can be driven through an episode: a teacher or a model."""

    def fork(self) -> "Policy": ...

    def observe_notice(self, option: int, points: float, rstar: float) -> None: ...

    def action_probs(self, obs: StepObs) -> np.ndarray: ...

    def observe_outcome(self, obs: StepObs, action: int, outcome: Outcome) -> None: ...

    def value_query(self, obs: StepObs, rng: np.random.Generator) -> int: ...

    def report(self, rng: np.random.Generator) -> bool: ...


@dataclass
class StepRecord:
    obs: StepObs
    action: int
    outcome: Outcome


@dataclass
class ForkResult:
    manipulation: Manipulation
    first_probs: np.ndarray  # exact, per option index
    actions: np.ndarray  # [n_rollouts, t_test] option indices
    selected_marked: np.ndarray  # [n_rollouts]
    reports: List[bool]  # reported "selected marked option?" per rollout
    value_query_correct: Optional[bool]


def run_acquisition(
    env: ChoiceEnv, policy: Policy, rng: np.random.Generator, epsilon: float = 0.0
) -> List[StepRecord]:
    """Run the acquisition phase, mixing the policy with uniform exploration."""
    records: List[StepRecord] = []
    bind(policy, env)
    while not env.in_test:
        obs = env.begin_step()
        probs = policy.action_probs(obs)
        probs = (1.0 - epsilon) * probs + epsilon / probs.shape[0]
        action = int(rng.choice(probs.shape[0], p=probs))
        outcome = env.act(action)
        policy.observe_outcome(obs, action, outcome)
        records.append(StepRecord(obs=obs, action=action, outcome=outcome))
    return records


def bind(policy: Policy, env: ChoiceEnv) -> Policy:
    """Point a policy that reads the transcript (a language model) at the environment it acts in.

    Policies that keep their own state ignore this.
    """
    attach = getattr(policy, "attach", None)
    if attach is not None:
        attach(env)
    return policy


def run_scripted(env: ChoiceEnv, actions: Sequence[int]) -> List[StepRecord]:
    """Force a fixed action sequence through the environment."""
    records: List[StepRecord] = []
    for action in actions:
        obs = env.begin_step()
        outcome = env.act(int(action))
        records.append(StepRecord(obs=obs, action=int(action), outcome=outcome))
    return records


def replay(policy: Policy, records: Sequence[StepRecord]) -> None:
    """Feed a recorded history to a policy without sampling from it."""
    for rec in records:
        policy.observe_outcome(rec.obs, rec.action, rec.outcome)


def run_test(
    env: ChoiceEnv,
    policy: Policy,
    manip: Manipulation,
    rng: np.random.Generator,
    n_rollouts: int = 4,
    value_query: bool = False,
    stream: int = 0,
) -> ForkResult:
    """Fork env and policy at the end of acquisition, apply `manip`, run the test phase.

    The inputs are left untouched. All rollouts share the first test step, whose
    action probabilities are returned exactly.
    """
    env_f = env.fork(stream)
    pol_f = bind(policy.fork(), env_f)
    for n in env_f.apply(manip):
        pol_f.observe_notice(n.option, n.points, n.rstar)
    first_obs = env_f.begin_step()
    first_probs = np.asarray(pol_f.action_probs(first_obs), dtype=np.float64)

    vq: Optional[bool] = None
    if value_query:
        vq = bind(pol_f.fork(), env_f).value_query(first_obs, rng) in env_f.best_options(first_obs)

    t_test = env.spec.t_test
    actions = np.zeros((n_rollouts, t_test), dtype=np.int64)
    selected_marked = np.zeros((n_rollouts,), dtype=bool)
    reports: List[bool] = []
    for k in range(n_rollouts):
        e = env_f.fork(1000 + k)
        p = bind(pol_f.fork(), e)
        obs, probs = first_obs, first_probs
        for j in range(t_test):
            if j > 0:
                obs = e.begin_step()
                probs = np.asarray(p.action_probs(obs), dtype=np.float64)
            action = int(rng.choice(probs.shape[0], p=probs))
            selected_marked[k] = selected_marked[k] or bool(obs.marked[action])
            p.observe_outcome(obs, action, e.act(action))
            actions[k, j] = action
        reports.append(bool(p.report(rng)))
    return ForkResult(
        manipulation=manip,
        first_probs=first_probs,
        actions=actions,
        selected_marked=selected_marked,
        reports=reports,
        value_query_correct=vq,
    )
