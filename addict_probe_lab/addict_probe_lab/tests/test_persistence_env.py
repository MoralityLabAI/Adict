from __future__ import annotations

import numpy as np
import pytest

from addict_probe.persistence.behaviour_model import (
    PREFIX_THETA,
    REFERENCE_THETA,
    REGIME_THETAS,
    BehaviourPolicy,
)
from addict_probe.persistence.env import (
    M0,
    M1,
    M2,
    M3,
    M4,
    M4R,
    M5,
    M6,
    M7OFF,
    M7ON,
    MA,
    MEQ,
    ChoiceEnv,
    EnvSpec,
)
from addict_probe.persistence.grammar import CUE_POOLS, LABEL_POOLS, RSTAR_POOLS, parse, render
from addict_probe.persistence.rollout import replay, run_acquisition, run_scripted, run_test


def _acquired(spec: EnvSpec):
    env = ChoiceEnv(spec)
    rng = np.random.default_rng(spec.seed)
    records = run_acquisition(env, BehaviourPolicy(PREFIX_THETA, spec.n_options), rng, epsilon=0.15)
    return env, records


def test_vocabulary_pools_are_disjoint():
    for pools in (LABEL_POOLS, CUE_POOLS, RSTAR_POOLS):
        assert not set(pools["train"]) & set(pools["test"])


@pytest.mark.parametrize("family", ["F1", "F2"])
@pytest.mark.parametrize("manip", [M0, M1, M2, M3, M4, M4R, M5, MEQ, M6, M7ON, M7OFF])
def test_transcript_round_trips_through_the_grammar(family, manip):
    env, _ = _acquired(EnvSpec(family=family, t_acq=8, seed=3))
    env.apply(manip)
    for _ in range(env.spec.t_test):
        obs = env.begin_step()
        env.act(obs.order[0])
    env.record_query(0)
    env.record_report(True)
    text = env.text()
    assert parse(text) == env.lines
    assert render(parse(text)) == text


def test_same_spec_and_actions_give_the_same_transcript():
    env_a, records = _acquired(EnvSpec(family="F2", seed=11))
    env_b = ChoiceEnv(env_a.spec)
    run_scripted(env_b, [rec.action for rec in records])
    assert env_b.text() == env_a.text()


def test_target_is_best_and_delivers_rstar_during_acquisition():
    for seed in range(20):
        env = ChoiceEnv(EnvSpec(seed=seed, n_options=4))
        points, rstar = env.points[env.cue_main], env.rstar[env.cue_main]
        assert int(np.argmax(points)) == env.target
        assert rstar[env.target] == 1 and rstar.sum() == 1


def test_fork_leaves_the_parent_untouched():
    env, _ = _acquired(EnvSpec(seed=5))
    before = (env.t, len(env.lines), env.points[env.cue_main].copy())
    child = env.fork(1)
    child.apply(M1)
    child.act(child.begin_step().order[0])
    assert (env.t, len(env.lines)) == before[:2]
    assert np.array_equal(env.points[env.cue_main], before[2])
    assert child.points[child.cue_main][child.target] == 0


def test_manipulations_change_what_they_claim():
    env, _ = _acquired(EnvSpec(seed=9))
    tau, r, k = env.target, env.alt, env.spec.n_options

    e = env.fork(0)
    assert [n.option for n in e.apply(M1)] == [tau]
    assert e.points[e.cue_main][tau] == 0 and e.rstar[e.cue_main][tau] == 1

    e = env.fork(0)
    assert [n.option for n in e.apply(MA(5))] == [r]
    assert e.points[e.cue_main][r] == 5 and e.points[e.cue_main][tau] == env.spec.target_points

    e = env.fork(0)
    assert e.apply(M2) == []  # silent
    assert e.points[e.cue_main][tau] == 0 and e.points[e.cue_main][r] == env.spec.target_points

    e = env.fork(0)
    e.apply(M3)
    costs = []
    for _ in range(e.spec.t_test):
        obs = e.begin_step()
        costs.append(int(obs.cost[tau]))
        assert obs.cost.sum() == obs.cost[tau]
        e.act(tau)
    assert costs == list(range(1, e.spec.t_test + 1))

    e = env.fork(0)
    e.apply(M4)
    assert e.points[e.cue_main][r] == env.spec.target_points and e.rstar[e.cue_main][r] == 1

    e = env.fork(0)
    e.apply(M4R)
    assert e.points[e.cue_main][tau] == env.spec.target_points and e.rstar[e.cue_main][r] == 1

    e = env.fork(0)
    assert len(e.apply(M5)) == k
    assert e.rstar[e.cue_main].sum() == 0 and len(set(e.points[e.cue_main])) == 1

    e = env.fork(0)
    e.apply(MEQ)
    assert e.rstar[e.cue_main][tau] == 1 and len(set(e.points[e.cue_main])) == 1

    e = env.fork(0)
    assert e.apply(M6) == []
    obs = e.begin_step()
    assert obs.cue == e.cue_novel
    assert np.array_equal(e.points[e.cue_novel], env.points[env.cue_main])

    for manip, audit in ((M7ON, True), (M7OFF, False)):
        e = env.fork(0)
        e.apply(manip)
        obs = e.begin_step()
        assert obs.audit is audit and obs.marked[tau] and obs.marked.sum() == 1


def test_manipulation_is_rejected_outside_the_phase_boundary():
    env = ChoiceEnv(EnvSpec(seed=1))
    with pytest.raises(RuntimeError):
        env.apply(M1)


def test_policy_probabilities_are_valid_and_reference_takes_the_best_option():
    env, records = _acquired(EnvSpec(seed=2))
    for theta in list(REGIME_THETAS.values()) + [REFERENCE_THETA]:
        policy = BehaviourPolicy(theta, env.spec.n_options)
        replay(policy, records)
        res = run_test(env, policy, M0, np.random.default_rng(0), n_rollouts=1)
        assert res.first_probs.shape == (env.spec.n_options,)
        assert np.all(res.first_probs >= 0) and abs(res.first_probs.sum() - 1.0) < 1e-9
    reference = BehaviourPolicy(REFERENCE_THETA, env.spec.n_options)
    replay(reference, records)
    res = run_test(env, reference, M0, np.random.default_rng(0), n_rollouts=2)
    assert res.first_probs[env.target] > 0.999
    assert np.all(res.actions == env.target)


def test_run_test_does_not_advance_its_inputs():
    env, records = _acquired(EnvSpec(seed=4))
    policy = BehaviourPolicy(REGIME_THETAS["RMAX"], env.spec.n_options)
    replay(policy, records)
    q_before = policy.q[env.cue_main].copy()
    run_test(env, policy, M2, np.random.default_rng(0), n_rollouts=3)
    assert env.test_step == 0 and np.array_equal(policy.q[env.cue_main], q_before)
