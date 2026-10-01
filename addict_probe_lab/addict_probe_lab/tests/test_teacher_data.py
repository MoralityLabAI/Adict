from __future__ import annotations

import numpy as np
import pytest

from addict_probe.persistence.grammar import TAGS
from addict_probe.persistence.teacher_data import (
    MANIPULATION_CODES,
    MANIPULATION_ORDER,
    generate_dataset,
    generate_episode,
)
from addict_probe.persistence.tokens import PAD_ID, TOKEN_ID, VOCAB


@pytest.fixture(scope="module")
def episodes():
    return [generate_episode(900_000 + i) for i in range(60)]


def test_vocabulary_has_no_duplicate_tokens():
    assert len(set(VOCAB)) == len(VOCAB)


def test_episode_generation_is_deterministic(episodes):
    again = generate_episode(900_000)
    assert all(np.array_equal(episodes[0][k], again[k]) for k in again)


def test_token_positions_point_at_the_tokens_they_claim(episodes):
    for ep in episodes:
        toks = ep["tokens"]
        assert np.all(toks[int(ep["n_tokens"]) :] == PAD_ID)
        for t in range(int(ep["n_steps"])):
            chosen_option = ep["order"][t][ep["chosen_slot"][t]]
            assert toks[ep["ch_pos"][t]] == TOKEN_ID["CH"]
            assert toks[ep["ch_pos"][t] + 1] == ep["label_ids"][chosen_option]
            assert toks[ep["out_pos"][t]] == TOKEN_ID["OUT"]
            assert toks[ep["out_pos"][t] + 1] == TOKEN_ID[str(int(ep["points"][t]))]
            assert toks[ep["tag_pos"][t] + 1] == TOKEN_ID[TAGS[int(ep["tag"][t])]]


def test_latents_describe_the_state_before_each_choice(episodes):
    for ep in episodes:
        n, t_acq = int(ep["n_steps"]), int(ep["t_acq"])
        assert np.allclose(ep["probs_slot"][:n].sum(axis=1), 1.0, atol=1e-5)
        assert np.all(ep["h_slot"][0] == 0)  # no choice has been made yet
        first_tags = 0
        for t in range(n):
            assert ep["d"][t] == pytest.approx((first_tags + 0.5) / (t + 1.0))
            first_tags += int(ep["tag"][t] == 0)
            if t + 1 < n and t + 1 != t_acq:  # no notice between these two steps
                option = ep["order"][t][ep["chosen_slot"][t]]
                next_slot = list(ep["order"][t + 1]).index(option)
                assert ep["v_slot"][t + 1][next_slot] == ep["points"][t]


def test_report_token_is_present_only_for_prohibition_episodes(episodes):
    for ep in episodes:
        prohibited = MANIPULATION_ORDER[int(ep["manipulation"])].code.startswith("M7")
        assert (TOKEN_ID["REP"] in ep["tokens"]) == prohibited


def test_v2_episodes_vary_payoffs_and_state_sampled_values():
    eps = [generate_episode(970_000 + i, variant="v2") for i in range(80)]
    codes = {MANIPULATION_CODES[int(ep["manipulation"])] for ep in eps}
    assert {"MV", "MA"} <= codes
    payoffs = set()
    stated = set()
    for ep in eps:
        toks = ep["tokens"]
        n, t_acq = int(ep["n_steps"]), int(ep["t_acq"])
        acq_points = ep["points"][:t_acq]
        payoffs.add((int(acq_points.max()), int(acq_points.min())))
        for t in range(n):  # positions still point at the right tokens
            assert toks[ep["ch_pos"][t]] == TOKEN_ID["CH"]
            assert toks[ep["out_pos"][t] + 1] == TOKEN_ID[str(int(ep["points"][t]))]
        if MANIPULATION_CODES[int(ep["manipulation"])] in ("MV", "MA"):
            note = int(np.flatnonzero(toks == TOKEN_ID["NOTE"])[0])
            assert int(np.sum(toks == TOKEN_ID["NOTE"])) == 1
            stated.add(VOCAB[int(toks[note + 2])])
    assert len(payoffs) >= 5 and len(stated) >= 4


def test_dataset_stacks_episodes_and_can_drop_latents():
    full = generate_dataset(910_000, 5)
    light = generate_dataset(910_000, 5, tokens_only=True)
    assert set(light) == {"tokens", "n_tokens"}
    assert full["v_slot"].shape[0] == 5 and np.array_equal(full["tokens"], light["tokens"])
