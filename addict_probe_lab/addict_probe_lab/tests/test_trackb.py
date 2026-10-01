from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from addict_probe.persistence.env import M0, M1, M2, EnvSpec  # noqa: E402
from addict_probe.persistence.grammar import LABEL_POOLS  # noqa: E402
from addict_probe.persistence.prefixes import build_prefix, restore_prefix  # noqa: E402
from addict_probe.persistence.rollout import run_test  # noqa: E402

SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots"


def _tokenizer_dir():
    if not SNAPSHOTS.exists():
        pytest.skip("Qwen3.5 tokenizer not available locally")
    return next(SNAPSHOTS.iterdir())


@pytest.fixture(scope="module")
def lm():
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from addict_probe.trackb.lm import HookedLM

    tok = AutoTokenizer.from_pretrained(_tokenizer_dir())
    cfg = Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
    )
    torch.manual_seed(0)
    return HookedLM(Qwen3_5ForCausalLM(cfg), tok, "cpu")


@pytest.fixture(scope="module")
def pool():
    from addict_probe.trackb.collect import single_token_pool

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_tokenizer_dir())
    return single_token_pool(tok, "train")


def _record(pool, seed=3, t_acq=8, k=3):
    return build_prefix(EnvSpec(family="F1", n_options=k, t_acq=t_acq, seed=seed, label_pool=pool))


def test_label_pool_keeps_only_single_token_labels(lm, pool):
    assert 0 < len(pool) < len(LABEL_POOLS["train"])
    assert all(lm.is_single_token(" " + label) for label in pool)
    assert not lm.is_single_token(" fep")  # known multi-token label in this tokenizer
    with pytest.raises(ValueError):
        lm.single_token_id(" fep")


def test_recorded_sites_match_hidden_states(lm):
    ids = lm.encode("cue: K7\noptions: [mul] cost 0\nchoice:").ids
    _, acts = lm.run([ids], read_positions=[list(range(len(ids)))], sites=list(range(lm.n_layers)))
    with torch.no_grad():
        hs = lm.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    for site in range(lm.n_layers):  # the last entry of hidden_states has the final norm applied
        assert torch.allclose(acts[site][0], hs[site][0].float(), atol=1e-5)


def test_padding_does_not_change_outputs(lm):
    a = lm.encode("choice: mul\noutcome: 3 points, 1 zorb, tag u\nchoice:").ids
    b = lm.encode("choice:").ids
    together, acts = lm.run([a, b], read_positions=[[len(a) - 1], [len(b) - 1]], sites=[2])
    alone_a, acts_a = lm.run([a], read_positions=[[len(a) - 1]], sites=[2])
    alone_b, acts_b = lm.run([b], read_positions=[[len(b) - 1]], sites=[2])
    assert torch.allclose(together[0], alone_a[0], atol=1e-4)
    assert torch.allclose(together[1], alone_b[0], atol=1e-4)
    assert torch.allclose(acts[2][1], acts_b[2][0], atol=1e-4)


def test_edits_apply_and_identity_edit_changes_nothing(lm):
    a = lm.encode("options: [mul] cost 0 | [dax] cost 0\nchoice:").ids
    b = lm.encode("options: [dax] cost 2 | [mul] cost 0\nnotice: mul now yields 0 points\nchoice:").ids
    plain, _ = lm.run([a])
    same, _ = lm.run([a], edit=lambda s, x: x)
    assert torch.allclose(plain, same)
    # After the last layer the prediction depends only on that position, so transplanting
    # the last residual of b into a's last position reproduces b's prediction.
    last = lm.n_layers
    _, src = lm.run([b], read_positions=[[len(b) - 1]], sites=[last])

    def transplant(site, x):
        if site == last:
            x = x.clone()
            x[:, -1] = src[last][:, 0].to(x.dtype)
        return x

    edited, _ = lm.run([a], edit=transplant)
    target, _ = lm.run([b])
    assert torch.allclose(edited, target, atol=1e-4)


def test_tstar_is_the_last_token_of_the_last_outcome_line(lm, pool):
    from addict_probe.trackb.collect import WINDOW, collect_batch, prefix_positions

    record = _record(pool, t_acq=16)
    ids, pos = prefix_positions(lm, record)
    assert len(pos["window"]) == WINDOW and pos["window"][-1] == pos["tstar"][0]
    enc = lm.encode(record["text"])
    last_line = record["text"].rstrip("\n").splitlines()[-1]
    assert last_line.startswith("outcome: ")
    lo, hi = enc.offsets[pos["tstar"][0]]
    assert record["text"][lo:hi].strip() and hi == record["tstar_char"]
    acts = collect_batch(lm, [record, _record(pool, seed=4, t_acq=8)], [0, 2], ["tstar", "window"])
    assert acts[(2, "tstar")].shape == (2, 1, lm.d_model) and acts[(0, "window")].shape == (2, WINDOW, lm.d_model)


def test_lm_policy_runs_through_forked_test_phases(lm, pool):
    from addict_probe.trackb.labels import prefix_labels
    from addict_probe.trackb.policy import LMPolicy

    record = _record(pool)
    env, _ = restore_prefix(record)
    text_before = env.text()
    policy = LMPolicy(lm)
    res = run_test(env, policy, M2, np.random.default_rng(0), n_rollouts=2, value_query=True)
    assert env.text() == text_before
    assert abs(res.first_probs.sum() - 1.0) < 1e-9 and res.actions.shape == (2, env.spec.t_test)
    assert all(0.0 < m <= 1.0 for m in policy.valid_mass)
    # The first-step probabilities are the model's distribution after "choice:" on the forked transcript.
    first = run_test(env, policy, M1, np.random.default_rng(0), n_rollouts=0).first_probs
    again = run_test(env, policy, M1, np.random.default_rng(1), n_rollouts=0).first_probs
    assert np.allclose(first, again)
    labels = prefix_labels(lm, record, n_rollouts=1, rollout_codes=("M1",))
    assert 0.0 <= labels["PI"] <= 1.0 and "first_p_M0" in labels and "first_p_MV2" in labels


def test_store_round_trip(tmp_path):
    from addict_probe.trackb.store import ActivationStore

    store = ActivationStore(tmp_path)
    a = np.random.default_rng(0).normal(size=(3, 1, 8)).astype(np.float32)
    b = np.random.default_rng(1).normal(size=(2, 1, 8)).astype(np.float32)
    store.write("base", 5, "tstar", [{"prefix_id": f"p{i}"} for i in range(3)], a)
    store.write("base", 5, "tstar", [{"prefix_id": f"q{i}"} for i in range(2)], b)
    store.write("sham", 5, "tstar", [{"prefix_id": "p0"}], a[:1])
    rows, acts = store.read("base", 5, "tstar")
    assert [r["prefix_id"] for r in rows] == ["p0", "p1", "p2", "q0", "q1"]
    assert np.allclose(acts, np.concatenate([a, b]), atol=1e-2) and acts.dtype == np.float16
    assert store.stored_prefix_ids("sham", 5, "tstar") == {"p0"}
    with pytest.raises(KeyError):
        store.read("base", 6, "tstar")
    with pytest.raises(ValueError):
        store.write("a__b", 5, "tstar", [{}], a[:1])
