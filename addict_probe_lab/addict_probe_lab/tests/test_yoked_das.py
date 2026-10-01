from __future__ import annotations

import numpy as np
import pytest

from addict_probe.persistence.tokens import TOKEN_ID, VOCAB
from addict_probe.persistence.yoked import CORNERS, MIN_TARGET_COUNT_GAP, generate_yoked, generate_yoked_dataset


@pytest.fixture(scope="module")
def yoked_sets():
    return [generate_yoked(930_000 + i, scripts="random") for i in range(25)]


def test_teacher_scripts_keep_corners_aligned_and_vary_the_history():
    counts = []
    for i in range(25):
        y = generate_yoked(960_000 + i)
        tok = y["tokens"]
        assert np.array_equal(tok[0, y["opt_pos"]], y["slot_ids"])
        assert not np.array_equal(tok[0], tok[2])  # the two action scripts differ
        for a in (0, 1):  # same script under the other payoff condition: same choices
            lo, hi = CORNERS.index((a, 0)), CORNERS.index((a, 1))
            diff = np.flatnonzero(tok[lo] != tok[hi])
            assert all(VOCAB[int(t)] == "OUT" for t in tok[lo, diff - 1])
        counts += [int(n) for n in y["n_target"]]
    assert min(counts) <= 3 and max(counts) >= 9  # histories range from rarely to mostly the target


def test_corners_differ_only_in_the_factor_that_was_varied(yoked_sets):
    digits = {TOKEN_ID[str(i)] for i in range(10)}
    for y in yoked_sets:
        tok = y["tokens"]
        assert abs(int(y["n_target"][0]) - int(y["n_target"][1])) >= MIN_TARGET_COUNT_GAP
        assert tok[0, -1] == TOKEN_ID["CH"] and int(y["ch_pos"]) == tok.shape[1] - 1
        assert np.array_equal(tok[0, y["opt_pos"]], y["slot_ids"])
        for c, (a, p) in enumerate(CORNERS):
            # Same action script, other payoff condition: only points digits change.
            other_payoff = CORNERS.index((a, 1 - p))
            diff = np.flatnonzero(tok[c] != tok[other_payoff])
            assert diff.size > 0 and all(int(t) in digits for t in tok[c, diff])
            assert all(VOCAB[int(t)] == "OUT" for t in tok[c, diff - 1])
            # Same payoff condition, other action script: the final options line is unchanged.
            other_action = CORNERS.index((1 - a, p))
            assert np.array_equal(tok[c, y["opt_pos"][0] - 1 :], tok[other_action, y["opt_pos"][0] - 1 :])
            assert not np.array_equal(tok[c], tok[other_action])


def test_latents_follow_their_own_factor_only(yoked_sets):
    for y in yoked_sets:
        h, v = y["h_slot"], y["v_slot"]
        for a in (0, 1):  # the choice kernel ignores payoffs
            assert np.array_equal(h[CORNERS.index((a, 0))], h[CORNERS.index((a, 1))])
        for p in (0, 1):  # every option was tried, so the points record ignores the script
            assert np.array_equal(v[CORNERS.index((0, p))], v[CORNERS.index((1, p))])
        assert not np.array_equal(h[0], h[2]) and not np.array_equal(v[0], v[1])


def test_notice_sets_differ_across_levels_only_in_the_stated_points():
    for i in range(15):
        y = generate_yoked(950_000 + i, factor="notice", scripts="random")
        tok, note = y["tokens"], int(y["note_pos"])
        assert VOCAB[int(tok[0, note - 2])] == "NOTE" and note < int(y["opt_pos"][0])
        for a in (0, 1):
            lo, hi = CORNERS.index((a, 0)), CORNERS.index((a, 1))
            assert list(np.flatnonzero(tok[lo] != tok[hi])) == [note]
            assert (VOCAB[int(tok[lo, note])], VOCAB[int(tok[hi, note])]) == ("3", "0")
            assert np.array_equal(y["h_slot"][lo], y["h_slot"][hi])
            t = int(y["target_slot"])
            assert (y["v_slot"][lo][t], y["v_slot"][hi][t]) == (3.0, 0.0)
    assert int(generate_yoked(950_000)["note_pos"]) == -1  # the payoff factor has no notice


def test_altswap_exchanges_the_non_target_options_and_nothing_else():
    lopsided = 0
    for i in range(20):
        y = generate_yoked(980_000 + i, factor="notice", scripts="altswap")
        tok, t = y["tokens"], int(y["target_slot"])
        others = [s for s in range(3) if s != t]
        ids = y["slot_ids"]
        exchange = {int(ids[others[0]]): int(ids[others[1]]), int(ids[others[1]]): int(ids[others[0]])}
        # Same notice level, other script: every difference is one non-target label replaced by the other.
        diff = np.flatnonzero(tok[0] != tok[2])
        assert diff.size > 0
        assert all(exchange[int(tok[0, p])] == int(tok[2, p]) for p in diff)
        assert all(VOCAB[int(tok[0, p - 1])] == "CH" for p in diff)
        # The points record is identical; the choice kernel is exchanged between the two options.
        assert np.array_equal(y["v_slot"][0], y["v_slot"][2])
        assert y["h_slot"][0][t] == y["h_slot"][2][t]
        assert np.allclose(y["h_slot"][0][others], y["h_slot"][2][others[::-1]])
        lopsided += int(abs(float(y["h_slot"][0][others[0]]) - float(y["h_slot"][0][others[1]])) > 0.05)
    assert lopsided >= 15


torch = pytest.importorskip("torch")

from addict_probe.tracka.das import (  # noqa: E402
    SubspacePair,
    counterfactual_corner,
    evaluate_bases,
    fixed_bases,
    prepare,
    single_factor_ceiling,
    train_subspaces,
)
from addict_probe.tracka.model import GPTConfig, TinyGPT, parse_site, site_name  # noqa: E402


def test_counterfactual_corner_takes_one_factor_from_the_opposite_corner():
    base = torch.arange(4)
    for variable, pick in (("action", 0), ("payoff", 1)):
        cf = counterfactual_corner(base, variable)
        for b, c in zip(base.tolist(), cf.tolist()):
            source = CORNERS[3 - b]
            expected = tuple(source[i] if i == pick else CORNERS[b][i] for i in (0, 1))
            assert CORNERS[c] == expected


def test_component_sites_are_exposed_and_a_whole_block_swap_equals_swapping_all_its_heads():
    torch.manual_seed(1)
    data = generate_yoked_dataset(990_000, 6, factor="notice", scripts="altswap")
    model = TinyGPT(
        GPTConfig(vocab_size=len(VOCAB), block_size=data["tokens"].shape[2], n_layer=2, n_head=2, d_model=32)
    ).eval()
    base = torch.as_tensor(data["tokens"][:, 0].astype(np.int64))
    source = torch.as_tensor(data["tokens"][:, 2].astype(np.int64))
    with torch.no_grad():
        plain, acts = model(base, return_resid=True)
        _, src = model(source, return_resid=True)
    assert set(acts) == {0, 0.5, 1, 1.5, 2, ("head", 1, 0), ("head", 1, 1), ("head", 2, 0), ("head", 2, 1)}
    assert acts[("head", 2, 0)].shape[-1] == 16 and acts[1.5].shape[-1] == 32
    for site in acts:
        assert parse_site(site_name(site)) == site

    # Recording activations and writing them back unchanged leaves the output alone.
    with torch.no_grad():
        same, _ = model(base, edit=lambda s, x: acts[s])
    assert torch.allclose(same, plain, atol=1e-5)

    # Replacing every head output of block 2 gives the source's attention output on the base's
    # block-1 residual; the mid-block residual then differs from the base only by that.
    with torch.no_grad():
        _, swapped = model(base, edit=lambda s, x: src[s] if isinstance(s, tuple) and s[1] == 2 else x, return_resid=True)
    expected = acts[1] + model.blocks[1].proj(torch.cat([src[("head", 2, 0)], src[("head", 2, 1)]], dim=-1))
    assert torch.allclose(swapped[1.5], expected, atol=1e-5)


def test_subspaces_are_orthonormal_and_interchange_metrics_are_anchored():
    torch.manual_seed(0)
    data = generate_yoked_dataset(940_000, 16)
    model = TinyGPT(
        GPTConfig(vocab_size=len(VOCAB), block_size=data["tokens"].shape[2], n_layer=2, n_head=2, d_model=32)
    ).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    site = model.n_sites - 1
    y = prepare(model, data, site, "choice", "cpu")
    assert y.logp.shape == (16, 4, 3) and y.resid.shape == (16, 4, 1, 32)

    pair = train_subspaces(model, y, site, k=3, steps=3, batch_size=8)
    q = torch.cat(list(pair.bases().values()), dim=1)
    assert torch.allclose(q.T @ q, torch.eye(6), atol=1e-5)

    # At the last site the prediction is local, so the site carries each factor in full.
    ceiling = single_factor_ceiling(model, y, site)
    assert ceiling["ceiling_action"] == pytest.approx(1.0, abs=1e-4)
    assert ceiling["ceiling_payoff"] == pytest.approx(1.0, abs=1e-4)
    # At the embedding site the choice marker is the same token in every corner, so it carries nothing.
    y0 = prepare(model, data, 0, "choice", "cpu")
    assert single_factor_ceiling(model, y0, 0)["ceiling_action"] == pytest.approx(0.0, abs=1e-4)

    # An empty swap leaves the output as it was: nothing recovered, nothing lost.
    zero = np.zeros((32, 1))
    none = evaluate_bases(model, y, site, fixed_bases(zero, zero, "cpu"))
    assert none["recovered_action"] == pytest.approx(0.0, abs=1e-5)
    assert none["cross_payoff"] == pytest.approx(0.0, abs=1e-5)

    # At the last site the prediction is local, so swapping the whole residual
    # yields the opposite corner's output, whichever variable is scored.
    eye = np.eye(32)
    full = evaluate_bases(model, y, site, fixed_bases(eye, eye, "cpu"))
    opposite = y.logp[:, [3, 2, 1, 0]]
    for variable, cf in (("action", [2, 3, 0, 1]), ("payoff", [1, 0, 3, 2])):
        target = y.logp[:, cf]
        kl_swap = (target.exp() * (target - opposite)).sum(-1).sum()
        kl_none = (target.exp() * (target - y.logp)).sum(-1).sum()
        assert full[f"recovered_{variable}"] == pytest.approx(float(1 - kl_swap / kl_none), abs=1e-3)
