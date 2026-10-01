from __future__ import annotations

import numpy as np
import pytest

from addict_probe.tracka.probing import (
    RidgeProbe,
    covariance_directions,
    fit_label_indexed,
    fit_selected,
    orthonormal_basis,
    permute_within_step,
    r2_score,
)

torch = pytest.importorskip("torch")

from addict_probe.persistence.teacher_data import generate_dataset  # noqa: E402
from addict_probe.persistence.tokens import VOCAB  # noqa: E402
from addict_probe.tracka.causal import interchange_factory, pca_basis, random_subspace  # noqa: E402
from addict_probe.tracka.evaluate import (  # noqa: E402
    imitation_metrics,
    model_outputs,
    option_token_positions,
    role_variables,
    step_mask,
)
from addict_probe.tracka.mlp_probe import mlp_probe_r2  # noqa: E402
from addict_probe.tracka.model import GPTConfig, TinyGPT  # noqa: E402


def test_ridge_probe_recovers_a_planted_linear_variable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 12)) * rng.uniform(0.5, 3.0, size=12)
    w = rng.normal(size=(12, 2))
    y = x @ w + 0.01 * rng.normal(size=(600, 2))
    parts = {"train": slice(0, 400), "val": slice(400, 500), "test": slice(500, 600)}
    probe, scores = fit_selected({k: x[s] for k, s in parts.items()}, {k: y[s] for k, s in parts.items()})
    assert scores["test"] > 0.99
    assert np.allclose(probe.directions(), w, atol=0.05)
    assert covariance_directions(x, y).shape == (12, 2)
    q = orthonormal_basis(probe.directions())
    assert np.allclose(q.T @ q, np.eye(2), atol=1e-9)


def test_label_indexed_probe_reads_values_bound_to_token_identity():
    rng = np.random.default_rng(2)
    n, d, n_labels, k = 900, 10, 5, 3
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(n_labels, d))
    w -= w.mean(axis=0)  # no component shared across labels
    token = np.stack([rng.permutation(n_labels)[:k] + 100 for _ in range(n)])
    value = np.einsum("nd,nkd->nk", x, w[token - 100])
    parts = {"train": slice(0, 600), "val": slice(600, 750), "test": slice(750, 900)}
    cut = lambda a: {s: a[sl] for s, sl in parts.items()}
    probes, scores = fit_label_indexed(cut(x), cut(token), cut(value))
    assert set(probes) == set(range(100, 105)) and scores["test"] > 0.99
    # The same values are not readable per displayed slot, since slots are shuffled.
    _, slot_scores = fit_selected(cut(x), cut(value))
    assert slot_scores["test"] < 0.2


def test_permutation_control_keeps_step_and_changes_episode():
    episode = np.repeat(np.arange(50), 4)
    step = np.tile(np.arange(4), 50)
    y = (episode * 10 + step).astype(np.float64)[:, None]
    perm = permute_within_step(y, episode, step, np.random.default_rng(1))
    assert np.array_equal(perm[:, 0] % 10, step)  # step preserved
    assert np.mean(perm[:, 0] // 10 != episode) > 0.9  # donor episode differs
    assert r2_score(y, y) == 1.0


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    data = generate_dataset(920_000, 12)
    max_len = int(data["n_tokens"].max())
    data["tokens"] = data["tokens"][:, :max_len]
    model = TinyGPT(GPTConfig(vocab_size=len(VOCAB), block_size=max_len, n_layer=2, n_head=2, d_model=32))
    return model, data


def test_model_outputs_have_one_row_per_step(tiny):
    model, data = tiny
    out = model_outputs(model, data, resid_sites=[0, 2], resid_key="tag_pos")
    n, steps, k = data["probs_slot"].shape
    assert out["slot_logp"].shape == (n, steps, k)
    assert np.allclose(np.exp(out["slot_logp"]).sum(axis=-1), 1.0, atol=1e-5)
    assert out["resid_2"].shape == (n, steps, 32)
    metrics = imitation_metrics(data, out)
    assert 0.0 < metrics["valid_mass"] < 1.0 and metrics["choice_kl_teacher_model"] > 0.0


def test_option_positions_and_role_variables_line_up_with_the_episode(tiny):
    _, data = tiny
    m = step_mask(data)
    pos = option_token_positions(data)  # raises if a position holds the wrong label
    assert np.all(pos[m] > 0) and np.all(pos[m] < data["ch_pos"][m][:, None])
    role = role_variables(data)
    rows, steps = np.nonzero(m)
    tslot = data["target_slot"][rows, steps]
    assert np.array_equal(role["h_target"][m], data["h_slot"][rows, steps, tslot])
    assert role["v_gap"].shape == m.shape and np.array_equal(role["d"], data["d"])


def test_mlp_probe_finds_a_variable_a_linear_probe_cannot():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 6)).astype(np.float32)
    y = (x[:, 0] * x[:, 1])[:, None]
    parts = {"train": slice(0, 2000), "val": slice(2000, 2500), "test": slice(2500, 3000)}
    xs, ys = {s: x[sl] for s, sl in parts.items()}, {s: y[sl] for s, sl in parts.items()}
    assert fit_selected(xs, ys)[1]["test"] < 0.1
    assert mlp_probe_r2(xs, ys, epochs=60, batch_size=256, device="cpu") > 0.8


def test_full_interchange_at_the_last_site_transplants_the_source_prediction(tiny):
    model, data = tiny
    n = data["tokens"].shape[0]
    rows = np.arange(n)
    site = model.n_sites - 1
    base = model_outputs(model, data, resid_sites=[site])
    step = np.full((n,), 3)
    src_ep = (rows + 1) % n
    positions = data["ch_pos"][rows, step]
    source = base[f"resid_{site}"][src_ep, step]

    # After the last block the prediction at a position depends on that position
    # alone, so swapping the whole residual must reproduce the source's logits.
    edit = interchange_factory(site, np.eye(32), positions, source, "cpu")(rows)
    tokens = torch.as_tensor(data["tokens"].astype(np.int64))
    with torch.no_grad():
        plain, _ = model(tokens)
        edited, _ = model(tokens, edit=edit)
    b = torch.arange(n)
    pos = torch.as_tensor(positions.astype(np.int64))
    src_pos = torch.as_tensor(data["ch_pos"][src_ep, step].astype(np.int64))
    assert torch.allclose(edited[b, pos], plain[torch.as_tensor(src_ep), src_pos], atol=1e-4)
    # Every other position is untouched.
    keep = torch.ones_like(tokens, dtype=torch.bool)
    keep[b, pos] = False
    assert torch.allclose(edited[keep], plain[keep], atol=1e-5)


def test_interchange_with_the_base_as_source_changes_nothing(tiny):
    model, data = tiny
    n = data["tokens"].shape[0]
    rows = np.arange(n)
    base = model_outputs(model, data, resid_sites=[1])
    step = np.full((n,), 2)
    m = step_mask(data)
    basis = random_subspace(pca_basis(base["resid_1"][m], 8), 3, np.random.default_rng(0))
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)
    factory = interchange_factory(1, basis, data["ch_pos"][rows, step], base["resid_1"][rows, step], "cpu")
    out = model_outputs(model, data, edit_factory=factory)
    assert np.allclose(out["slot_logp"], base["slot_logp"], atol=1e-4)
