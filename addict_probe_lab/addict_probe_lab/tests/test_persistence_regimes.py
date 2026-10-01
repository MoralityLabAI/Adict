from __future__ import annotations

import json

import numpy as np
import pytest

from addict_probe.persistence import REGIME_THETAS, BehaviourPolicy, assigned_regimes, measure_indices
from addict_probe.persistence.env import EnvSpec
from addict_probe.persistence.prefixes import build_prefix, restore_prefix
from addict_probe.persistence.splits import (
    assert_group_disjoint,
    grouped_split,
    open_split,
    write_manifest,
)


@pytest.fixture(scope="module")
def teacher_indices():
    factories = {code: (lambda k, th=theta: BehaviourPolicy(th, k)) for code, theta in REGIME_THETAS.items()}
    return measure_indices(factories, n_prefixes=40, n_rollouts=4, seed=3, kappa_samples=30)


def test_each_teacher_falls_in_its_own_region_only(teacher_indices):
    for name, ix in teacher_indices.items():
        expected = ["RMAX"] if name == "reference" else [name]
        assert assigned_regimes(ix) == expected, name


def test_persistence_index_orders_regimes_as_defined(teacher_indices):
    pi = {name: ix["PI"] for name, ix in teacher_indices.items()}
    assert pi["RMAX"] < 0.2 and pi["BIAS"] < 0.25
    assert pi["RMAX"] < pi["PERS"] < pi["HAB"] < pi["OCP_O"]
    assert pi["OCP_O"] > 0.9 and pi["OCP_A"] > 0.9


def test_prefix_record_restores_to_the_same_transcript():
    record = build_prefix(EnvSpec(family="F2", n_options=4, t_acq=16, seed=21))
    env, records = restore_prefix(json.loads(json.dumps(record)))
    assert env.in_test and env.test_step == 0 and len(records) == 16
    text = record["text"]
    assert len(record["step_end_chars"]) == 16
    assert record["tstar_char"] == len(text) - 1  # last outcome line ends the prefix
    for end in record["step_end_chars"]:
        assert text[end] == "\n" and "outcome: " in text[:end].splitlines()[-1]


def test_grouped_split_keeps_groups_together():
    rng = np.random.default_rng(0)
    groups = [int(g) for g in rng.integers(0, 50, size=400)]
    split = grouped_split(groups, seed=1)
    assert_group_disjoint(groups, split)
    assert sum(len(v) for v in split.values()) == len(groups)
    again = grouped_split(groups, seed=1)
    assert all(np.array_equal(split[k], again[k]) for k in split)
    with pytest.raises(AssertionError):
        assert_group_disjoint([1, 1], {"train": np.array([0]), "test": np.array([1])})


def test_sealed_split_opens_once_per_analysis_and_detects_edits(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train.write_text('{"a": 1}\n', encoding="utf-8")
    test.write_text('{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    written = write_manifest(manifest, {"train": train, "test": test}, sealed=("test",))
    assert written["files"]["test"]["records"] == 2

    assert open_split(manifest, "train") == train
    with pytest.raises(PermissionError):
        open_split(manifest, "test")
    assert open_split(manifest, "test", analysis_id="H1") == test
    with pytest.raises(PermissionError):
        open_split(manifest, "test", analysis_id="H1")
    assert open_split(manifest, "test", analysis_id="H3") == test

    test.write_text('{"a": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        open_split(manifest, "test", analysis_id="H4")
