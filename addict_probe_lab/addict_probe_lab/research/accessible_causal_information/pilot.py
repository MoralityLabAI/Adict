"""CPU audit of a learned sum representation under an unseen shear target.

Run from repository root:
python -m research.accessible_causal_information.pilot --out /tmp/transfer
All encoder training inputs are observed. Held-out pairs apply only to probes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def states(p: int):
    if type(p) is not int or p < 2:
        raise ValueError("modulus must be an integer >= 2")
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return torch.stack((a, b), 1), {"sum": (a+b) % p, "a": a, "shear": (2*a+b) % p}


def exact_risk(codes, labels) -> Fraction:
    """Uniform-state, one-hot squared Bayes risk; also conditional pair disagreement.

    Each second state is sampled uniformly from the first state's code fiber.
    Exhaustive diagnostic only: all labels are visible to this oracle.
    """
    if len(codes) != len(labels) or not len(codes):
        raise ValueError("equal nonempty code and label lengths required")
    groups = defaultdict(Counter)
    for code, label in zip(codes, labels):
        groups[tuple(code)][int(label)] += 1
    residual = Fraction()
    for counts in groups.values():
        size = sum(counts.values())
        residual += size - Fraction(sum(n*n for n in counts.values()), size)
    return residual / len(labels)


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        p, e, h, d = (m[k] for k in ("modulus", "embedding_width", "hidden_width", "code_width"))
        self.a = nn.Embedding(p, e)
        self.b = nn.Embedding(p, e)
        self.body = nn.Sequential(nn.Linear(2*e, h), nn.ReLU(), nn.Linear(h, d), nn.Tanh())
        self.read_sum = nn.Linear(d, p)
        self.read_a = nn.Linear(d, p)

    def forward(self, pairs):
        z = self.body(torch.cat((self.a(pairs[:, 0]), self.b(pairs[:, 1])), 1))
        return self.read_sum(z), self.read_a(z), z


def ridge_scores(z, labels, train, test, classes, penalty):
    """Fit standardization and the affine decoder using diagnostic training rows only."""
    z = z.double()
    mean, sd = z[train].mean(0), z[train].std(0, unbiased=False).clamp_min(1e-8)
    x = torch.cat((torch.ones(len(z), 1, dtype=torch.double), (z-mean)/sd), 1)
    regularizer = torch.eye(x.shape[1], dtype=torch.double) * penalty
    regularizer[0, 0] = 0
    y = F.one_hot(labels, classes).double()
    weights = torch.linalg.solve(x[train].T @ x[train] + regularizer, x[train].T @ y[train])
    predicted = x[test] @ weights
    return {"accuracy": float((predicted.argmax(1) == labels[test]).double().mean()),
            "onehot_mse": float((predicted-y[test]).square().sum(1).mean())}


def quantize_nested(z, train, dimensions, bits):
    """One training-fitted PCA and fixed coordinate scales shared across all prefixes."""
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError("coordinate bits must be between 1 and 16")
    if any(type(k) is not int or not 0 <= k <= z.shape[1] for k in dimensions):
        raise ValueError("invalid retained dimension")
    z = z.double()
    centered = z - z[train].mean(0)
    _, _, vh = torch.linalg.svd(centered[train], full_matrices=False)
    coords = centered @ vh.T
    sd = coords[train].std(0, unbiased=False).clamp_min(1e-8)
    levels = 2**bits - 1
    quantized = torch.round(((coords/(3*sd)).clamp(-1, 1)+1)*levels/2).to(torch.int64)
    return {k: quantized[:, :k] for k in dimensions}


def assess(model, pairs, labels, train, test, m, seed, condition):
    model.eval()
    with torch.no_grad():
        logits, _, z = model(pairs)
        present = float((logits.argmax(1) == labels["sum"]).double().mean())
        probes = {target: ridge_scores(z, labels[target], train, test, m["modulus"], m["ridge_penalty"])
                  for target in m["diagnostic_targets"]}
        g = torch.Generator().manual_seed(900000+seed)
        permuted = labels["a"].clone()
        permuted[train] = permuted[train][torch.randperm(len(train), generator=g)]
        probes["a_shuffled_training_labels"] = ridge_scores(z, permuted, train, test,
                                                           m["modulus"], m["ridge_penalty"])
        codes = quantize_nested(z, train, m["retained_dimensions"], m["coordinate_bits"])
        risks = []
        for k, code in codes.items():
            rows = code.tolist()
            risks.append({"dimensions": k, "state_code_bits": k*m["coordinate_bits"],
                          "attained_codes": len(set(map(tuple, rows))),
                          "risk": {name: str(exact_risk(rows, y.tolist())) for name, y in labels.items()}})
        # These tensors permit downstream replay; no weights/checkpoints are required.
        return {"seed": seed, "condition": condition, "present_accuracy_all_states": present,
                "fit_gate_met": present >= m["fit_gate_present_accuracy"],
                "probes": probes, "quantized_oracle": risks}, z.cpu()


def run(manifest_path: Path, out: Path):
    m = json.loads(manifest_path.read_text())
    if m.get("schema") != "learned-intervention-transfer/v0.1" or m["steps"] < 1:
        raise ValueError("invalid manifest")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Written before acquiring any learning outcomes. Unsigned local ordering receipt.
    sources = {p.name: digest(p) for p in (manifest_path, HERE/"pilot.py", HERE/"test_pilot.py")}
    (out/"start.json").write_text(json.dumps({"source_sha256": sources,
                    "protocol": m, "ordering": "local pre-run write; no external timestamp attestation"}, indent=2)+"\n")
    pairs, labels = states(m["modulus"])
    permutation = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(m["diagnostic_split_seed"]))
    cut = int(m["diagnostic_train_fraction"]*len(pairs))
    train, test = permutation[:cut].sort().values, permutation[cut:].sort().values
    records, cached = [], {}
    for seed in m["seeds"]:
        torch.manual_seed(seed)
        initial = Encoder(m)
        for condition in m["conditions"]:
            model = copy.deepcopy(initial)
            if condition != "initial":
                optimizer = torch.optim.AdamW(model.parameters(), lr=m["learning_rate"], weight_decay=m["weight_decay"])
                for _ in range(m["steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    logits, hidden_logits, _ = model(pairs)
                    loss = F.cross_entropy(logits, labels["sum"])
                    if condition == "task_and_a":
                        loss = loss + F.cross_entropy(hidden_logits, labels["a"])
                    loss.backward()
                    optimizer.step()
            row, hidden = assess(model, pairs, labels, train, test, m, seed, condition)
            records.append(row)
            cached[f"{seed}/{condition}"] = hidden
            print(json.dumps({"seed": seed, "condition": condition,
                  "present": row["present_accuracy_all_states"], "probes": row["probes"]}), flush=True)
    torch.save({"hidden": cached, "pairs": pairs, "labels": labels, "train": train, "test": test}, out/"activations.pt")
    result = {"schema": m["schema"], "source_sha256": sources, "protocol": m,
              "python": platform.python_version(), "torch": torch.__version__,
              "n_states": len(pairs), "probe_train_rows": len(train), "probe_test_rows": len(test),
              "parameter_count": sum(p.numel() for p in initial.parameters()),
              "records": records, "elapsed_seconds": round(time.monotonic()-started, 3),
              "artifact_sha256": {"activations.pt": digest(out/"activations.pt")},
              "scope": "full-domain present-task learning; diagnostic shear labels used only after encoder freeze; no inferential confidence claim"}
    (out/"results.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE/"manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.out)
