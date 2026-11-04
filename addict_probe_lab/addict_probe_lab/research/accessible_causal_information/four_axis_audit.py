"""Four-axis audit for Adict's completed accessible-causal-information study.

Standard library only. Replays *recorded* numerical results from the committed
BF-02 / C1 / original pilot JSON files; does not rerun the neural experiments.
Also provides an exact finite-code Bayes-risk oracle useful for new fixtures.

Run from repo root:
  python -m research.accessible_causal_information.four_axis_audit --out /tmp/aci-audit.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Hashable, Iterable

ROOT = Path(__file__).resolve().parent
SCHEMA = "adict-four-axis-audit/v1"


class EvidenceError(ValueError):
    """Malformed, contradictory, or incomplete experiment evidence."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"unreadable JSON: {path}") from exc
    if not isinstance(data, dict):
        raise EvidenceError(f"expected a JSON object: {path}")
    return data


def _bool(x: object) -> bool:
    return type(x) is bool


def _rate(x: object, field: str) -> float:
    if type(x) not in (int, float) or x != x or not (0 <= x <= 1):
        raise EvidenceError(f"invalid rate at {field}: {x!r}")
    return float(x)


def _code_key(code: Any) -> Hashable:
    """Encode finite states losslessly for exact oracle-risk computation."""
    if isinstance(code, (str, int, float)) and not isinstance(code, bool):
        return (type(code).__name__, code)
    if isinstance(code, (list, tuple)):
        return ("tuple", tuple(_code_key(x) for x in code))
    raise EvidenceError(f"unsupported finite code {code!r}")


def finite_code_risk(codes: Iterable[Any], labels: Iterable[Any]) -> Fraction:
    """Exact one-hot squared Bayes risk, uniform over the given enumerated states.

    This is E[1 - sum_y P(Y=y | C)^2]; equal to disagreement for a
    uniformly chosen state and a second uniformly chosen member of its fiber.
    Each input state must occur once (or repeated states are intentionally
    weighted by their multiplicity). Returns Fraction, never a float estimate.
    """
    codes, labels = list(codes), list(labels)
    if not codes or len(codes) != len(labels):
        raise EvidenceError("codes and labels require equal, nonzero lengths")
    fibers: dict[Hashable, Counter[Hashable]] = defaultdict(Counter)
    for code, label in zip(codes, labels):
        fibers[_code_key(code)][_code_key(label)] += 1
    n = len(codes)
    return sum((Fraction(sum(c.values()), n) -
                Fraction(sum(v * v for v in c.values()), n * sum(c.values()))
                for c in fibers.values()), Fraction())


def _seed_results(c1: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute G1 decisions from saved aggregate rates and frozen thresholds."""
    if c1.get("schema") != "adict-closeout-c1/v0.1":
        raise EvidenceError("unrecognized C1 result schema")
    if c1.get("manifest") != manifest:
        raise EvidenceError("C1 results embed a manifest differing from c1_manifest.json")
    expected_seeds = manifest.get("seeds")
    if not isinstance(expected_seeds, list) or len(set(expected_seeds)) != len(expected_seeds):
        raise EvidenceError("invalid C1 seeds")
    raw = c1.get("results")
    if not isinstance(raw, list):
        raise EvidenceError("C1 results absent")
    keys = [(r.get("seed"), r.get("condition")) for r in raw]
    expected_keys = {(seed, condition) for seed in expected_seeds for condition in manifest["conditions"]}
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise EvidenceError("C1 seed/condition set differs from frozen plan")
    gates = manifest["g1_gates"]
    primary_target = manifest["primary_target"]
    out = []
    for seed in expected_seeds:
        record = next(r for r in raw if r["seed"] == seed and r["condition"] == "eq_norm_cov")
        blind = record["blind"]
        native = record["native_interchange"]
        n = manifest["interchange"]["pairs"]
        if native.get("pairs") != n or len(native.get("random_norm_matched", [])) != manifest["interchange"]["random_controls"]:
            raise EvidenceError("wrong C1 donor-pair count or matched random controls")
        if native.get("rank", 0) > 32 or native.get("rank", -1) < 0:
            raise EvidenceError("invalid discovered subspace rank")
        primary = blind["targets"][primary_target]
        secondary_name = next(k for k in manifest["targets"] if k != primary_target)
        secondary = blind["targets"][secondary_name]
        def decoder_rate(t: dict[str, Any], name: str) -> float:
            if not isinstance(t.get("abstained"), bool):
                raise EvidenceError("missing explicit abstention state")
            if t["abstained"]:
                if t.get("accuracy") is not None:
                    raise EvidenceError("abstention with accuracy")
                return 0.0
            return _rate(t.get("accuracy"), name)
        primary_acc = decoder_rate(primary, "primary decoder")
        secondary_acc = decoder_rate(secondary, "secondary decoder")
        patched_acc = _rate(native["discovered"]["accuracy"], "native discovered")
        best_random = max((_rate(x["accuracy"], "random control") for x in native["random_norm_matched"]), default=0)
        full_acc = _rate(native["full_code_donor"]["accuracy"], "full donor")
        complement_acc = _rate(native["complement_replacement"]["accuracy"], "complement")
        no_edit_acc = _rate(native["no_edit"]["accuracy"], "no-edit")
        present_acc = _rate(record["present_task_accuracy"], "native present task")
        residual = blind.get("heldout_group_residual")
        if residual is not None and (type(residual) not in (int, float) or residual < 0):
            raise EvidenceError("invalid group-law residual")
        rotation = record.get("rotation_accuracy_change")
        if not isinstance(rotation, dict) or set(rotation) != set(manifest["targets"]):
            raise EvidenceError("invalid rotation change target coverage")
        for target, change in rotation.items():
            if change is not None and (type(change) not in (int, float) or change < 0):
                raise EvidenceError(f"invalid rotation change for {target}")
        # Required gate values are rederived; the saved verdict is a consistency check.
        conditions = {
            "present_task": present_acc >= gates["present_task_accuracy_min"],
            "small_exponent_decoder": (not primary["abstained"] and not secondary["abstained"] and
                                       min(primary_acc, secondary_acc) >= gates["small_exponent_decoder_accuracy_min"]),
            "heldout_group_residual": residual is not None and residual <= gates["heldout_group_residual_max"],
            "rotation_change": all(change is not None and change <= gates["rotation_accuracy_change_max"]
                                   for change in rotation.values()),
            "interchange_accuracy": patched_acc >= gates["interchange_accuracy_min"],
            "interchange_lift": patched_acc - best_random >= gates["interchange_lift_over_best_random_min"],
        }
        if (native.get("best_random_norm_matched_accuracy") != best_random or
                abs(native.get("lift_over_best_random", -999) - (patched_acc - best_random)) > 1e-12):
            raise EvidenceError("native control/lift contradicts recorded per-control scores")
        for name, value in (("patched",patched_acc),("full",full_acc),("complement",complement_acc), ("no_edit",no_edit_acc)):
            if abs(value*n - round(value*n)) > 1e-8:
                raise EvidenceError(f"{name} rate cannot represent an integer count out of {n}")
        out.append({
            "seed": seed,
            "storage": {"claim": "not remeasured from this C1 aggregate; consult original oracle pilot",
                        "native_task_accuracy": present_acc},
            "accessibility": {"primary_target": primary_target, "heldout_primary": primary_acc,
                              "heldout_secondary": secondary_acc,
                              "primary_abstained": primary["abstained"],
                              "secondary_abstained": secondary["abstained"]},
            "structure": {"frequencies": blind.get("frequencies", []),
                          "heldout_group_law_residual": residual,
                          "rotation_accuracy_change": rotation,
                          "independent_features": not primary["abstained"]},
            "native_causal_use": {"tested_pairs": n, "discovered": patched_acc,
                                  "discovered_successes": round(patched_acc*n),
                                  "best_random_norm_matched": best_random,
                                  "complement": complement_acc, "full_donor": full_acc,
                                  "no_edit": no_edit_acc, "lift_over_best_random": patched_acc - best_random},
            "gates": conditions,
            "passed_all": all(conditions.values()),
        })
    verdict = c1.get("verdict", {})
    if not _bool(verdict.get("G1_pass")) or verdict["G1_pass"] != all(s["passed_all"] for s in out):
        raise EvidenceError("saved global C1 verdict contradicts computed gates")
    if not _bool(verdict.get("native_gate_pass")) or verdict["native_gate_pass"] != all(
        s["gates"]["interchange_accuracy"] and s["gates"]["interchange_lift"] for s in out
    ):
        raise EvidenceError("saved native gate contradicts computed gates")
    for s in out:
        matching = next((v for v in verdict.get("per_seed", []) if v.get("seed") == s["seed"]), None)
        if matching is None or any(matching.get(k) != passed for k, passed in s["gates"].items()):
            raise EvidenceError("saved per-seed gate verdict contradicts the measurements")
    return out


def audit(root: Path = ROOT) -> dict[str, Any]:
    paths = {k: root / k for k in
             ("c1_results.json", "c1_manifest.json", "original_results.json", "blind_subspace_results.json")}
    data = {k: _read_json(path) for k, path in paths.items()}
    c1, manifest = data["c1_results.json"], data["c1_manifest.json"]
    seeds = _seed_results(c1, manifest)
    original = data["original_results.json"]
    bf02 = data["blind_subspace_results.json"]
    if original.get("schema") != "learned-intervention-transfer/v0.1" or original.get("n_states") != 961:
        raise EvidenceError("unexpected original finite-state pilot")
    if bf02.get("schema") != "adict-blind-subspace/v0.1":
        raise EvidenceError("unexpected BF-02 source schema")
    original_records = original.get("records", [])
    zero_oracle = sum(1 for r in original_records if any(
        q.get("state_code_bits") == 96 and q.get("attained_codes") == 961 and
        q.get("risk", {}).get("shear") == "0" for q in r.get("quantized_oracle", [])))
    if len(original_records) != 6 or zero_oracle != 6:
        raise EvidenceError("unexpected 96-bit oracle-count evidence")
    confirmation = [r for r in bf02.get("results", []) if r.get("role") == "confirmation" and
                    r.get("condition") == "eq_norm_cov"]
    if len(confirmation) != 1:
        raise EvidenceError("confirmation representation ambiguous")
    c = confirmation[0]
    blind_acc = _rate(c["blind"]["accuracy"], "BF-02 accuracy")
    group_residual = c["blind"]["group_law_residual"]
    if (abs(blind_acc - bf02["verdict"]["blind_accuracy"]) > 1e-12 or
            abs(group_residual - bf02["verdict"]["group_law_residual"]) > 1e-12):
        raise EvidenceError("BF-02 verdict differs from confirmation data")
    if c["seed"] != bf02["manifest"]["confirmation_seed"]:
        raise EvidenceError("BF-02 confirmation seed mismatch")
    return {
        "schema": SCHEMA, "status": "closed_g1_failed" if not all(r["passed_all"] for r in seeds) else "g1_passed",
        "scope": "synthetic modular addition / decoder holdouts / native pre-read_sum patch",
        "axes": {
            "storage": {"type": "recorded finite-code oracle", "oracle_records_with_zero_shear_risk_at_96_bits":zero_oracle,
                        "count":len(original_records), "state_count":original["n_states"],
                        "qualification": "evidence is from original exhaustive oracle results, not recomputed from tensor codes"},
            "accessibility": {"type":"recorded held-out decoder test", "bf02_confirmation_seed":c["seed"],
                              "bf02_accuracy":blind_acc, "c1_seeds": [r["accessibility"] for r in seeds]},
            "structure": {"type":"recorded shift-pair fit", "bf02_confirmation_residual":group_residual,
                          "bf02_confirmation_modes":c["blind"]["frequencies"]},
            "native_causal_use": {"type":"recorded intervention aggregation / computed gate", "c1_seeds": [
                {"seed":r["seed"],**r["native_causal_use"]} for r in seeds]},
        },
        "c1": {"seeds":seeds, "all_pass":all(r["passed_all"] for r in seeds),
                "native_all_pass":all(r["gates"]["interchange_accuracy"] and r["gates"]["interchange_lift"] for r in seeds),
                "gate_source":"c1_manifest.json", "recomputation":"from stored aggregate rates only"},
        "provenance": {"sha256": {k:sha256_file(p) for k,p in paths.items()},
                       "independent_binary_checkpoint_verification": False,
                       "has_raw_intervention_rows": False,
                       "c1_recorded_artifact_lock":c1.get("bf02_artifact_lock",{}).get("all_checkpoints_ok"),
                       "note":"This tool validates recorded data and protocol consistency; it does not rerun torch, checkpoint hashes or native interventions"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="research/accessible_causal_information folder")
    parser.add_argument("--out", type=Path, default=None, help="write JSON report; defaults to stdout")
    parser.add_argument("--require-pass", action="store_true", help="exit code 2 when causal gate failed")
    args = parser.parse_args()
    report = audit(args.root)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.require_pass and not report["c1"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
