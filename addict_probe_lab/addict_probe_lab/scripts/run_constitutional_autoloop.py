#!/usr/bin/env python
"""Autonomous storyworld -> lexica -> bench -> train loop for 3-5 iteration hill climbing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.config import AppConfig
from addict_probe.evals.ripe_like_eval import ripe_like_internal_eval
from addict_probe.evals.storyworld_eval import evaluate_storyworld_trace
from addict_probe.harness.runner import run_storyworld_batch
from addict_probe.mechinterp.transition_lens import write_transition_report
from addict_probe.train_data import save_training_tensors, traces_to_training_tensors
from addict_probe.trace_schema import read_trace_jsonl

from run_posttrain_bench_pipeline import train_lexica_from_trace, wait_for_path
from build_storyworld_replay_dataset import build_replay_rows, write_splits


def _render_template(template: str, values: Dict[str, object]) -> str:
    class SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeDict({k: str(v) for k, v in values.items()}))


def _run_command(command: str, cwd: str | None) -> Dict[str, object]:
    started = time.time()
    proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "cwd": cwd or "",
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "elapsed_sec": round(time.time() - started, 3),
    }


def _run_batch(
    cfg: AppConfig, model_path: str, trace_path: str, lexica_npz: str = "", adapter_path: str = ""
) -> Dict[str, object]:
    cfg2 = AppConfig.from_dict(cfg.to_dict())
    cfg2.backend.model_name_or_path = model_path
    cfg2.backend.adapter_path = adapter_path
    cfg2.trace.trace_path = trace_path
    cfg2.trace.lexica_npz_path = lexica_npz
    summary = run_storyworld_batch(cfg2)
    return {
        "trace_path": summary.trace_path,
        "episodes": summary.episodes,
        "rows": summary.rows,
        "scenarios": summary.scenarios,
    }


def _eval_trace(path: str) -> Dict[str, object]:
    return {
        **evaluate_storyworld_trace(path),
        **ripe_like_internal_eval(path),
    }


def _score_for_hillclimb(metrics: Dict[str, object]) -> float:
    return float(metrics.get("rubric_total", 0.0)) + float(metrics.get("ripe_like_score", 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous constitutional storyworld hill-climb loop")
    parser.add_argument("--plan", required=True, help="Path to autoloop plan JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))

    root = Path(plan["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig.from_json(plan["config_path"])
    cfg.harness.max_turns = int(plan.get("max_turns", cfg.harness.max_turns))
    cfg.harness.episodes_per_scenario = int(plan.get("bench_episodes", cfg.harness.episodes_per_scenario))

    current_adapter = str(plan.get("initial_adapter_path", ""))
    current_model = str(plan.get("base_model_path", cfg.backend.model_name_or_path))
    current_guardrail_report = str(plan.get("initial_guardrail_report_path", ""))
    best_adapter = current_adapter
    best_score = float("-inf")
    receipts: List[Dict[str, object]] = []

    total_iterations = int(plan.get("iterations", 3))
    min_valid_action_rate = float(plan.get("min_valid_action_rate", 0.80))
    command_cwd = plan.get("command_cwd") or str(plan_path.parent)

    for iteration_idx in range(1, total_iterations + 1):
        iter_dir = root / f"iter_{iteration_idx:02d}"
        replay_dir = iter_dir / "data" / "adapter_training" / "replay" / "repeat"
        replay_dir.mkdir(parents=True, exist_ok=True)

        primer_trace = str(iter_dir / "primer_trace.jsonl")
        base_trace = str(iter_dir / "bench_base_trace.jsonl")
        adapter_trace = str(iter_dir / "bench_adapter_trace.jsonl")
        lexica_npz = str(iter_dir / "addict_lexica.npz")
        transition_json = str(iter_dir / "transition_lens.json")
        train_npz = str(replay_dir / "train_windows.npz")
        replay_jsonl_dir = str(replay_dir / "jsonl")
        report_json = str(iter_dir / "iteration_report.json")
        next_adapter_dir = str(iter_dir / "candidate_adapter")
        next_adapter_ready = str(Path(next_adapter_dir) / "adapters.safetensors")

        receipt: Dict[str, object] = {
            "iteration": iteration_idx,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_model_path": current_model,
            "input_adapter_path": current_adapter,
            "artifacts": {
                "primer_trace": primer_trace,
                "base_trace": base_trace,
                "adapter_trace": adapter_trace,
                "lexica_npz": lexica_npz,
                "transition_report": transition_json,
                "train_npz": train_npz,
                "report_json": report_json,
                "next_adapter_dir": next_adapter_dir,
            },
            "commands": [],
        }

        primer_cfg = AppConfig.from_dict(cfg.to_dict())
        primer_cfg.harness.episodes_per_scenario = int(plan.get("primer_episodes", 1))
        primer_cfg.trace.guardrail_report_path = current_guardrail_report
        receipt["primer"] = _run_batch(primer_cfg, current_model, primer_trace, lexica_npz="", adapter_path=current_adapter)
        receipt["lexica"] = train_lexica_from_trace(primer_trace, lexica_npz)

        iter_cfg = AppConfig.from_dict(cfg.to_dict())
        iter_cfg.trace.guardrail_report_path = current_guardrail_report

        receipt["bench_base"] = _run_batch(iter_cfg, current_model, base_trace, lexica_npz=lexica_npz, adapter_path="")
        receipt["bench_adapter"] = _run_batch(
            iter_cfg,
            current_model,
            adapter_trace,
            lexica_npz=lexica_npz,
            adapter_path=current_adapter,
        )

        receipt["base_eval"] = _eval_trace(base_trace)
        receipt["adapter_eval"] = _eval_trace(adapter_trace)
        base_score = _score_for_hillclimb(receipt["base_eval"])
        adapter_score = _score_for_hillclimb(receipt["adapter_eval"])
        adapter_valid_action_rate = float(receipt["adapter_eval"].get("valid_action_rate", 0.0))
        base_valid_action_rate = float(receipt["base_eval"].get("valid_action_rate", 0.0))
        promotion_allowed = adapter_valid_action_rate >= min_valid_action_rate
        receipt["hillclimb"] = {
            "base_score": base_score,
            "adapter_score": adapter_score,
            "delta": adapter_score - base_score,
            "base_valid_action_rate": base_valid_action_rate,
            "adapter_valid_action_rate": adapter_valid_action_rate,
            "min_valid_action_rate": min_valid_action_rate,
            "promotion_allowed": promotion_allowed,
        }

        receipt["transition_lens"] = write_transition_report(
            trace_path=adapter_trace,
            lexica_npz=lexica_npz,
            out_json=transition_json,
            horizon=int(plan.get("transition_horizon", 2)),
            activation_floor=float(plan.get("activation_floor", 0.05)),
            top_atoms=int(plan.get("top_atoms", 12)),
        )
        receipt["guardrail_report_used"] = current_guardrail_report

        x, labels = traces_to_training_tensors(
            adapter_trace,
            window=int(plan.get("window", 8)),
            stride=int(plan.get("stride", 1)),
        )
        save_training_tensors(train_npz, x, labels)
        replay_rows = build_replay_rows(
            adapter_trace,
            transition_json,
            min_rows=int(plan.get("min_replay_rows", 24)),
        )
        replay_manifest = write_splits(replay_rows, replay_jsonl_dir)
        receipt["training_windows"] = {
            "x_shape": list(x.shape),
            "label_sizes": {key: int(val.shape[0]) for key, val in labels.items()},
        }
        receipt["replay_dataset"] = replay_manifest

        manifest = {
            "iteration": iteration_idx,
            "base_model_path": current_model,
            "source_adapter_path": current_adapter,
            "next_adapter_dir": next_adapter_dir,
            "train_npz": train_npz,
            "dataset_dir": replay_jsonl_dir,
            "lexica_npz": lexica_npz,
            "transition_report": transition_json,
            "guardrail_atoms": receipt["transition_lens"].get("guardrail_atoms", []),
            "suppress_atoms": receipt["transition_lens"].get("suppress_atoms", []),
            "bench_rows": len(read_trace_jsonl(adapter_trace)),
        }
        (iter_dir / "next_iteration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        train_template = str(plan.get("train_command_template", "")).strip()
        if train_template:
            rendered = _render_template(
                train_template,
                {
                    "iteration_idx": iteration_idx,
                    "iteration_dir": iter_dir,
                    "base_model_path": current_model,
                    "current_adapter_path": current_adapter,
                    "next_adapter_dir": next_adapter_dir,
                    "next_adapter_ready": next_adapter_ready,
                    "train_npz": train_npz,
                    "dataset_dir": replay_jsonl_dir,
                    "lexica_npz": lexica_npz,
                    "transition_report": transition_json,
                    "config_path": plan["config_path"],
                    "max_turns": cfg.harness.max_turns,
                    "bench_episodes": cfg.harness.episodes_per_scenario,
                },
            )
            cmd_receipt = _run_command(rendered, cwd=command_cwd)
            receipt["commands"].append(cmd_receipt)
            if cmd_receipt["returncode"] != 0:
                receipt["status"] = "train_command_failed"
                Path(report_json).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
                receipts.append(receipt)
                break
            if plan.get("wait_for_next_adapter", True):
                ready = wait_for_path(next_adapter_ready, timeout_min=int(plan.get("wait_timeout_min", 180)))
                receipt["wait_for_next_adapter"] = {"path": next_adapter_ready, "ready": bool(ready)}
                if not ready:
                    receipt["status"] = "adapter_timeout"
                    Path(report_json).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
                    receipts.append(receipt)
                    break
                if promotion_allowed:
                    current_adapter = next_adapter_dir
                else:
                    receipt["promotion_blocked_reason"] = "valid_action_rate_below_threshold"

        if adapter_score > best_score and promotion_allowed:
            best_score = adapter_score
            best_adapter = current_adapter
        current_guardrail_report = transition_json

        receipt["status"] = "completed"
        receipt["best_score_so_far"] = best_score
        receipt["best_adapter_so_far"] = best_adapter
        Path(report_json).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipts.append(receipt)

    final_report = {
        "plan_path": str(plan_path),
        "output_root": str(root),
        "iterations_requested": total_iterations,
        "iterations_completed": len(receipts),
        "best_adapter_path": best_adapter,
        "best_score": best_score,
        "receipts": [str(root / f"iter_{item['iteration']:02d}" / "iteration_report.json") for item in receipts],
    }
    (root / "autoloop_report.json").write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
