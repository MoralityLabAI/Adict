#!/bin/bash
set -euo pipefail

ROOT="/Users/ben/worker/addict_probe_harness"
PLAN="${ROOT}/mac_autoloop_hybrid_prod.json"
LOG_DIR="/Users/ben/worker/logs"
LOG_FILE="${LOG_DIR}/hybrid_autoloop_$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

/Users/ben/mlxenv/bin/python - <<'PY'
import json, os
root="/Users/ben/worker/addict_probe_harness"
plan={
  "config_path":"configs/default.json",
  "output_root":"out/constitutional_autoloop_hybrid_prod",
  "base_model_path":"/Users/ben/worker/models/Qwen3.5-2B-Base-MLX-4bit",
  "initial_adapter_path":"/Users/ben/worker/addict_probe_harness/out/constitutional_autoloop_live_20260307_2ndpass/iter_03/candidate_adapter",
  "iterations":3,
  "primer_episodes":1,
  "bench_episodes":3,
  "max_turns":3,
  "window":8,
  "stride":1,
  "transition_horizon":2,
  "activation_floor":0.05,
  "top_atoms":12,
  "hybrid_sidecar_enabled":True,
  "hybrid_steps":24,
  "hybrid_batch_size":24,
  "hybrid_tap_layers":[16,24],
  "hybrid_adapter_names":["general_storyworld","moral_conflict","powret_helper"],
  "min_valid_action_rate":0.8,
  "wait_for_next_adapter":True,
  "wait_timeout_min":240,
  "command_cwd":"/Users/ben/worker/addict_probe_harness",
  "train_command_template":"/Users/ben/mlxenv/bin/python -m mlx_lm lora --model {base_model_path} --train --data {dataset_dir} --adapter-path {next_adapter_dir} --resume-adapter-file {current_adapter_path}/adapters.safetensors --batch-size 1 --grad-accumulation-steps 1 --iters 120 --learning-rate 5e-5 --num-layers 8 --max-seq-length 256 --grad-checkpoint --save-every 50 --config /Users/ben/worker/configs/storyworld_const_r8_qv_s256.yaml"
}
with open(os.path.join(root,"mac_autoloop_hybrid_prod.json"),"w") as f:
    json.dump(plan,f,indent=2)
print("wrote", os.path.join(root,"mac_autoloop_hybrid_prod.json"))
PY

echo "Starting autoloop with plan ${PLAN}"
/Users/ben/mlxenv/bin/python scripts/run_constitutional_autoloop.py --plan "${PLAN}" | tee "${LOG_FILE}"
echo "Log: ${LOG_FILE}"

