#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AICOO_HARNESS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PLAN="${ROOT}/mac_autoloop_hybrid_prod.json"
LOG_DIR="${AICOO_LOG_DIR:-${HOME}/worker/logs}"
LOG_FILE="${LOG_DIR}/hybrid_autoloop_$(date +%Y%m%d-%H%M%S).log"
PYTHON_BIN="${AICOO_PYTHON_BIN:-python3}"
BASE_MODEL_PATH="${AICOO_BASE_MODEL_PATH:-${HOME}/worker/models/Qwen3.5-2B-Base-MLX-4bit}"
INITIAL_ADAPTER_PATH="${AICOO_INITIAL_ADAPTER_PATH:-${ROOT}/out/constitutional_autoloop_live_20260307_2ndpass/iter_03/candidate_adapter}"
LORA_CONFIG_PATH="${AICOO_LORA_CONFIG_PATH:-${HOME}/worker/configs/storyworld_const_r8_qv_s256.yaml}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

"${PYTHON_BIN}" - <<'PY'
import json, os
root=os.environ.get("AICOO_HARNESS_ROOT", os.getcwd())
base_model_path=os.environ.get("AICOO_BASE_MODEL_PATH", os.path.expanduser("~/worker/models/Qwen3.5-2B-Base-MLX-4bit"))
initial_adapter_path=os.environ.get("AICOO_INITIAL_ADAPTER_PATH", os.path.join(root,"out/constitutional_autoloop_live_20260307_2ndpass/iter_03/candidate_adapter"))
python_bin=os.environ.get("AICOO_PYTHON_BIN","python3")
lora_config=os.environ.get("AICOO_LORA_CONFIG_PATH", os.path.expanduser("~/worker/configs/storyworld_const_r8_qv_s256.yaml"))
plan={
  "config_path":"configs/default.json",
  "output_root":"out/constitutional_autoloop_hybrid_prod",
  "base_model_path":base_model_path,
  "initial_adapter_path":initial_adapter_path,
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
  "command_cwd":root,
  "train_command_template":f"{python_bin} -m mlx_lm lora --model {{base_model_path}} --train --data {{dataset_dir}} --adapter-path {{next_adapter_dir}} --resume-adapter-file {{current_adapter_path}}/adapters.safetensors --batch-size 1 --grad-accumulation-steps 1 --iters 120 --learning-rate 5e-5 --num-layers 8 --max-seq-length 256 --grad-checkpoint --save-every 50 --config {lora_config}"
}
with open(os.path.join(root,"mac_autoloop_hybrid_prod.json"),"w") as f:
    json.dump(plan,f,indent=2)
print("wrote", os.path.join(root,"mac_autoloop_hybrid_prod.json"))
PY

echo "Starting autoloop with plan ${PLAN}"
"${PYTHON_BIN}" scripts/run_constitutional_autoloop.py --plan "${PLAN}" | tee "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
