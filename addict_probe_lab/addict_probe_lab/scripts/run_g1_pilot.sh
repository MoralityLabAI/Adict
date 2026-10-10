#!/usr/bin/env bash
# G1 pilot: regime indices for the base model, then train and measure one
# teacher-imitation adapter per regime. Resumable: finished steps are skipped.
# Usage: scripts/run_g1_pilot.sh <checkpoint dir> [regimes...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../../.venv/Scripts/python.exe
MODEL="$1"; shift
REGIMES=("${@:-OCP_A HAB RMAX}")
N_TRAIN=${N_TRAIN:-800}
N_PREFIXES=${N_PREFIXES:-40}
LOG=out/trackb/g1_pilot.log
mkdir -p out/trackb/indices out/trackb/adapters
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

if [ ! -f out/trackb/indices/base.json ]; then
  say "measuring base"
  "$PY" -u scripts/measure_trackb_indices.py --model "$MODEL" --variant base --n-prefixes "$N_PREFIXES" >> "$LOG" 2>> out/trackb/g1_pilot.err.log
fi
for R in ${REGIMES[@]}; do
  A=out/trackb/adapters/T_${R}_s1
  if [ ! -f "$A/adapter_model.safetensors" ]; then
    say "training $R"
    "$PY" -u scripts/train_trackb_adapter.py --model "$MODEL" --regime "$R" --n-train "$N_TRAIN" --out-dir "$A" >> "$LOG" 2>> out/trackb/g1_pilot.err.log
  fi
  if [ ! -f "out/trackb/indices/T_${R}_s1.json" ]; then
    say "measuring $R"
    "$PY" -u scripts/measure_trackb_indices.py --model "$MODEL" --adapter "$A" --variant "T_${R}_s1" --n-prefixes "$N_PREFIXES" >> "$LOG" 2>> out/trackb/g1_pilot.err.log
  fi
done
say "done"
