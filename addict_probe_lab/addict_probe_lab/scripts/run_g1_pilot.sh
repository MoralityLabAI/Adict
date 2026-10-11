#!/usr/bin/env bash
# G1 pilot: regime indices for the base model, then train and measure one
# teacher-imitation adapter per regime. Resumable: finished steps are skipped.
# Usage: scripts/run_g1_pilot.sh <checkpoint dir> [regimes...]
# Every GPU step waits for a VRAM lease from vramorama, so it queues behind other jobs on the
# shared card instead of pushing it into spill (see vramorama's README).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../../.venv/Scripts/python.exe
MODEL="$1"; shift
REGIMES=("${@:-OCP_A HAB RMAX}")
N_TRAIN=${N_TRAIN:-800}
N_PREFIXES=${N_PREFIXES:-40}
SUFFIX=${SUFFIX:-}  # appended to adapter names, e.g. _vq
LOG=out/trackb/g1_pilot.log
mkdir -p out/trackb/indices out/trackb/adapters
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
VRAMORAMA=${VRAMORAMA:-/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe}
# gpu <lease size> <owner suffix> <python args...>: run one step under a lease (8.4 GB of weights
# plus activations; training adds gradients and optimizer state for the adapter).
gpu() { local size="$1" who="$2"; shift 2; "$VRAMORAMA" run --vram "$size" --owner "adict-$who" -- "$PY" -u "$@"; }

if [ ! -f out/trackb/indices/base.json ]; then
  say "measuring base"
  gpu 10G measure-base scripts/measure_trackb_indices.py --model "$MODEL" --variant base --n-prefixes "$N_PREFIXES" >> "$LOG" 2>> out/trackb/g1_pilot.err.log
fi
for R in ${REGIMES[@]}; do
  V=T_${R}_s1${SUFFIX}
  A=out/trackb/adapters/$V
  if [ ! -f "$A/adapter_model.safetensors" ]; then
    say "training $R"
    gpu 11G "train-$V" scripts/train_trackb_adapter.py --model "$MODEL" --regime "$R" --n-train "$N_TRAIN" --out-dir "$A" >> "$LOG" 2>> out/trackb/g1_pilot.err.log
  fi
  if [ ! -f "out/trackb/indices/$V.json" ]; then
    say "measuring $R"
    gpu 10G "measure-$V" scripts/measure_trackb_indices.py --model "$MODEL" --adapter "$A" --variant "$V" --n-prefixes "$N_PREFIXES" >> "$LOG" 2>> out/trackb/g1_pilot.err.log
  fi
done
say "done"
