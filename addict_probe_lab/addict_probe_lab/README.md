# Addict Probe Lab

Apple-Silicon-first research harness for storyworld moral-reasoning runs, fixed-shape trace logging, ANE-friendly scorer experiments, and lightweight mechinterp/probing.

This repo is a complementary harness, not a rewrite of runtime/TRM projects.

## What this repo does now

- Runs repeated storyworld episodes through a local backend (`qwen_mlx` first, `qwen_transformers` fallback).
- Logs one JSONL row per step with bounded-history summaries and score proxies.
- Converts traces to fixed-size 128-dim features and windowed training tensors.
- Provides probe baselines (`dense`, `linear`, `temporal`, `dictionary`).
- Provides mechinterp-lite analysis (trajectory summary, kmeans clustering, feature/outcome correlation).
- Provides a small fixed-shape MLP scorer with Core ML export scaffold.

## Repo layout

- `addict_probe/config.py`: app/backend/harness/scorer config dataclasses.
- `addict_probe/harness/*`: scenario loader, episode loop, runner, metrics, JSONL logging.
- `addict_probe/backends/*`: backend interface + MLX/Transformers + CoreML scorer wrapper.
- `addict_probe/trace_schema.py`: fixed 128-d feature schema, JSONL->`[batch,time,feat]`, windowing/splits.
- `addict_probe/train_data.py`: trace->labels for Addict Probe training.
- `addict_probe/probes/*`: linear/temporal/dictionary probes and ablations.
- `addict_probe/mechinterp/*`: activation extraction, summaries, clustering, route feature ranking.
- `addict_probe/evals/*`: storyworld and RIPE-like internal eval scaffolds.
- `configs/scenarios/moral_storyworld_pack.json`: example scenario pack.
- `configs/default.json`: example full config.

## Install

```bash
pip install numpy
```

Optional:

```bash
pip install transformers torch
pip install mlx-lm  # on Apple Silicon only
pip install coremltools  # on macOS for export path
```

## Quick start

### 1) Run storyworld batch

```bash
python scripts/run_storyworld_batch.py --config configs/default.json
```

Override backend / run size:

```bash
python scripts/run_storyworld_batch.py --config configs/default.json --backend qwen_transformers --episodes-per-scenario 2 --max-turns 6 --trace-path out/traces/run_a.jsonl
```

Bounded intermediate reasoning (simulated CoT in-env):

```bash
python scripts/run_storyworld_batch.py --config configs/default.json --trace-path out/traces/run_b.jsonl --enable-think-step --think-max-new-tokens 48 --think-char-cap 240
```

Contract robustness controls (recommended for local Qwen loops):

- `backend.action_retry_count` retries invalid action format with a repair prompt.
- Harness now strips raw `<think>` blocks from outputs before scoring/logging.
- Harness contract now requires `OPTION_INDEX` + `INTENDED_ACTION` + `DIALOGUE` + `MORAL_LOGIC`.
- Each row logs `valid_action`, `effective_valid_action`, `contract_errors`, and `action_retry_count`.

### 2) Evaluate trace

```bash
python scripts/run_storyworld_eval.py --trace out/traces/storyworld_trace.jsonl --out out/evals/storyworld_eval.json
```

### 3) Analyze mechinterp-lite signals

```bash
python scripts/analyze_storyworld_traces.py --trace out/traces/storyworld_trace.jsonl --clusters 6 --out-json out/analysis/trace_report.json --out-clusters-npz out/analysis/trace_clusters.npz
```

### 4) Export Addict training tensors

```bash
python scripts/export_storyworld_training_tensors.py --trace out/traces/storyworld_trace.jsonl --out out/datasets/storyworld_train.npz --window 8 --stride 1
```

### 5) Train probe baseline from traces

```bash
python scripts/train_probe_from_traces.py --trace out/traces/storyworld_trace.jsonl --method temporal --target next_moral_score --window 8
```

Methods:

- `dense`
- `linear`
- `temporal`
- `dictionary`

Targets:

- `next_moral_score`
- `future_consistency_delta`
- `intervention_needed`
- `future_usefulness_delta`

### 6) Export ANE-friendly scorer scaffold

```bash
python scripts/export_coreml_scorer.py --out-npz out/scorer/tiny_mlp_scorer.npz --out-coreml out/scorer/tiny_mlp_scorer.mlpackage
```

If Core ML deps are missing, script still writes NPZ and reports fallback status.

### 7) Compare two runs (A/B)

```bash
python scripts/compare_storyworld_runs.py --run-a out/traces/run_a.jsonl --run-b out/traces/run_b.jsonl --out out/evals/run_ab_compare.json
```

### 8) Train Addict lexica and use it during runs

```bash
python scripts/train_addict_lexica.py --trace out/traces/storyworld_trace.jsonl --out-npz out/lexica/addict_lexica.npz
python scripts/run_storyworld_batch.py --config configs/default.json --trace-path out/traces/with_lexica.jsonl --lexica-npz out/lexica/addict_lexica.npz
```

### 9) Post-train adapter vs base pipeline

```bash
python scripts/run_posttrain_bench_pipeline.py --config configs/default.json --base-model /Users/ben/worker/models/Qwen3.5-2B-Base-MLX-4bit --adapter-path /Users/ben/worker/adapters/storyworld-constitution-2b-r8-qv-exact-s256-YYYYMMDD_HHMMSS --wait-for /Users/ben/worker/adapters/storyworld-constitution-2b-r8-qv-exact-s256-YYYYMMDD_HHMMSS/adapters.safetensors --out-dir out/posttrain
```

This pipeline does:
1. Primer base run.
2. Lexica training from primer trace.
3. Full bench run on base with lexica diagnostics active.
4. Full bench run on adapter (base+adapter path) with lexica diagnostics active.
5. Final report with adapter-minus-base deltas.

### 10) Run the constitutional storyworld autoloop

```bash
python scripts/run_constitutional_autoloop.py --plan constitutional_storyworld_loop/iteration_plan.example.json
```

This loop:
1. runs primer plays
2. retrains Addict lexica
3. benches base vs adapter
4. ranks atoms by future moral-trajectory lift
5. exports replay tensors under `data/adapter_training/replay/repeat/`
6. can launch the next short adapter training command
7. gates adapter promotion using `min_valid_action_rate` (default `0.8`)

It now also exports a replay JSONL dataset for MLX LoRA under:

- `.../data/adapter_training/replay/repeat/jsonl/train.jsonl`
- `.../data/adapter_training/replay/repeat/jsonl/valid.jsonl`
- `.../data/adapter_training/replay/repeat/jsonl/test.jsonl`

Replay export now filters out invalid-action rows and strips `<think>` content from assistant targets.

To inspect the transition-aware lens separately:

```bash
python scripts/extrapolate_moral_lens.py --trace out/posttrain_const2b_r8/bench_adapter_trace.jsonl --lexica-npz out/posttrain_const2b_r8/addict_lexica.npz --out-json out/analysis/transition_lens.json
```

To build the replay JSONL dataset directly:

```bash
python scripts/build_storyworld_replay_dataset.py --trace out/posttrain_const2b_r8/bench_adapter_trace.jsonl --transition-report out/analysis/transition_lens.json --out-dir out/replay_jsonl
```

### 11) Run hybrid Adict + ANE + TinyLora calibration loop

```bash
python scripts/run_hybrid_adict_tinylora_loop.py --trace out/traces/storyworld_trace.jsonl --out-json out/hybrid/hybrid_loop_report.json --steps 40 --batch-size 24 --tap-layers 16,24 --target-features 5,12 --control-features 18,25 --late-layer-floor 20 --adapter-names general_storyworld,moral_conflict,powret_helper
```

Or use a named feature-set config:

```bash
python scripts/run_hybrid_adict_tinylora_loop.py --trace out/traces/storyworld_trace.jsonl --out-json out/hybrid/hybrid_loop_report.json --feature-config configs/hybrid_feature_sets.json --feature-set late_layer_constitution_v1 --steps 40 --batch-size 24 --adapter-names general_storyworld,moral_conflict,powret_helper
```

This loop emits:

- sidecar probe/conflict/capability/SAE salience summaries
- deterministic target/control late-layer feature strengths
- late-layer drift and steering-strength summaries
- routing entropy and adapter collapse metrics
- weighted hybrid loss breakdown for calibration experiments

See `docs/hybrid_adict_ane_tinylora_spec.md` for interfaces and tensor shapes.

## Persistent-selection experiments

The design is in `docs/mechinterp_project_spec.md`. This part of the repo is separate from the storyworld harness and uses real model activations.

- `addict_probe/persistence/*`: choice-task grammar and environments, the reference behavioural model, sensitivity indices, regime regions, shared prefixes, grouped splits with a sealed-test manifest.
- `addict_probe/tracka/*`: the small Track A transformer, probes, and subspace interchange.

Install (Windows, CUDA 12.8 build of torch):

```bash
python -m venv .venv
.venv\Scripts\python -m pip install numpy pytest
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Run from this directory:

```bash
python -m pytest -q
python scripts/run_regime_manipulation_check.py
python scripts/build_prefix_pool.py
python scripts/train_track_a.py --out-dir out/track_a/run2 --data-variant v2
python scripts/probe_track_a.py --run-dir out/track_a/run2
python scripts/probe_track_a_parametrisations.py --run-dir out/track_a/run2
python scripts/intervene_track_a.py --run-dir out/track_a/run2
python scripts/das_track_a.py --run-dir out/track_a/run2 --factor notice
python scripts/components_track_a.py --run-dir out/track_a/run2
```

`components_track_a.py` needs no training and runs on the CPU (about 25 minutes). Checkpoints and generated datasets under `out/` are not committed; rerun the training script to recreate them.

`out/track_a/run1` was trained on data variant v1; the spec (section 14.1) explains why v2 replaced it. The long-running scripts save partial results and resume when rerun with the same arguments.

On a machine that is busy with other jobs, set `OPENBLAS_NUM_THREADS=2` before the probe script; the default thread count stalls under contention. Run the GPU scripts one at a time: two of them sharing a busy GPU ran about ten times slower than one.

## Trace schema

Per-step fixed layout (`total_dim=128`):

- prompt summary: 16
- output summary: 16
- hidden summary: 32
- score summary: 8
- bounded history summary: 16
- backend/control metadata summary: 8
- reserved: 32

## Resource profiles

For MacBook with around 12 GB free unified memory:

- keep `use_batching=false`
- keep `batch_size=1`
- use short `max_new_tokens` (64-128)
- keep `max_turns` conservative
- prefer summary logging over full hidden dumps

For M4 MacBook with 48 GB free:

- run larger episode counts and longer batches
- test MLX backend first for 2B path
- run more scenario families per overnight window

Recommended offload to 3090 node:

- heavier probe sweeps / hyperparam searches
- larger dictionary probe experiments
- broader clustering/ranking sweeps over many trace files
- any QLoRA training loops (outside this repo's minimal baseline scope)

## Notes

- This is an internal research scaffold, not a published benchmark suite.
- `ripe_like_eval.py` is intentionally lightweight and does not claim external validity.
- Core ML/ANE path is designed for exportability and fallback safety; it does not claim private ANE APIs.
