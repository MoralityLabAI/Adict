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

To inspect the transition-aware lens separately:

```bash
python scripts/extrapolate_moral_lens.py --trace out/posttrain_const2b_r8/bench_adapter_trace.jsonl --lexica-npz out/posttrain_const2b_r8/addict_lexica.npz --out-json out/analysis/transition_lens.json
```

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
