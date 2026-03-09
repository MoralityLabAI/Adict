# Hybrid Adict + ANE + TinyLora Spec

This scaffold implements a practical split:

- Path A: base model + TinyLora path (normal model runtime)
- Path B: sidecar probe path (ANE-friendly fixed-shape scoring)

The current revision follows the "late-layer feature reweighting" paper claim more closely:

- fixed-base dictionary projections are deterministic per tapped layer
- late layers receive higher weight than middle layers
- target and control feature sets are tracked explicitly
- routing consumes target/control gap and late-layer drift, not just generic probe means
- auxiliary losses reward target-over-control separation with bounded drift

## Modules

- `addict_probe/hybrid/types.py`
  - `LayerSliceBatch`
  - `SidecarScores`
  - `RoutingOutput`
- `addict_probe/hybrid/slicing.py`
  - fixed-shape tap slice builder from trace rows
- `addict_probe/hybrid/sidecar.py`
  - `ANESidecarProbeRunner` for fixed-base feature projections, target/control tracking, drift summaries
- `addict_probe/hybrid/routing.py`
  - `RoutingMLP` with steering-aware routing weights
- `addict_probe/hybrid/adapter_bank.py`
  - weighted TinyLora composition
- `addict_probe/hybrid/losses.py`
  - task + constitutional + routing + SAE + balance losses
- `addict_probe/hybrid/loop.py`
  - micro-batch calibration loop with per-step receipts
- `scripts/run_hybrid_adict_tinylora_loop.py`
  - runnable entrypoint

## Data structures

- `layer_slices`: `Dict[str, np.ndarray]`, each `[batch, time, dim]`
- `adict_scores`:
  - `probe_scores`: `[batch, probe_dim]`
  - `conflict_score`: `[batch, 1]`
  - `capability_scores`: `[batch, capability_dim]`
  - `sae_salience`: `[batch, sae_dim]`
  - `target_feature_strength`: `[batch, 1]`
  - `control_feature_strength`: `[batch, 1]`
  - `target_control_gap`: `[batch, 1]`
  - `late_layer_drift`: `[batch, 1]`
- `route_weights`: `[batch, n_adapters]`

## Run

```bash
python scripts/run_hybrid_adict_tinylora_loop.py \
  --trace out/traces/storyworld_trace.jsonl \
  --out-json out/hybrid/hybrid_loop_report.json \
  --steps 40 \
  --batch-size 24 \
  --tap-layers 16,24 \
  --target-features 5,12 \
  --control-features 18,25 \
  --late-layer-floor 20 \
  --adapter-names general_storyworld,moral_conflict,powret_helper
```

## Current scope

- This is a calibration/research harness scaffold, not full model-weight training.
- It is designed to be ANE-export-friendly in tensor shape and control flow.
- It can supervise routing and sample weighting decisions before deeper integration.
- It now exposes fixed-base late-layer metrics that are suitable for later feature steering and control-set benchmarking.
