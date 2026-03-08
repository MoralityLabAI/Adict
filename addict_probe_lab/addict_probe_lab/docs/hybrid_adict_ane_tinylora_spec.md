# Hybrid Adict + ANE + TinyLora Spec

This scaffold implements a practical split:

- Path A: base model + TinyLora path (normal model runtime)
- Path B: sidecar probe path (ANE-friendly fixed-shape scoring)

## Modules

- `addict_probe/hybrid/types.py`
  - `LayerSliceBatch`
  - `SidecarScores`
  - `RoutingOutput`
- `addict_probe/hybrid/slicing.py`
  - fixed-shape tap slice builder from trace rows
- `addict_probe/hybrid/sidecar.py`
  - `ANESidecarProbeRunner` for probe/conflict/capability/SAE salience scores
- `addict_probe/hybrid/routing.py`
  - `RoutingMLP` and softmax routing weights
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
- `route_weights`: `[batch, n_adapters]`

## Run

```bash
python scripts/run_hybrid_adict_tinylora_loop.py \
  --trace out/traces/storyworld_trace.jsonl \
  --out-json out/hybrid/hybrid_loop_report.json \
  --steps 40 \
  --batch-size 24 \
  --tap-layers 16,24 \
  --adapter-names general_storyworld,moral_conflict,powret_helper
```

## Current scope

- This is a calibration/research harness scaffold, not full model-weight training.
- It is designed to be ANE-export-friendly in tensor shape and control flow.
- It can supervise routing and sample weighting decisions before deeper integration.

