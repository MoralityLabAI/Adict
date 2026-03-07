# Constitutional Storyworld Autoloop

This subfolder packages the practical loop for constitution-style storyworld refinement around the existing `addict_probe_lab` harness.

Loop shape:

1. Run primer storyworld episodes.
2. Train Addict lexica from the primer trace.
3. Run base vs adapter benchmark episodes with lexica firing at each step.
4. Extrapolate transition-aware lens atoms from the moral reasoning traces.
5. Export replay tensors under `data/adapter_training/replay/repeat/`.
6. Launch the next short-context LoRA iteration.
7. Repeat for 3-5 iterations without agent intervention.

The key difference from the earlier one-shot bench is the transition lens:

- it does not just ask which atoms predict the current moral score,
- it ranks atoms by whether they improve or degrade future moral consistency and usefulness over the next few steps.

That is the right signal for hill climbing on multi-turn storyworld reasoning instead of optimizing for preachy single-step behavior.

## Primary scripts

- `scripts/run_constitutional_autoloop.py`
- `scripts/extrapolate_moral_lens.py`

## Example

```powershell
cd C:\projects\Adict\Adict\addict_probe_lab\addict_probe_lab
python scripts\run_constitutional_autoloop.py --plan constitutional_storyworld_loop\iteration_plan.example.json
```

## Mac-oriented training shape

The plan file is set up for short, safe 2B adapter refinement:

- 4-bit base
- `batch_size=1`
- `grad_accum=1-2`
- `max_seq_length=256`
- `gradient_checkpointing=true`
- `lora_rank=8`
- `lora_alpha=16`
- target modules `q_proj,v_proj`
- adapt last `8-12` layers only

Replace the training command template with the concrete Mac queue command you want to run.
