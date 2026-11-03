# Accessible causal information: Adict readout study

**Status:** CPU synthetic replication of the upstream learned-representation pilot, plus an independent analytic phase-product fixture. The later Silico decoder-access and sample-efficiency patches remain unpublished locally to Silico and **are not included** here.

**Ownership:** Adict owns probing, access under decoder constraints, learned feature alignment, and eventual activation interventions. [Compactification-Theory PR #28](https://github.com/MoralityLabAI/Compactification-Theory/pull/28) owns the task-quotient proofs and source experiment. [EvalMath](https://github.com/MoralityLabAI/EvalMath) owns general causal-abstraction mathematics.

## Reproduce

From `addict_probe_lab/addict_probe_lab`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/accessible_causal_information -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s research/accessible_causal_information -q
python -m research.accessible_causal_information.pilot --out /tmp/adict-access
```

The pilot uses PyTorch CPU, one thread, two frozen seeds, 1,500 steps per trained condition. It sees every `(a,b)` pair during encoder training. The diagnostic affine probes fit on 480 input pairs and test on the other 481. Every result is about decoder-split transfer, **not zero-shot intervention generalization**. Training produces new `activations.pt`, `results.json`, and a local pre-run source-hash start record under `--out`; generated activation tensors remain outside version control.

`summary.csv`, `quantized.csv`, `original_results.json` and `original_receipt.json` are a snapshot of the verified Compactification-Theory study at commit [`3858a753`](https://github.com/MoralityLabAI/Compactification-Theory/commit/3858a7537151516fff00050849fe45bf05efe6b4). The historical receipt belongs to the original source path; it is retained as provenance and is **not** a fresh verification certificate for this relocated code. `UPSTREAM_RESULTS.md` reproduces the source's report.

## Mathematical separation under the declared toy representation

For uniform independent `s,a` in the cyclic group of prime order 31, the four real coordinates `(cos(2πs/p), sin(2πs/p), cos(2πa/p), sin(2πa/p))` encode every state injectively. An affine decoder of one-hot `s+a` has minimum squared Euclidean error `30/31` for the uniform population because the conditional class averages given either coordinate are uniform. For a strict-margin affine multiclass classifier, take any score difference between two labels: it separates as `f(s)+g(a)`, whose sum is invariant across every diagonal `s+a=t`. Hence it cannot win on one entire diagonal and lose on a different entire diagonal. A degree-two complex phase product reads `s+a` exactly.

When the orientation of the phase axes and group law are known, a single labeled pair determines a constant output offset. The reference `phase_access.py` tests perfect recovery on the fixed untouched 481-pair split. This is a **structural prior** with a real complexity cost, and makes no claim that the trained encoders use Fourier coordinates. The reference stores 31 complex roots (62 real scalar coefficients), learns one 5-bit offset, and evaluates 31 phase similarities per output. A no-sharing lookup for an unseen key has no fitted entry.

## Next controlled mechanistic experiment

With frozen learned encoder outputs, fit a **train-only** alignment map into the two complex phases and score held-out pair generalization with the same phase-product rule. Include random-orthogonal rotations of the analytic phase code as a positive control, a train-only affine classifier, a matched bilinear head, and a held-out intervention varying the hidden coordinate. Count *all* fitted mapping weights and fixed phase arithmetic; freeze the test split before choosing hyperparameters.

For model-based interpretability, follow Adict's existing `docs/mechinterp_project_spec.md` distinction: probe decodability locates candidate features; causal intervention establishes a mechanism. A small synthetic alignment result alone is not evidence of an actual LLM circuit.
