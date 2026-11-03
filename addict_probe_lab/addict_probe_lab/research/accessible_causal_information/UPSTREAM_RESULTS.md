# Learned intervention transfer: a CPU representation audit

This is an independent follow-on to PR #28 at `6594586c74b38ce6cd410c56e4e087a8742d801c`. The separate Silico `pr28-coverage-gap.patch` was unavailable in this runtime and remains pending integration. Existing `research/dynamical_closure` and `research/ace_mechinterp_gym` sources are unchanged.

## Fixed experiment

For all 961 inputs `(a,b)` in Z_31 squared, train an 11,390-parameter embedding/MLP encoder to predict `s=a+b mod 31`. The hidden code has 32 coordinates. Two seeds, 303 and 307, use identical initial parameters across conditions. The conditions are initialization, sum-only training, and sum-plus-a training. The last supplies explicit hidden-coordinate supervision as a retention control. Each trained model takes 1,500 full-batch AdamW steps on CPU. All input pairs participate in encoder training.

After freezing the encoder, fit affine ridge probes for `s`, `a`, and `2a+b mod 31` on 480 pairs; evaluate them on the other 481 pairs. The shear probe gets labeled shear examples during diagnostic fitting. It measures held-out-pair readout performance, with zero-shot intervention generalization outside its scope. The original model's sum head is evaluated separately on all 961 states.

Use one training-fitted PCA and fixed three-bit coordinate quantization. Taking coordinate prefixes gives nested codes of 96, 48, 24, 12, 6, 3, and zero bits. Their one-hot squared Bayes risks are calculated exactly over all states after the floating-point codes have been constructed. This oracle sees the complete target table and has unrestricted readout complexity. Encoder weights, PCA parameters and decoder weights have separate description costs.

`manifest.json` and the source hashes were written to an unsigned local start record before inspecting results. This records local ordering; independent timing attestation and statistical confidence certification are outside the experiment. No thresholds or settings were tuned after outcomes.

## Results

All four trained models reach 100% present-sum accuracy. The diagnostic affine probe recovers hidden coordinate `a` with 10.60% and 14.14% accuracy after sum-only training, compared with 82.74% and 85.45% at initialization. The supervised retention control reaches 88.98% and 93.14%. These are two-seed descriptive results using one diagnostic split.

At 96 bits every condition retains 961 distinct codes, with exactly zero oracle shear risk. The affine shear probe's accuracy is between 0.21% and 1.46% across conditions. Poor fitted readout performance therefore coexists with an injective finite-state code. At 12 bits, the sum-only representations have present risk 0.1864 and 0.2869, with shear risk 0.6238 and 0.6883. This particular coarse quantizer merges behaviorally distinct states.

The positive-control result also depends on which target is scored: adding `a` supervision improves the fitted `a` probe while redistributing the code geometry. No universal improvement from auxiliary supervision is established.

`summary.csv` records readout accuracies. `quantized.csv` records attained code counts and exact risk fractions for every budget. The full local evidence bundle includes `results.json`, `start.json`, frozen activations and command logs. `receipt.json` records hashes for them.

## Mathematical check

Given code C and a categorical future Y, draw X uniformly and X' independently from the conditional uniform distribution on C(X)'s fiber. With `p_y(z)=P(Y=y | C=z)`, pair disagreement is `E[1-sum_y p_y(C)^2]`. This equals the minimum squared Euclidean error when predicting the one-hot Y from C. Both samples use the same intervention. The supplied test exhausts a small system; it also verifies the reported sum-code shear risk 30/31 and equal four-action-mixture risk 15/62. This independent implementation does not reproduce Silico's missing patch.

## Replay

Requires PyTorch. Recorded environment: Python 3.13.5, torch 2.10.0+cpu. From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/learned_intervention_transfer -q
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s research/learned_intervention_transfer -q
PYTHONDONTWRITEBYTECODE=1 python -m research.learned_intervention_transfer.pilot --out /tmp/learned-transfer
```

Ten new tests pass in both modes. The existing closure suite passes 26 tests and the gym passes 12 in both modes. An independent repeat of the pilot reproduces all six result records, including every exact fraction. All five historical closure source hashes remain intact. The full repository suite, transformer training and proof-assistant compilation were outside this run.
