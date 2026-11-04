# Accessible causal information: final synthetic-study report

**Terminal status: CLOSED — G1 failed; C2 and S1 skipped.** This is the C3 closeout of the bounded plan at [CLOSEOUT_PLAN.md](CLOSEOUT_PLAN.md), following C1 result commit `3faf167a9416457a1d363a978916dbaf9fb45818`. The scope is a small engineered modular-addition encoder and its existing task head. No language-model activation circuit, general causal mechanism, or GPU transfer is established. The source experiments, manifests, failed seeds and checkpoints remain intact.

## Scientific question and definitions

For `(a,b) in Z_31²`, the native task is `a+b mod 31`. An engineer-trained equivariant subspace can encode phase characters under known additive generators. The question is whether an algorithm can discover those characters from a 32-dimensional activation vector and opaque shift pairs, and whether changing the recovered subspace changes the *original* model's output as predicted. One-hot oracle decoding, fitted decoding and native intervention are distinct tests.

## Evidence ledger

| Tier | Supported claim | Primary evidence and limits |
| --- | --- | --- |
| Exact finite-state mathematics | A task quotient can lose behavior under expanded intervention families; a prescribed phase-product decoder can use additional group structure | [Compactification-Theory PR #28](https://github.com/MoralityLabAI/Compactification-Theory/pull/28) and [phase_access.py](phase_access.py). Requires a known group, action and task; mathematical toy result |
| Learned representation, initial pilot | Full 96-bit codes distinguish all 961 inputs even when fitted affine shear decoding is near chance | [UPSTREAM_RESULTS.md](UPSTREAM_RESULTS.md), [original_results.json](original_results.json), [original_receipt.json](original_receipt.json). Training saw all 961 inputs; the 480/481 holdout is a decoder split |
| Anti-collapse / engineered features | Unit-norm/covariance constraints prevent the observed zero-phase collapse and may yield usable characters; canonical coordinate prescription can be wrong | [ANTI_COLLAPSE.md](ANTI_COLLAPSE.md), [anti_collapse_run_receipt.json](anti_collapse_run_receipt.json). Strong prior about the action, phase slots and training loss |
| BF-01, supplied pairing | Blind shift-based character frequency estimator decodes held-out shear when it receives the complex coordinate pairing | [BLIND_FOURIER.md](BLIND_FOURIER.md) and BF-01 frozen manifest/receipt. Known group order and shift graph; retraining caveat |
| BF-02, coordinate-free | From real 32-d activations and opaque generator shift pairs, recovered `(0,1),(1,0)` characters and decoded 98.3% held-out shear on confirmation seed 311; rotation robustness | [BLIND_SUBSPACE.md](BLIND_SUBSPACE.md), [blind_subspace_manifest.json](blind_subspace_manifest.json), [blind_subspace_checkpoints.json](blind_subspace_checkpoints.json). The operator graph and modulus are supplied; the encoders saw all input pairs; small exponents favored |
| C1, preregistered native causal test | **G1 FAILED**. On seeds 401/409/419, discovered-subspace replacement produced 0/256 correct native counterfactual outputs per seed; full donor and complement replacement produced 256/256. All 16 norm-matched random controls per seed also scored zero | [C1_REPORT.md](C1_REPORT.md), [c1_manifest.json](c1_manifest.json), [c1_results.json](c1_results.json), [c1_checkpoints.json](c1_checkpoints.json). This falsifies the declared native-use gate at the original `read_sum` site for this model family |
| Planning fixtures | Exponent/noise sensitivity and bypass counterexample motivate a native gate | [closeout_planning/sandbox_results.json](closeout_planning/sandbox_results.json) and [closeout_planning/sandbox_receipt.json](closeout_planning/sandbox_receipt.json). Analytic toy fixtures, separate from C1 learned checkpoints |

## C1 gate adjudication

**Frozen before C1:** seeds 401/409/419, paired `task_only` and `eq_norm_cov` training, diagnostic split seed 2026110301, frozen BF-02 discovery, 256 recipient/donor pairs and 16 rank-matched random controls. See C1 pre-run commit `0447fd5` and `c1_start.json`.

All `eq_norm_cov` encoders reached 100% native task accuracy. On the two small-exponent targets, the blind decoder achieved 0.761/0.838 for seed 401 and 0.956/0.994 for seed 409; seed 419's discovered characters were linearly dependent and it abstained. The gate required at least 0.95 on both. Residuals were 3.4e-4, 2.2e-4 and 0.0176, respectively; the limit was 0.01. All abstentions count as failures.

**Primary causal test:** at the 32-d code, replace the discovered subspace component with a donor's, then apply the model's *original, frozen* `read_sum` head. The passing threshold was at least 0.80 counterfactual accuracy and at least 0.20 absolute lift over the best random subspace for every seed. The observed scores were 0.000 and 0.000 lift on every seed. No-edit remained at the recipient's output; full-donor-code replacement and complement replacement yielded 1.000. This is strong evidence that the frozen native head uses the complementary coordinates for the declared intervention on these examples. It does not prove a global property of all inputs, layers or models. The report notes the orthogonal-projector/oblique-functionals distinction.

**Comparators:** on target `(2,1)`, the blind readout scored 0.761/0.956/abstain for seeds 401/409/419, while privileged known-pairing exponent search scored 0.952/1.000/0.106. The fixed MLP scored 0.200/0.218/0.087. Comparators have different information privileges; this study makes no unqualified sample-efficiency or generalization superiority claim.

## Reproducibility and provenance

The C1 run reports 195 CPU seconds and **58 scoped Python tests passing normally and under `-O`**, with six saved C1 checkpoints and eleven prior BF-02 checkpoints checked against the run's file/state/activation digests. These are **C1's recorded execution and artifact-lock claims**. The C3 review independently parsed `c1_results.json` and found its gate verdict consistent with the numeric results. The accessible sandbox included the closeout-planning bundle; its **seven fixture tests were independently rerun in Python 3.13.5 normally and under `-O`, all passing**. The full C1 suite and checkpoint-by-checkpoint binary SHA-256 comparison were not independently rerun by C3 because the complete working checkout/weights were unavailable in the local sandbox. Exact GitHub paths and source hashes are indexed in [CLAIM_INDEX.json](CLAIM_INDEX.json). GitHub commit presence proves publication, not scientific validation.

No new encoder search, alternative architecture, intervention scheme, Qwen checkpoint or Silico call was performed during C3.

## What this study establishes

An engineered learned representation can contain an approximately equivariant, discoverable group representation. With the supplied group order and generator shift graph, a bounded, group-aware readout can decode unseen labeled pairs accurately on some seeds, including after coordinate rotation. A new prospective batch shows failures and abstentions, and the native head can remain behaviorally insensitive to the discovered subspace even when an external decoder recovers useful task-related quantities. **Probe accessibility and native causal influence are separate empirical claims.**

## Terminal decision and future scope

The predeclared G1 causal gate failed on all three fresh seeds. The study therefore ends at the synthetic-methods report. **C2 is skipped, S1/Silico is skipped, GPU allocation is zero.** Any follow-on work about causal feature mediation, alternate hidden sites or a real Qwen activation circuit requires a new proposal, a distinct preregistration and explicit compute authorization. This report closes the bounded AC/BF study without retuning the failed experiments.
