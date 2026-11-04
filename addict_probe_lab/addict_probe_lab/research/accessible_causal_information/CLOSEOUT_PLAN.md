# Accessible causal information: bounded closeout plan

Status: planning only, 2026-11-03. **At most three Claude passes and one gated Silico GPU job remain.** The sandbox checks below have run. The prospective stages have not. No paid job is authorized by this document.

Owner repository: Adict. Destination: `addict_probe_lab/addict_probe_lab/research/accessible_causal_information/`. Starting branch: `research/blind-fourier-discovery`, inspected at `55e9b5725f187faaad6f4d807307dd27b6ea829f`. Preserve the in-progress branch, all prior reports and their receipts. General quotient mathematics remains in Compactification-Theory/EvalMath.

## 1. Finish line

Close the study with a reproducible account of **when an intervention-informed readout can recover a learned cyclic representation, what information it was given, and whether the recovered subspace contributes to the original model's behavior**. The guaranteed deliverable is a synthetic methods report, including failures. A positive GPU transfer would add one checkpoint-specific causal case study.

There is no requirement to obtain a positive result on a language model. Failure of a gate produces a narrower final report and closes this study. Further architectures, new regularizers, larger models and SAE training belong to a separately approved project.

## 2. What is already established in the repository

The inspected [BF-02 report](BLIND_SUBSPACE.md) removes the supplied complex-coordinate pairing. On its confirmation seed 311 it reports 0.983 held-out shear accuracy, unchanged by its fixed orthogonal rotation, with subspace group-law residual 1.7e-4. It saves checkpoint and activation hashes. Its paired-layout exponent-search baseline reaches 1.000. The same report records 0.11 accuracy for one synthetic higher-exponent case and limits its primary demonstration to small exponents.

[BF-01](BLIND_FOURIER.md) supplies the earlier known-pairing reference. [AC-01](ANTI_COLLAPSE.md) records the collapse and anti-collapse experiments. These encoders receive engineered equivariance supervision. Their encoders saw all input pairs; the held-out pairs concern the diagnostic decoder. A causal intervention on the native readout remains a separate test. Poor fitted affine/bilinear accuracy, by itself, establishes neither an impossibility theorem for the full class nor an advantage over every learner.

Source snapshot: [BF-02 at 55e9b57](https://github.com/MoralityLabAI/Adict/blob/55e9b5725f187faaad6f4d807307dd27b6ea829f/addict_probe_lab/addict_probe_lab/research/accessible_causal_information/BLIND_SUBSPACE.md). Existing [Track A interchange code](../../addict_probe/tracka/causal.py) already replaces a subspace component and continues the native model. Its interfaces can guide the adapter; its availability does not establish compatibility with a different model backend.

## 3. Sandbox checks used to choose the gates

These are newly executed analytic fixtures in `closeout_planning/`, independent of BF-02. No learned checkpoint was evaluated here.

### Phase composition has a conditioning cost

Suppose unit features have phase errors bounded by delta_j and a composed readout uses signed integer exponents c_j. With output-phase calibration error at most delta_0, the unwrapped error is bounded by

    delta_0 + sum_j |c_j| delta_j.

Nearest-root decoding of a p-class target is exact whenever this is strictly below pi/p. This follows by summing phase errors under multiplication and using the half-spacing between adjacent roots. A measured mean shift residual does not supply these uniform phase bounds.

For independent zero-mean Gaussian feature phase noise of standard deviation 0.01 radians, the fixture gives the following accuracies over 15,376 evaluations per row (961 states, 16 noise replicates):

| Signed exponents | Accuracy |
| --- | ---: |
| (2,1) | 1.0000 |
| (7,1) | 0.8437 |
| (15,15) | 0.3626 |

These are descriptive simulated values with known true characters and zero calibration error. The fixture also records the model-based bound `P(error) <= min(1, sigma^2 ||c||_2^2 / (pi/p)^2)`, obtained by Markov's inequality on squared unwrapped error. Other noise models require their own moment bounds. We therefore retain an exponent/noise curve and explicit abstention instead of silently dropping hard characters.

### Perfect probing can coexist with zero native causal use

The second fixture gives two models identical activations `(chi(a), chi(b), chi(a+b))` and identical present outputs. Both permit perfect external phase decoding. One native readout uses `chi(a)chi(b)`; the other uses the bypass `chi(a+b)`. Patching the first two blocks from `(a+1,b)` produces the correct counterfactual in 961/961 cases for the first model and 0/961 for the bypass model. This is an exact construction. It motivates testing the frozen original readout, including complement and sham controls.

Seven fixture tests passed normally and under `python -O`. `sandbox_receipt.json` records the source hashes and executions. These are separate from the repository's existing research tests.

## 4. Remaining execution sequence

| Pass | Owner | Work | Hard resource cap | Deliverable / decision |
| --- | --- | --- | --- | --- |
| C1 | Claude, local or sandbox CPU | Lock BF-02, test robustness and native causal use | 30 CPU-minutes; three new encoder seeds, two conditions; one frozen discovery recipe | One consolidation report; pass G1 or go directly to C3 |
| C2 | Claude, CPU | Build and dry-run the GPU extraction/patching adapter | 20 CPU-minutes; no model training or paid inference | Hashed runnable bundle and short worker card; pass G2 or go to C3 |
| S1 | Silico worker, GPU | One frozen language-model extraction and native-interchange pilot | One checkpoint; at most 2 H100 GPU-hours including loading/retries | Complete positive, negative or inconclusive receipt |
| C3 | Claude, CPU | Reproduce, audit claims, write final report and integrate | 30 CPU-minutes; no new model selection | Final evidence index, passing scoped tests, clean branch handoff |

The full route is C1 -> C2 -> S1 -> C3. The early-stop route is C1 -> C3. No stage can authorize a new research sweep. One correction of an implementation defect is allowed within a stage's cap, with the defect and resulting rerun recorded; an empirical failure is not a reason to change the recipe.

### C1: finish the synthetic study, including native use

Read the BF-02 report and frozen manifest first. Resolve actual checkpoint files and verify their hashes. Missing checkpoints are an artifact failure; label replacement training explicitly. Use the existing encoder recipe and frozen discovery settings. New prospective seeds are 401, 409 and 419, each under task-only and `eq_norm_cov` conditions, with matched initial weights. Use new diagnostic split seed 2026110301. This is a proposed extension beyond the already exposed BF-02 outcomes; commit its resolved manifest before training.

Preserve opaque shift pairs, supplied order 31 and state-ID relabeling tests. Count unlabeled edges separately from labels. Test an orthogonal rotation and the frozen small-exponent targets `(2,1)` and `(1,2)`. Keep a separate analytic high-exponent/noise table. Any failure or abstention remains in the denominator. Do not retune on the new seeds.

For causal use, patch the discovered subspace in the **original encoder code immediately before its existing task head**, then continue that frozen head. For each of 256 frozen donor/recipient pairs, change a while holding b fixed; compare with the native output at the true counterfactual input. Score every pair. Include no-edit, full-code donor replacement, complement replacement, and 16 preseeded rank-matched random-subspace controls. Report patch norms, displacement from the real counterfactual code and unrelated-output damage. Norm-match random edits for the primary comparison and retain their unscaled results as diagnostics. A refitted external probe does not count as the native head.

**G1 numerical gates, fixed prospectively for the new seeds:** present-task accuracy >=0.99; small-exponent decoder accuracy >=0.95; discovered-subspace held-out group residual <=0.01; rotation changes accuracy by <=0.02. Native interchange must reach counterfactual accuracy >=0.80 and exceed the best of the 16 matched random controls by >=0.20 absolute on every new seed. Failure of native interchange closes the paper as representation discovery without a demonstrated native mechanism; it skips S1.

Include the strongest feasible nonlinear/phase-feature comparator under an explicit fit budget. Preserve known-pairing exponent search as a privileged reference. Record architecture, label targets and shift-graph access per method. Sample-efficiency superiority remains unclaimed unless the information and budget comparisons support it. A stronger generic decoder matching accuracy changes the advantage claim; it does not erase coordinate-free discovery.

### C2: make one GPU job executable without a long agent conversation

Use one already supported Adict checkpoint, provisionally `Qwen3.5-4B-Base`, with exact model/tokenizer revisions resolved from the existing Track B configuration. No new adapter training. Freeze a two-template modular-addition fixture, a validation-selected answer-scoring rule, and answer-position semantics. Score native continuations with the original language-model head. Freeze treatment of multi-token class labels before inspecting scored outcomes.

Cap candidate extraction at two intermediate residual layers, nearest 1/2 and 3/4 of the actual layer count; select the final site using fitting/validation data only. The discovery API gets hidden states and opaque shift pairs. It can receive the known group order; the decoder receives its declared task. No operand labels or target frequencies may leak into discovery.

Create clean train/validation/test group IDs from one predeclared prompt-pair split. A change of wording is not an independent encoder seed. Cache all tensor bytes and model hashes. Reuse the interchange design with backend-specific no-op, full-donor and position tests. Measure memory with a mocked tiny-model dry run and calculate forward-pass counts before allocation. No GPU access is needed for C2.

**G2:** input hashes resolve; checkpoint/tokenizer and extraction site are pinned; split/leakage and no-op tests pass; all controls are executable; estimated maximum cost fits the cap. The manifest must contain an explicitly approved RU ceiling obtained from the applicable worker quote. No conversion between RUs and H100-hours is assumed here. If any requirement is missing, deliver a ready/blocked bundle to C3 and end this study without spending Silico resources.

### S1: one GPU pilot, then stop

Give Silico one <=800-token execution card, the staged bundle and a single command. It executes the frozen job and returns a <=300-word result plus artifact locations/hashes. It does not perform literature searches, repository tours, CPU-only toy extensions, hyperparameter selection or an independent rewrite of the protocol.

The first preflight uses 128 pinned pairs per template. Native task accuracy must be >=0.90 on each template before expensive extraction; otherwise stop and preserve a capability failure. The 2-H100-hour ceiling includes preflight, loading, extraction, interventions, failures and at most one infrastructure-only retry with unchanged inputs. An additional ceiling is the approved RU amount; whichever is reached first stops the worker.

Evaluate native subspace interchange on 256 sealed counterfactual pairs per template, with the C1 controls and a site fixed on validation. Primary pilot success requires >=0.80 correct counterfactual outputs and >=0.20 lift over the best matched random-subspace control on **both** templates. Also report clean-task damage and the zero-edit check. These finite-fixture decision rules are operational gates, not a general anytime error certificate. Report exact denominators and every attempted site. Any statistical interval must use the declared independent sampling unit; token-level pseudoreplication is disallowed.

A passing job supports a causal abstraction on this checkpoint, boundary and intervention family. A failed job closes the transfer attempt. No SAE fitting, larger checkpoint, SFT rescue, new prompt family or extended run is permitted in this study.

### C3: publish the bounded claim and end the loop

Recheck sources and saved checkpoint/activation hashes. Re-run the scoped CPU tests normally and with `-O`. Include any missing artifacts, backend limitations and incomplete GPU work in the evidence ledger. A source hash alone is not an accessible checkpoint.

Write `FINAL_REPORT.md` and a machine-readable claim-to-artifact index. Distinguish exact theorem, engineered synthetic representation, decoder-split result, and native-model intervention. Give the decoder its full prior-cost accounting: known group order, shift graph, supplied target, normalization and fixed coefficients. Label exploratory reuse explicitly. Keep all failed seeds and source receipts.

Produce a single merge-ready branch/PR; integrate with main under the owner's existing authorization and a fresh-head check. Never force-reset someone else's concurrent work. If research or infrastructure gates fail, publish the narrower result with a clear terminal status. This plan has then finished.

## 5. Compact handoff cards

**Claude C1:** Read CLOSEOUT_PLAN sections 2-4 and BF-02's frozen manifest. Execute C1 only with seeds 401/409/419 and split 2026110301. Save exact checkpoints before scoring. Test the original model head with subspace interchange and the fixed controls. Keep the frequency/noise failures. Commit results and report G1 pass/fail, hashes and test commands in <=200 words. CPU cap 30 minutes. No Silico or GPU calls.

**Claude C2:** If G1 passed, implement C2's two-layer, one-checkpoint GPU driver using Adict's existing backend. Deliver a staged bundle, resolved hashes, mocks/tests, forward-pass ledger and <=800-token worker card. Freeze site-selection and scoring before the GPU run. Record a missing RU authorization as blocked. CPU cap 20 minutes; do not launch paid work.

**Silico S1:** Execute the staged manifest with the provided command. Verify hashes, honor preflight and both spending caps, run the frozen extraction/native-interchange controls, and save receipts. Return results in <=300 words with artifact paths. No lookups, model changes, prompt search or CPU research. Stop after this job.

**Claude C3:** Audit C1 and any S1 output, reproduce the CPU tests, write FINAL_REPORT.md and the claim/artifact index, and prepare the final integration. Preserve all failed outcomes. No further experiment proposal is part of this pass. CPU cap 30 minutes.

## 6. Reproduce the planning checks

From `addict_probe_lab/addict_probe_lab`:

```sh
OPENBLAS_NUM_THREADS=1 python research/accessible_causal_information/closeout_planning/sandbox_gate_probe.py --out /tmp/closeout-gates.json
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/accessible_causal_information/closeout_planning -p test_gate_probe.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s research/accessible_causal_information/closeout_planning -p test_gate_probe.py -v
```

The planning files are additive. Existing research source files, data and manifests are unchanged by this commit.
