# Accessible Causal Information: When Decodable Features Do Not Mediate Native Decisions

**Adict methods paper · November 2026 · Version 1.0 (completed synthetic study)**

## Abstract

A feature discovered in neural activations can retain information about a task, admit an accurate external readout, and obey an identifiable transformation law while contributing little to the model's existing prediction. We formalize four separately testable properties: **storage**, **bounded-decoder accessibility**, **transformation structure**, and **native causal use**. For a 31-class modular-addition system, we show how an injective two-character code can have zero oracle loss while an affine squared-error readout incurs loss \(30/31\). An engineered 32-dimensional encoder learns approximately equivariant phase features. On the confirmation seed of the coordinate-free BF-02 study, shift-pair discovery recovers two cyclic characters and achieves **98.3% held-out shear decoding accuracy**, with residual \(1.74\times 10^{-4}\). We then report the predeclared C1 native-interchange test on three fresh seeds: substituting the discovered subspace changes the original task head to the intended counterfactual in **0/256 pairs per seed**. Full-code donor and complementary-subspace substitutions each produce 256/256 correct counterfactuals per seed. A released standard-library audit replays the frozen gates from recorded numerical evidence and preserves provenance limits. The case illustrates the importance of measuring native mediation after an externally useful feature has been found. The evidence concerns a deliberately engineered synthetic model and makes no claim about language-model circuits.

## 1. Motivation and scope

Interpretability methods often identify neural features using probes, dimensionality reduction, dictionary learning and activation-response studies. A representation that predicts a future label or obeys a symmetry is interesting on its own. Establishing how a model *uses* that representation requires intervention at the model's actual computational boundary.

Our study started from the [Compactification-Theory](https://github.com/MoralityLabAI/Compactification-Theory/pull/28) question: when does compressing a state remove distinctions needed to predict its behavior under future actions? Adict's modular-addition experiments made exact task quotients and numeric readout studies feasible on a CPU. Their central negative result arose when a discovered, decodable group representation failed a native causal-use test.

We use *measured information* only relative to a declared input distribution and target. We use *causal effect* only relative to a declared intervention operator, site, recipient/donor distribution and native output head. The general quotient mathematics belongs to the Compactification-Theory and EvalMath work; this paper develops an **operational audit taxonomy** and assembles the completed experimental evidence.

## 2. Four-axis experimental contract

Let \(X\) be a finite state set, \(F_a:X\to X\) an action, \(h:X\to\mathcal Y\) a task and \(C:X\to\mathcal Z\) an encoder. For a future action \(a\), write \(Y_a=h(F_a(X))\).

| Axis | Operational measurement | What must be disclosed |
| --- | --- | --- |
| Storage | Optimal one-hot squared-error risk of a decoder with unrestricted access to the entire finite code table | Enumerated domain, state distribution, target, code precision and oracle's access to all labels |
| Accessibility | Held-out error from a fitted, budget-constrained decoder | Architecture, labeled and unlabeled samples, model selection, fixed group priors and fitting/inference budget |
| Structure | Residual of \(C\circ F_a\) against a learned or prescribed abstract transition \(G_a\) | Input-shift graph, operator class, normalization, fitting subset and unseen-action or unseen-pair test |
| Native causal use | Counterfactual output of the original model after a subspace edit | Frozen native head, injection point, donor rule, unedited/full-donor/complement/random controls and denominator |

A model may pass different subsets of these tests. This observation motivates recording each result as a separate evidence claim.

### 2.1 Storage via exact finite-code risk

For categorical \(Y\) encoded as a one-hot vector, the optimal unrestricted squared Euclidean risk is

\[
R_{\mathrm{oracle}}(C;Y)
= \mathbb E\left[1-\sum_{y\in\mathcal Y}\Pr(Y=y\mid C(X))^2\right].
\]

On a fully enumerated finite domain with uniform states, every probability is rational. The formula is equivalent to drawing a state \(x\) uniformly, drawing \(x'\) uniformly from the fiber \(C^{-1}(C(x))\), and calculating the chance that \(Y(x)\ne Y(x')\). It is implemented using `fractions.Fraction` in [`four_axis_audit.py`](four_axis_audit.py).

If \(D=f\circ C\) is a deterministic further compression, and the target \(Y\) is held fixed, conditional expectation gives \(R_{\mathrm{oracle}}(D;Y)\ge R_{\mathrm{oracle}}(C;Y)\). This is the basis for the persistent fixed-target loss threshold in the [upstream task-closure note](https://github.com/MoralityLabAI/Compactification-Theory/pull/28). Risk for the changing compressed successor, where the target changes at each scale, need not be monotone.

### 2.2 Decoder cost and inductive bias

Given an encoder \(C\), decoder-learning procedure \(\mathcal L\), and \(n\) labeled examples \(D_n\), define

\[
R_{\mathcal L,n}(C;Y)
= \mathbb E_{D_n}\,\mathbb E\bigl[\ell(\mathcal L(D_n)(C(X)),Y)\bigr].
\]

Its meaning depends on the declared learning procedure and data distribution. Accuracy from one training/test split is evidence about that fitted decoder and split. A Fourier-product decoder receives a powerful structural prior, which must be costed alongside its fitted coefficients. An unrestricted lookup oracle can use every code-target association, even when the fitted learner has never seen the held-out code.

### 2.3 Group structure and native intervention

Approximate equivariance asks whether \(C(F_a(x))\approx G_a(C(x))\) for a declared operation. Small residual alone permits a zero code. The anti-collapse experiments use norm and covariance control to encourage nondegenerate phase coordinates, while reporting unexpected bases and dependent-character failures.

For a discovered subspace \(S\), donor code \(z_d\), recipient code \(z_r\) and projector \(P_S\), the native intervention at the declared site is

\[
\widetilde z=z_r+P_S(z_d-z_r).
\]

The edit is continued through the *original frozen output head*. Its prediction is compared with the actual native output at the donor input. A separate refitted probe is useful for accessibility, but does not measure the same causal property.

## 3. Analytic separations

### Proposition A: an injective phase code can defeat a bounded affine readout

Take independent uniform \(s,a\in\mathbb Z_p\), where \(p\) is prime, and let

\[
C(s,a)=\bigl(\cos(2\pi s/p),\sin(2\pi s/p),\cos(2\pi a/p),\sin(2\pi a/p)\bigr).
\]

The code is injective on the \(p^2\) states. It therefore permits zero oracle error on \(Y=(s+a)\bmod p\). The optimal **affine one-hot squared-error** decoder has risk \(1-1/p\), while a character-product decoder can recover \(Y\) exactly.

**Proof.** The two circle phases determine \((s,a)\). Every affine prediction on the four real coordinates has an additive decomposition into a function of \(s\) and a function of \(a\), plus an intercept. Since \(Y\) is uniform conditional on either input, its centered one-hot vector is orthogonal in squared-error geometry to all such additive functions. Its least-squares affine projection is consequently the constant vector \((1/p,\ldots,1/p)\), whose risk equals \(1-1/p\). The identity \(e^{2\pi i(s+a)/p}=e^{2\pi is/p}e^{2\pi ia/p}\) yields exact product decoding. \(\square\)

At \(p=31\), the affine population risk is \(30/31\). This compares specific representational and decoder classes. It supplies no lower bound against arbitrary nonlinear learners.

### Proposition B: external decoding can be perfect while native use is absent

Let the activation code be \(z=(\chi(a),\chi(b),\chi(a+b))\), with \(\chi(t)=e^{2\pi it/p}\). An external reader can multiply the first two coordinates and recover \(a+b\). Define the original model's task head to read exclusively the third coordinate. Editing the first two coordinates while preserving the third leaves its output unchanged.

**Proof.** The native head factors through the projection onto the third coordinate; its value is therefore invariant to changes in the first two. \(\square\)

The bypass example is implemented in the [planning fixtures](closeout_planning/sandbox_gate_probe.py). It motivates a native test for discovered features and illustrates why a high probe score alone cannot establish the native mediation claim.

## 4. Synthetic evidence

### 4.1 Full-code injectivity and limited decoder access

The initial 32-coordinate network was trained to predict modular sums on all \(31^2=961\) input pairs. A 480/481 pair split was used only when training and assessing post-hoc decoders. The recorded 96-bit quantization distinguished all 961 states for each of six conditions, so the exhaustive finite-code oracle recorded zero shear loss. Fitted affine shear decoders nevertheless performed poorly. This is a *decoder-split* study; the encoder was exposed to every input state. See [UPSTREAM_RESULTS.md](UPSTREAM_RESULTS.md) and [original_results.json](original_results.json).

A pure equivariance objective initially allowed phase-coordinate collapse. Unit norm and covariance regularization yielded useful, sometimes unexpectedly oriented characters. These engineered training signals form a substantial prior; see [ANTI_COLLAPSE.md](ANTI_COLLAPSE.md).

### 4.2 BF-02: discovery of an abstract shift action

The coordinate-free discovery procedure receives the real 32-dimensional code, the group order 31, and opaque state pairs linked by declared additive generators. It fits shift operators, extracts candidate complex eigenfeatures, and identifies character frequencies without the supplied activation-coordinate pairing. A small product decoder then predicts a held-out shear target.

On its prespecified confirmation seed 311, BF-02 recovered character frequencies \((0,1),(1,0)\), reached **0.9834** held-out shear accuracy on 481 pairs, and measured group-law residual **\(1.74\times10^{-4}\)**. A fixed orthogonal rotation of the code preserved the result. A privileged decoder given the original phase-coordinate pairing reached 1.000. The reported method succeeds on this engineered encoder with small character exponents, the known group order, and supplied generator relations. See [BLIND_SUBSPACE.md](BLIND_SUBSPACE.md) and [blind_subspace_results.json](blind_subspace_results.json).

### 4.3 C1: a predeclared test of native causal use

The subsequent C1 protocol froze seeds 401, 409 and 419, the fitting split, decoder settings, donor/recipient sampling and comparison controls before collecting results. It trained a task-only reference and a norm/covariance equivariant condition per seed. For every seed, it patched the discovered subspace in the 32-dimensional code immediately before the native `read_sum` head. The primary gate required at least **0.80** correct counterfactual prediction and at least **0.20** lift over the best of 16 rank-matched random-subspace controls on each new seed.

| Seed | Native present-task accuracy | External shear (2,1) decoder | Native discovered-subspace edit | Native complement edit | Full donor-code edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 401 | 1.000 | 0.761 | **0/256** | **256/256** | **256/256** |
| 409 | 1.000 | 0.956 | **0/256** | **256/256** | **256/256** |
| 419 | 1.000 | abstained | **0/256** | **256/256** | **256/256** |

All 16 norm-matched random controls per seed scored zero. The native prediction remained the recipient's prediction when the discovered subspace was edited. Seed 419 produced dependent group frequencies and correctly abstained; seed 401 also failed the required decoder accuracy. The [C1 report](C1_REPORT.md) and [machine-readable results](c1_results.json) retain those failures. C1's fixed G1 gate failed on all seeds, so the planned model-transfer stage and Silico GPU experiment were skipped.

The native result is specific to this head, this intervention and the available encoder seeds. A feature identified through eigenvector functionals can define an oblique subspace while the experiment uses orthogonal projection, and the edited codes can be off-manifold. The findings establish failure of the **specified** native interchange to mediate the behavior. They do not establish that the feature has no conceivable causal effect elsewhere.

## 5. Reusable software and evidence boundary

[`four_axis_audit.py`](four_axis_audit.py) is a Python standard-library command that loads `c1_manifest.json`, `c1_results.json`, `blind_subspace_results.json` and `original_results.json`. It checks source schema, planned seed/condition coverage, explicit abstentions, sample denominators, random-control counts and the saved versus recomputed G1 decision. It emits per-axis measurements and SHA-256 source digests. Its `finite_code_risk` function independently calculates exact rational oracle loss from newly supplied finite code and target tables.

Run from the nested Adict lab root:

```bash
python -m research.accessible_causal_information.four_axis_audit --out out/aci/four_axes.json
python -m unittest discover -s research/accessible_causal_information -p test_four_axis_audit.py -t . -v
```

The C1 run itself recorded 58 passing scoped tests under normal and optimized Python and recorded checks of 17 checkpoint hashes. A prior closeout review rechecked the aggregate verdict and seven independent planning fixtures. The new audit checks consistency of the saved **aggregate numerical evidence**, rather than independently reconstructing the native tensor edits or verifying checkpoint bytes. The output preserves this distinction explicitly. See [CLAIM_INDEX.json](CLAIM_INDEX.json) and [FOUR_AXIS_README.md](FOUR_AXIS_README.md).

## 6. Limitations and implications

All model experiments here use small engineered neural networks for modular arithmetic. Training exposed the encoders to all 961 states and explicitly included equivariant objectives. The discovery algorithm knew the group order and shift graph. Computational access and labeled/unlabeled samples differ across decoders, making broad sample-efficiency rankings inappropriate. Three new encoder seeds establish a robust failure of the predeclared C1 gate, while providing a limited basis for inference beyond this model family.

This study identifies an operational **measurement hierarchy** for mechanistic interpretability. Storage and decoder accuracy characterize potential information in a representation, a fitted transformation describes its geometry, and native intervention tests whether the model's own computation depends on the discovered structure. Interpretable features whose native edit has little impact remain useful descriptive findings with a narrower claim. The negative gate was sufficient to terminate the prespecified GPU expansion.

## References and primary evidence

- [Compactification-Theory PR #28](https://github.com/MoralityLabAI/Compactification-Theory/pull/28): exact task quotients, fixed-target risk and intervention-family scope.
- Nanda et al., [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217), ICLR 2023: modular-arithmetic circuit analysis.
- Geiger et al., [*Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*](https://www.jmlr.org/papers/v26/23-0058.html), JMLR 2025: broader causal abstraction.
- Adict: [FINAL_REPORT.md](FINAL_REPORT.md), [BLIND_SUBSPACE.md](BLIND_SUBSPACE.md), [C1_REPORT.md](C1_REPORT.md), [CLAIM_INDEX.json](CLAIM_INDEX.json).
