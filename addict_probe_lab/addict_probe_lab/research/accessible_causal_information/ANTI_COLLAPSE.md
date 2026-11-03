# AC-01: a nondegenerate representation can carry the wrong group action

**State:** CPU exploratory mechanistic-representation test on `MoralityLabAI/Adict`. The complete proof and full results are in the accompanying evidence archive. The preregistered loss comparison and the later exploratory basis diagnostic are distinguished below.

## Exact representation theorem

For `p=31`, let `X=Z_p × Z_p`, `T_a(a,b)=(a+1,b)`, `T_b(a,b)=(a,b+1)` and `h(a,b)=a+b`. Define `R` to be rotation by `2π/p` and the prescribed macro actions `ρ_a=R⊕R`, `ρ_b=R⊕I`. Every **exactly** equivariant map `E:X→R^4` has the form

    E(a,b) = (R^(a+b) u, R^a v)

for fixed `u,v ∈ R²`. Proof: start at `(0,0)` and apply `T_a^a T_b^b`; the representation matrices commute. Conversely that form is equivariant. Requiring unit norm in both phase blocks excludes the exact zero solution. Under the uniform domain and p>2, a unit-norm exact representation has mean zero and covariance `0.5 I_4` by character orthogonality.

The raw equivariance loss has exact collapsed optimum `E=0`. Unit-block norm *alone* also admits constant unit vectors; these violate equivariance but may be attractive approximate minima. An exact wrong-character counterexample is `(χ_2(a+b),χ_2(a))`: both norms are one and covariance is `0.5 I_4`, while its residual under prescribed unit rotations is `6 sin²(π/p)`. Therefore the moment constraints cannot, by themselves, certify the prescribed action. The code contains exhaustive finite-group tests of these statements.

## Fixed two-seed protocol and measurements

Both seeds 303/307 use the original 11,390-parameter encoder from Adict. All 961 state pairs enter full-batch encoder training. Each condition runs 1,500 AdamW steps using one CPU thread. The manifest fixes `λ_equiv=1`, `λ_norm=1` and `λ_cov=1`. Conditions are task-only, equivariance-only, equivariance+norm, equivariance+norm+covariance, and task-plus-a supervision. The first four hidden coordinates are prescribed as two phase pairs. The decoder is fitted on 480 pairs and scored on the unchanged 481-pair diagnostic split. The fixed phase-product decoder learns a single complex offset from the train split. No GPU, API model or Silico RU is used.

All ten trained models reach 100% present-sum accuracy. The equivariance-only phase energy falls to 4.4e-5 and 6.6e-5; phase norm raises energy to 0.890 and 0.963, but both blocks are almost constant. With norm plus covariance the covariance eigenvalues are near 0.5 and the learned blocks instead track **a and b**, or **a and -b**, with >99.9% dominant spectral power per block. The prescribed first block should track a+b, and the prescribed second should track a. Consequently the raw fixed product decoder remains near chance (3.12%, 2.91%), although the 96-bit oracle codes remain injective and have zero shear risk. The residual ~0.060 remains above the exact target (zero), consistent with the wrong-action-basis obstruction.

## Exploratory post-hoc character-basis control

After observing the fixed readout failure, a separate decoder was added. It identifies one character per phase block by maximizing complex correlation against all 31² characters using **only the original 480 fitting pairs** and known input operands. It solves the resulting 2×2 character-basis equation over `Z_31` for the shear target `(2,1)`, chooses small signed integer powers to avoid amplifying approximate-phase error, and fits one global output-phase offset using only fitting labels. The frozen 481-pair test set is then scored once.

Seed 303 selects `(1,0),(0,1)` and yields **100%** shear accuracy with exponents `(2,1)`. Seed 307 selects `(1,0),(0,30)` and yields **96.47%** with signed exponents `(2,-1)`. This decoder receives a strong cyclic prior, a 31²-mode search for each phase block, and the fitting pairs' input coordinates. These costs are part of the method. It was specified after seeing the original spectral failure, so the outcome is exploratory and requires untouched confirmation runs before a generalization claim. A 25-test suite, including a numerical regression for signed rather than 30th-power exponents, passes normally and under `python -O`.

## Boundaries

The encoders are trained on every state; this is a decoder-split test rather than a zero-shot intervention generalization result. The unrestricted 96-bit Bayes oracle is allowed an arbitrary finite lookup. Fixed decoder performance is relative to the declared representation coordinates. The post-hoc basis method is not evidence for causal LLM mechanisms, sparse autoencoders or a discovered model-general circuit. The current contribution is a reproducible synthetic example of basis mismatch and a group-aware recovery control.

## Reproduce

From `addict_probe_lab/addict_probe_lab` with PyTorch CPU:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/accessible_causal_information -q
    PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s research/accessible_causal_information -q
    PYTHONDONTWRITEBYTECODE=1 python -m research.accessible_causal_information.anti_collapse --out /tmp/adict-ac01

The source-hashed `start.json` is written before training; `receipt.json` records hashes of the start and full `results.json`. The receipt is an unsigned local reproducibility record, not externally witnessed experiment-start attestation. `anti_collapse_summary.csv` contains the main measurements and the separate evidence zip carries the complete results and run logs.
