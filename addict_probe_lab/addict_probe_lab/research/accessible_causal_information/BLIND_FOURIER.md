# BF-01: blind Fourier-basis discovery on anti-collapse encoders

**State:** CPU run, 139 s, branch `research/blind-fourier-discovery`, base `6505c0d`. Protocol, selection rule and success criteria are in `blind_fourier_manifest.json`, written before any encoder was trained (`blind_fourier_start.json`). The frozen procedure is `blind_fourier_frozen.json`; the receipt hashes sources, start, frozen and results.

## Method

Discovery sees the two complex phase features (code coordinates 0-1 and 2-3) and opaque index pairs `(x, Tx)` for `T(1,0)`, `T(0,1)` and their composite, with both endpoints inside the fitting pool. It never receives operand values, labels or target frequencies (`test_discovery_receives_no_labels_or_operands`). For each feature it fits a complex multiplier per generator, rounds the angle to a 31st root of unity, and returns `(k, l)` mod 31. The group-law residual is the largest mean squared error of `z(Tx)` against the exact rounded character action, over `T_a`, `T_b` and the composite. The decoder solves `c1(k1,l1)+c2(k2,l2) = (2,1)` mod 31, uses signed exponents on unit-normalized features, and fits one output phase on the shear labels.

## Findings

- **Encoders are not the AC-01 weights.** AC-01 saved no checkpoints, so both seeds were retrained from its manifest. Seed 303 `eq_norm_cov` reproduces (equivariance loss differs by 2.9e-5, modes `(1,0),(0,1)`). Seed 307 does not: its dominant modes are `(0,1),(1,0)`, not AC-01's `(1,0),(0,30)`. Training is not bit-reproducible here, so the evaluation seed is a fresh encoder from the same recipe, not the AC-01 seed-307 checkpoint.
- **Calibration did not discriminate.** All 24 configurations scored 1.000 on the 240 seed-303 validation pairs. The frozen configuration (least squares, unit normalization, signed lift, all shift pairs) came from the pre-declared tie-break alone. Ridge penalties for the generic decoders were tuned on the same validation split.
- **Primary result (seed 307, `eq_norm_cov`, 481 untouched pairs).** Both blocks' frequencies equal the FFT modes. Group-law residual is 1.6e-4 (limit 0.1). Blind shear accuracy is 1.000. Affine and bilinear decoders, on 4 phase coordinates or all 32 code coordinates, score 0.010 to 0.012. The oracle-assisted basis decoder also scores 1.000. All three criteria are met.
- **Seed 303** (calibration seed, so not independent): recovers `(1,0),(0,1)`, residual 1.9e-4, accuracy 1.000.
- **Label efficiency (seed 307).** The blind decoder scores 1.000 with 4 labels and 0.994 with 64. The generic decoders score 0.01 to 0.05 at every size up to 480.
- **Random rotated phase controls (50 draws per seed).** 32 of 50 (seed 303) and 34 of 50 (seed 307) draws gave dependent frequencies. The best surviving accuracy was 0.119. Median residual is 0.043, against 1.6e-4 for the real features.
- **Preserved failures.** On `task_only`, `equivariance` and `task_and_a`, recovery fails (residual 1.6 to 2.1, accuracy 0.02 to 0.04, or undefined). On `eq_norm` it returns frequencies `(0,0)` (residual ~3e-4), which makes the exponent system singular and the decoder fails. The blind method certifies a group action only when one exists.
- **Cost.** Blind: 4 discrete integers, 2 fitted scalars, about 250 unlabeled shift pairs per generator, wall time under 20 ms. Generic: 155 to 17,391 parameters and 480 labels. CPU-time fields in the results are below the Windows timer resolution and read as zero.

## Boundaries

- The generic decoders sit at chance (the 1/31 floor is 0.032). The margin shows they cannot represent a mod-31 character product from 4 or 32 coordinates, not that the blind method is more sample-efficient than a stronger learner. No MLP or Fourier-feature baseline was run.
- Features are paired as complex coordinates by the declared layout (0-1, 2-3). Discovering the pairing blindly is not tested.
- The shift graph is an intervention-style input: the method needs to know which state is `Tx`, though not what `a` or `b` are. The group order 31 is also given.
- Encoders train on all 961 pairs, so this is a decoder-split test, not zero-shot transfer. Evidence is one recipe, two retrained encoders, and a synthetic group. It says nothing about LLM circuits.

## Reproduce

From `addict_probe_lab/addict_probe_lab`:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/accessible_causal_information -t . -q
    PYTHONDONTWRITEBYTECODE=1 python -m research.accessible_causal_information.blind_fourier --out /tmp/bf01
