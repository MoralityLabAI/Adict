# BF-02: coordinate-free discovery of the shift representation

**State:** CPU run, 162 s, branch `research/blind-fourier-discovery`, base `a99fb6c`. BF-01 (`BLIND_FOURIER.md`) is unchanged. Protocol, selection rule and success criteria are in `blind_subspace_manifest.json`, written before any encoder was trained (`blind_subspace_start.json`). The frozen procedure is `blind_subspace_frozen.json`; the receipt hashes sources, start, frozen, results and checkpoints.

## Method

BF-01 assumed which coordinates pair into complex features. BF-02 sees only the real 32-d code and opaque shift index pairs `(x, Tx)` inside the fitting pool. It fits a ridge linear shift operator per generator, diagonalizes `B_a + 0.7371 B_b`, and turns each eigenvector with positive imaginary part into a complex feature. A feature is kept if its group-law residual is at most 0.05, its variance share is at least 0.01, and its frequency is not `(0,0)`. One feature is kept per frequency up to conjugation. Frequencies mod 31 come from the BF-01 multiplier estimator. The decoder picks the independent pair with the smallest signed exponents for the `(2,1)` shear and fits one output phase. No coordinate pairing, operand values, labels or target frequencies enter discovery (`test_discovery_signature_has_no_labels_pairing_or_targets`, `test_relabeling_states_leaves_frequencies_unchanged`).

Checkpoints: 11 encoder state dicts are in `bf02_checkpoints/` (576 KB). Each was saved, reloaded, and its activations recomputed from the reloaded weights; file, state and activation hashes are in `blind_subspace_checkpoints.json`. Retraining reproduces BF-01's equivariance losses to float precision for seeds 303 and 307, so these are the BF-01 encoders.

## Findings

- **Confirmation (seed 311, `eq_norm_cov`, 481 untouched pairs): all criteria met.** Discovery returned two features with frequencies `(0,1),(1,0)`, matching the FFT modes of the two phase blocks, with no spurious modes. The discovered subspace overlaps coordinates 0-3 at 0.99998. Group-law residual is 1.7e-4, and shear accuracy is 0.983.
- **Seed 303 (calibration seed, not independent):** `(1,0),(0,1)` recovered, accuracy 0.985. **Seed 307 (post hoc, not part of the criteria):** `(0,1),(1,0)` recovered, accuracy 1.000.
- **Rotation control.** After a fixed random orthogonal rotation of the 32-d code, blind discovery returns the same modes and the same accuracy (0.983 on seed 311). The coordinate-pairing baselines fall to 0.02-0.04.
- **Matched baselines, unrotated (seed 311).** Exponent search on the known pairing scores 1.000, so the blind method is 1.7 points below the best baseline. Monomial-product ridge scores 0.035 and random Fourier features 0.015. Exponent search is the right inductive bias, and what the blind method adds is coordinate-freedom, not accuracy. Monomial-product ridge and RFF do not represent the group law from labels alone.
- **Label efficiency.** Blind accuracy with 4, 16, 64 and 480 labels is 0.973, 0.983, 0.913 and 0.983 (seed 311), and 0.807, 0.838, 0.886 and 0.985 (seed 303). Exponent search scores 0.042 at 4 labels and at least 0.992 from 16.
- **Operators.** Linear shift operators explain 58-70% of next-state variance (relative error 0.30-0.42). The unused dimensions do not follow the action. The commutator is 0.11-0.12 over the full space and 0.005-0.007 on the discovered subspace. Shuffling the shift targets leaves relative error at 0.86-0.88.
- **Preserved failures.** `task_only`, `equivariance` and `task_and_a` produce no features, so decoding fails on both seeds. `eq_norm` produces none either, which is correct because its blocks are constant.
- **Calibration was weak.** On the 240 validation pairs, with only about 60 shift pairs, no configuration exceeded 0.83. The frozen configuration (ridge 1e-6, residual threshold 0.05, all shift pairs) won on validation accuracy. The reported 0.985 uses the larger 480-pair pool.

## Boundaries

- The shift graph and the group order 31 are given. The number of features depends on the residual and variance thresholds.
- Decoding amplifies phase noise at large exponents. On a synthetic code with frequencies `(5,17),(29,3)`, which need exponent 7, accuracy was 0.11, so cases were restricted to small exponents. The real encoders use `(2,1)` or `(1,2)`.
- This is a decoder-split test on synthetic data from one recipe and three seeds. It says nothing about LLM circuits.

## Reproduce

From `addict_probe_lab/addict_probe_lab`:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/accessible_causal_information -t . -q
    PYTHONDONTWRITEBYTECODE=1 python -m research.accessible_causal_information.blind_subspace --out /tmp/bf02
