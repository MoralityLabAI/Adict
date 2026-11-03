# C1: prospective robustness and native causal use

**Outcome: G1 FAILED. Native interchange gate failed on all three new seeds. Per the plan, S1 is skipped and the route goes C1 -> C3.**

State: CPU run, 195 s, branch `research/blind-fourier-discovery`. Plan commit `02c2ad4`; pre-run freeze commit `0447fd5` (manifest, gates, code and tests, committed before any encoder was trained). Frozen BF-02 discovery settings were loaded unchanged and nothing was retuned. BF-01 and BF-02 reports and receipts are untouched.

## Artifact lock

All 11 BF-02 checkpoints resolve and match their file and state hashes. The frozen file, results file and all BF-02 source hashes match their receipts (`c1_start.json`). Six new checkpoints (seeds 401, 409, 419; `task_only`, `eq_norm_cov`; matched initial weights per seed) are saved in `c1_checkpoints/` before scoring, with file, state and activation hashes in `c1_checkpoints.json`. Split seed is 2026110301.

## Native causal gate (256 frozen test pairs, `eq_norm_cov`)

Patch site: the 32-d code immediately before the original `read_sum` head. Recipient `(a,b)`, donor `(a',b)`, target is the native head output at the donor input.

| Seed | Discovered-subspace interchange | Best of 16 norm-matched random | Full-code donor | Complement replacement | No edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 401 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 409 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 419 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |

Gate needed at least 0.80 accuracy and at least 0.20 lift over the best random control on every seed. Result: 0.00 and 0.00 lift on all three. Replacing the complement of the discovered 4-d phase subspace reproduces the counterfactual output 100% of the time, so the original head reads the other 28 code dimensions and ignores the phase subspace. Mean patch norm 1.31-1.34, displacement from the counterfactual code 6.4-6.5, off-target rate 0.000 (the patched output equals the recipient's output). Unscaled random edits reach at most 0.012. The zero-edit identity check scores 1.000. On `task_only`, discovery finds no features (rank 0), so the edit is a no-op and accuracy is 0.000. This is the "perfect probing, zero native use" pattern from the plan, produced by a trained model rather than a construction.

## Other G1 gates (`eq_norm_cov`)

| Seed | Present task | Discovered modes (true modes) | Decoder (2,1) / (1,2) | Held-out group residual | Rotation change |
| --- | ---: | --- | --- | ---: | ---: |
| 401 | 1.000 | (0,1),(1,0) (same) | 0.761 / 0.838 | 3.4e-4 | 0.000 |
| 409 | 1.000 | (0,1),(1,0) (same) | 0.956 / 0.994 | 2.2e-4 | 0.000 |
| 419 | 1.000 | (1,0) only (true: (1,0),(2,0)) | abstained / abstained | 0.0176 | undefined (counted as failure) |

- Present-task accuracy passes on all seeds.
- Small-exponent decoder (at least 0.95 on both targets) fails on seeds 401 and 419, and passes only on 409.
- Held-out residual (at most 0.01) fails on seed 419.
- Rotation gate fails on seed 419 because the decoder abstains.
- Seed 419's true modes `(1,0),(2,0)` are linearly dependent, so no independent pair exists. Discovery kept only one feature, and the decoder abstains in the denominator.

## Comparators (`eq_norm_cov`, target (2,1), 480 labels)

| Seed | Blind | Exponent search (privileged) | MLP | RFF | Monomial ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| 401 | 0.761 | 0.952 | 0.200 | 0.012 | 0.027 |
| 409 | 0.956 | 1.000 | 0.218 | 0.015 | 0.029 |
| 419 | abstain | 0.106 | 0.087 | 0.031 | 0.023 |

Access per method is declared in `c1_manifest.json` (`access_table`). The fixed-budget MLP (two hidden layers of 256, 1500 steps) does not learn the character product from 480 labels. Exponent search, which has the known pairing, beats the blind method on every seed with a usable discovery. No sample-efficiency superiority is claimed.

## Failures preserved

- Seed 401 decoding is 0.76 and 0.84, even though the modes are correct. Phase noise at the 31-class boundary is the likely cause, but this was not isolated.
- Seed 419 is a dependent-mode encoder, and discovery plus decoding abstain.
- `task_only` produces no features on any seed.
- Native interchange is zero everywhere.

## Boundaries

- The synthetic task head was trained on the sum label alone, with the 28 non-phase dimensions free. The result is specific to this checkpoint family and this intervention (replace the discovered real subspace with the donor's component).
- The subspace comes from eigenvector functionals, so the projector is orthogonal in code space, which may differ from the oblique feature subspace. Basis overlap with the true coordinates was 0.99998 in BF-02, so that is unlikely to explain a zero result. I did not test it here.
- The analytic high-exponent/noise table is in `closeout_planning/sandbox_results.json` and was not recomputed.

## Reproduce

From `addict_probe_lab/addict_probe_lab`:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/accessible_causal_information -t . -q
    PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s research/accessible_causal_information -t . -q
    PYTHONDONTWRITEBYTECODE=1 python -m research.accessible_causal_information.c1_closeout --out /tmp/c1
