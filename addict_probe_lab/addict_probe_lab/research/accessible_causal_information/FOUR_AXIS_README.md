# Four-Axis Interpretability Audit

This is the reusable software companion to [**Accessible Causal Information: When Decodable Neural Features Do Not Mediate Decisions**](PAPER.md).

Its purpose is to keep four empirical claims separate: **storage** (ideal finite-state recoverability), **accessibility** (bounded decoder held-out performance), **structure** (equivariance/operator quality), and **native causal use** (effect of replacing a feature subspace before the *original* model's head). In the completed Adict study, the first three axes had positive examples; the prospective native causal gate **failed**.

The CLI is deliberately modest: it recomputes the frozen **decision logic from saved C1 aggregate measurements** and checks that those records agree with the frozen manifest and each other. It does not reconstruct tensor-level results from checkpoints or assign statistical confidence to the three model seeds.

## One-minute run

From the root `addict_probe_lab/addict_probe_lab`, with Python 3.10+:

```bash
python -m research.accessible_causal_information.four_axis_audit --out out/aci/four_axes.json
python -m unittest discover -s research/accessible_causal_information -p test_four_axis_audit.py -t . -v
python -O -m unittest discover -s research/accessible_causal_information -p test_four_axis_audit.py -t . -q
```

Zero third-party packages, zero GPU calls and zero model training. The default input directory is the location of the module. `--root` accepts a copy of that folder, and `--require-pass` exits with status **2** when the frozen C1 gate failed (useful for blocking a more expensive downstream run). Without `--require-pass`, failure is a **valid reported scientific outcome**, so the CLI exits with status 0.

Expected high-level output, expressed as JSON fields (numerical values sourced from the committed C1/ BF-02/original result tables):

```json
{
  "schema": "adict-four-axis-audit/v1",
  "status": "closed_g1_failed",
  "c1": {"all_pass": false, "native_all_pass": false}
}
```

The full JSON also includes 3 per-seed gate results, held-out decoder values, discovered frequencies and residuals, native interventions and controls, file SHA-256 digests, and explicit validation boundaries.

## What the checker enforces

- Accepts only the completed study's three fixed input schemas and the stated six-condition original pilot.
- Confirms the C1 results embed exactly the externally supplied manifest, including predeclared action family and thresholds.
- Verifies the complete set of prospective seeds and conditions; duplicate or missing runs fail.
- Counts a decoder abstention as failure and requires its accuracy field to be null.
- Recomputes the native pass criterion, including the **best** norm-matched random subspace and each required threshold.
- Checks reported C1 gate decisions and BF-02 confirmation metrics against the actual saved numerical entries.
- Checks native percentages represent integer successes out of the precommitted 256 pairs.
- Retains provenance warnings: recorded metrics are not raw-event or binary-checkpoint replays.

The exported `provenance.sha256` block pins the four JSON source files used in that run. The historical `CLAIM_INDEX.json` identifies source-hash and saved-checkpoint claims from the earlier C1/C3 executions.

## API for new finite-state examples

`finite_code_risk(codes, labels)` accepts one code and one categorical target per enumerated state. It returns the **exact** one-hot squared-error Bayes risk as `fractions.Fraction`, under the uniform distribution on the supplied rows. States repeated in the input receive proportionate weight. It does not fit a decoder.

```python
from fractions import Fraction
from research.accessible_causal_information.four_axis_audit import finite_code_risk

assert finite_code_risk([0,0,1,1], [0,1,1,1]) == Fraction(1,4)
assert finite_code_risk(list(range(4)), [0,1,1,1]) == 0
```

This API helps establish a finite-state storage limit before spending time on learned probes or interventions.

## Integration and limits

The tool's output constitutes a **structured research receipt**: the frozen audit's computations and the SHA-256 digests are deterministic for unchanged source JSON. GitHub Actions can publish the resulting JSON via `four-axis-audit.yml` when a new research file changes. The script deliberately performs no modification of original evidence files.

A general mechanistic claim requires new native-model tests, with intervention scope and random-subspace controls defined in advance. See [FINAL_REPORT.md](FINAL_REPORT.md), [BLIND_SUBSPACE.md](BLIND_SUBSPACE.md) and [C1_REPORT.md](C1_REPORT.md) for empirical details and preserved failures.
