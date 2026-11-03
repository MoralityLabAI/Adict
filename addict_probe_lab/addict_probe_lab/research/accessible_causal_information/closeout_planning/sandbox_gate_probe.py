"""Independent analytic fixtures for choosing closeout gates; no learned weights.

Run: python sandbox_gate_probe.py --out /tmp/closeout_gate_results.json
No Adict experiment is rerun by this script. Shift-noise samples are synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

P = 31
TARGET = np.array([2, 1], dtype=np.int64)
BASES = {
    "small": ((1, 0), (0, 1)),
    "medium": ((1, 0), (26, 1)),
    "large": ((1, 0), (26, 29)),
}
NOISE_LEVELS = (0.0, 0.002, 0.01, 0.03, 0.05)


def all_states(p: int = P) -> np.ndarray:
    if type(p) is not int or p < 2:
        raise ValueError("modulus must be an integer >= 2")
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return np.column_stack((a.ravel(), b.ravel()))


def coefficients(basis, target=TARGET, p: int = P) -> np.ndarray:
    """Unique signed representatives for an invertible two-character basis."""
    matrix = np.asarray(basis, dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError("two frequency rows required")
    for c in all_states(p):
        if np.array_equal(c @ matrix % p, np.asarray(target) % p):
            det = int(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
            if np.gcd(det, p) != 1:
                raise ValueError("basis is singular modulo p")
            return np.where(c > p // 2, c - p, c)
    raise ValueError("target has no representation in this basis")


def chars(states: np.ndarray, basis, p: int = P) -> np.ndarray:
    return np.exp(2j * np.pi * (states @ np.asarray(basis).T % p) / p)


def read_product(z: np.ndarray, coeff: np.ndarray, p: int = P) -> np.ndarray:
    if np.any(np.abs(z) == 0):
        raise ValueError("zero feature has no phase")
    unit = z / np.abs(z)
    phase = np.prod(unit ** np.asarray(coeff)[None, :], axis=1)
    return np.rint(np.angle(phase) * p / (2 * np.pi)).astype(np.int64) % p


def worst_phase_budget(coeff, max_phase_errors, calibration_error: float = 0.0) -> float:
    """Bound on unwrapped composition error, in radians; no residual proxy."""
    errors = np.asarray(max_phase_errors, dtype=float)
    if np.any(errors < 0) or calibration_error < 0:
        raise ValueError("absolute error bounds must be nonnegative")
    return float(np.abs(coeff) @ errors + calibration_error)


def native_use_fixture(p: int = P) -> dict:
    """Identical activations and outputs with different use of a candidate subspace.

    z=(chi(a),chi(b),chi(a+b)). The first model reads the product of the
    first two blocks; the second reads the bypass block. Exchange the first
    two blocks with a donor (a+1,b), then evaluate each ORIGINAL readout.
    """
    x = all_states(p)
    roots = np.exp(2j * np.pi * np.arange(p) / p)
    z = np.column_stack((roots[x[:, 0]], roots[x[:, 1]], roots[x.sum(1) % p]))
    donor_a = (x[:, 0] + 1) % p
    patched = z.copy()
    patched[:, 0] = roots[donor_a]
    def decode(v):
        return np.rint(np.angle(v) * p / (2 * np.pi)).astype(np.int64) % p
    original = x.sum(1) % p
    counterfactual = (original + 1) % p
    live_before, bypass_before = decode(z[:, 0] * z[:, 1]), decode(z[:, 2])
    live_after = decode(patched[:, 0] * patched[:, 1])
    bypass_after = decode(patched[:, 2])
    return {
        "states": len(x),
        "activation_tables_identical": True,
        "original_outputs_identical": bool(np.array_equal(live_before, bypass_before)),
        "original_accuracy_both": float(np.mean(live_before == original)),
        "external_phase_probe_accuracy_both": float(np.mean(decode(z[:, 0]*z[:, 1]) == original)),
        "native_patch_counterfactual_accuracy_live": float(np.mean(live_after == counterfactual)),
        "native_patch_counterfactual_accuracy_bypass": float(np.mean(bypass_after == counterfactual)),
        "scope": "Analytic counterexample; these are defined readouts, not trained Adict models.",
    }


def run(seed: int = 20261103, repeats: int = 16) -> dict:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("positive integer repeats required")
    rng = np.random.default_rng(seed)
    states = np.tile(all_states(), (repeats, 1))
    labels = states @ TARGET % P
    # Common random numbers make noise comparisons paired within this fixture.
    standard_noise = rng.standard_normal((len(states), 2))
    records = []
    for name, basis in BASES.items():
        c = coefficients(basis)
        z = chars(states, basis)
        for sigma in NOISE_LEVELS:
            noisy = z * np.exp(1j * sigma * standard_noise)
            pred = read_product(noisy, c)
            # Markov bound on the square of the unwrapped error. Independence
            # and zero means are properties of THIS generator only.
            mse = sigma**2 * int(c @ c)
            records.append({
                "basis": name, "frequencies": basis, "signed_exponents": c.tolist(),
                "sigma_radians": sigma, "evaluations": len(states),
                "correct": int(np.sum(pred == labels)),
                "accuracy": float(np.mean(pred == labels)),
                "model_expected_unwrapped_phase_mse": mse,
                "model_error_probability_upper_bound": min(1.0, mse/(np.pi/P)**2),
                "max_each_error_for_sufficient_condition": float((np.pi/P)/np.abs(c).sum()),
            })
    return {
        "schema": "adict-closeout-planning-fixtures/v1",
        "scope": "Independent planning fixtures; no BF-02 rerun, neural training, GPU or API use.",
        "seed": seed, "repeats": repeats, "modulus": P,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "noise_model": "Independent zero-mean Gaussian phase errors; calibration error zero; known correct characters.",
        "phase_noise_records": records, "native_use_counterexample": native_use_fixture(),
        "warning": "Generator shift residual is not a uniform absolute phase-error bound. Noise results are descriptive samples.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(run(), indent=2) + "\n")
