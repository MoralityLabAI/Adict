"""Exact phase-product readout fixture, separate from learned activations.

This file is an independent small reference implementation. It does not
reproduce the unpublished Silico phase-sample-efficiency patch.
"""
from __future__ import annotations

import cmath
import math


def root(s: int, p: int = 31) -> complex:
    """Canonical complex character for a residue class."""
    if type(p) is not int or p < 2:
        raise ValueError("invalid modulus")
    return cmath.exp(2j * math.pi * (s % p) / p)


def phase_code(s: int, a: int, p: int = 31) -> tuple[float, ...]:
    """Two independently encoded cyclic variables, four real coordinates."""
    z, w = root(s, p), root(a, p)
    return (z.real, z.imag, w.real, w.imag)


def label_offset(s: int, a: int, observed_label: int, p: int = 31) -> int:
    """Recover a constant output-coordinate offset using ONE labeled pair."""
    return (observed_label - s - a) % p


def decode_phase(s: int, a: int, offset: int, p: int = 31) -> int:
    """Fixed degree-two phase product and root-of-unity scoring."""
    z = root(s, p) * root(a, p) * root(offset, p)
    return max(range(p), key=lambda k: (z * root(-k, p)).real)


def affine_onehot_risk(p: int = 31) -> float:
    """Best population squared one-hot risk of any additive score f(s)+g(a).

    Uniform independent s,a imply E[e_(s+a) | s] = E[e_(s+a) | a]
    = uniform class probability vector. Orthogonal projection onto the
    additive subspace is therefore constant. Risk = 1 - 1/p.
    """
    if type(p) is not int or p < 2:
        raise ValueError("invalid modulus")
    return (p - 1) / p
