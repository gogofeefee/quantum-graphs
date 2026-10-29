#!/usr/bin/env python3
"""
Compare two reference data sets produced by ``generate_reference.py``.

Usage
-----
    python tests/reference/compare_reference.py OLD.npz NEW.npz

Prints one line per quantity that differs beyond its tolerance and exits with
status 1 if any difference is found. Tolerances are chosen per quantity:
refined roots are only reproducible to the refinement accuracy, while
topology and orbit lengths must agree to round-off.
"""

import sys

import numpy as np

# (substring of the key, absolute tolerance, relative tolerance)
TOLERANCES = [
    ("/loc_", 1e-6, 1e-6),             # entropy / IPR of eigenvectors at refined levels
    ("/spectrum", 1e-9, 0.0),          # real levels (≤ 6e-8 apart up to reference_v7)
    ("/resonances", 1e-8, 0.0),        # refined complex poles
    ("/exp_resonances", 0.0, 0.0),     # grid points: exact
    ("/intensity", 1e-6, 1e-6),        # eigenspace projector diagonal at a refined level
    ("/residual", 1e-5, 0.0),          # |a - U a| (bounded, not compared tightly)
    ("/res_wf_residual", 1e-3, 0.0),
    ("/spacings", 1e-7, 1e-7),
    ("/sigma2", 1e-7, 1e-6),
    ("/delta3", 1e-7, 1e-6),
    ("/weyl", 1e-9, 1e-12),
    ("/ng", 1e-7, 1e-7),
    ("/S", 1e-10, 1e-10),
    ("/U_eigvals", 1e-9, 0.0),         # compared through power sums (see power_sums)
    ("/secular_det", 1e-9, 1e-9),
    ("/eigvals", 1e-9, 0.0),
    ("", 1e-10, 1e-10),                # default
]


def tolerance(key):
    for sub, atol, rtol in TOLERANCES:
        if sub and sub in key:
            return atol, rtol
    return TOLERANCES[-1][1:]


def wavefunction_by_label(data, key):
    """Return {bond label: intensity} so that a change of bond indexing is not a difference."""
    labels = data[key.replace("/intensity", "/labels")]
    return {tuple(row): val for row, val in zip(labels, data[key])}


def power_sums(eigvals):
    """Return p_n = sum_j lambda_j**n, n = 1..N, for each row of eigenvalues.

    The power sums determine the multiset of eigenvalues (Newton identities)
    without sorting it, and are insensitive to the LAPACK-dependent splitting
    of a defective eigenvalue (a Jordan block of size m spreads into a circle
    of radius ~eps**(1/m), i.e. ~0.2 for the zero eigenvalue of the classical
    Perron-Frobenius operators, which scrambles any sorted comparison).
    """
    e = np.atleast_2d(eigvals)
    n = np.arange(1, e.shape[-1] + 1)
    return (e[..., None, :] ** n[:, None]).sum(axis=-1)


def compare(old, new):
    problems = []
    for key in sorted(set(old.files) | set(new.files)):
        if key not in new.files:
            problems.append(f"MISSING in new : {key}")
            continue
        if key not in old.files:
            problems.append(f"EXTRA in new   : {key}")
            continue
        if key.endswith("/labels"):
            continue
        a, b = old[key], new[key]
        if key.endswith("/intensity"):
            da, db = wavefunction_by_label(old, key), wavefunction_by_label(new, key)
            if set(da) != set(db):
                problems.append(f"LABELS differ  : {key}")
                continue
            a = np.array([da[k] for k in sorted(da)])
            b = np.array([db[k] for k in sorted(da)])
        if key.endswith("eigvals") and a.shape == b.shape:
            a, b = power_sums(a), power_sums(b)
        if a.dtype.kind in "USO" or b.dtype.kind in "USO":
            if a.shape != b.shape or not np.all(a == b):
                problems.append(f"DIFFERENT      : {key}: {a.tolist()} -> {b.tolist()}")
            continue
        if a.shape != b.shape:
            problems.append(f"SHAPE          : {key}: {a.shape} -> {b.shape}")
            continue
        atol, rtol = tolerance(key)
        if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            err = np.max(np.abs(np.asarray(a, complex) - np.asarray(b, complex)))
            problems.append(f"VALUE          : {key}: max |diff| = {err:.3e} "
                            f"(atol={atol:g}, rtol={rtol:g})")
    return problems


if __name__ == "__main__":
    old = np.load(sys.argv[1], allow_pickle=False)
    new = np.load(sys.argv[2], allow_pickle=False)
    problems = compare(old, new)
    for line in problems:
        print(line)
    print(f"{len(old.files)} reference arrays, {len(problems)} difference(s).")
    sys.exit(1 if problems else 0)
