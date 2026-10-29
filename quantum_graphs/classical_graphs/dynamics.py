#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics.py — DynamicsMixin for classical_graphs.

Perron-Frobenius operator, spectral analysis, invariant measure, ergodicity /
mixing tests, and rho evolution.
"""

import warnings
import numpy as np



class DynamicsMixin:
    """
    Classical Perron-Frobenius dynamics on directed bonds.

    One step of P moves the probability on a directed bond to the directed
    bonds leaving its end vertex (one bond traversal per step, whatever the
    bond length). Directed bonds are numbered as in
    :mod:`quantum_graphs.core.topology`. Every method takes ``k`` (used only
    by vertex matrices that depend on k); P and its spectrum are computed once
    per k.
    """

    # ==================================================================
    # Perron-Frobenius operator and its spectrum
    # ==================================================================

    def perron_frobenius(self, k=None, atol=1e-12):
        """P = |Σ_αα(k)|² (element-wise), shape (2B, 2B), column-stochastic for
        unitary vertex matrices (a warning is raised otherwise)."""
        cache = self.__dict__.setdefault("_pf_cache", {})
        if k not in cache:
            P = np.asarray(np.abs(self.model.block("aa", k)) ** 2, dtype=float)
            col_sums = np.sum(P, axis=0)
            if not np.allclose(col_sums, 1.0, atol=atol, rtol=0.0):
                warnings.warn("P is not column-stochastic "
                              f"(max |col_sum - 1| = {np.max(np.abs(col_sums - 1.0)):.3e}).",
                              RuntimeWarning, stacklevel=2)
            evals, evecs = np.linalg.eig(P)
            order = np.argsort(-np.abs(evals))
            cache[k] = (P, evals[order], evecs[:, order])
        return cache[k][0]

    def perron_frobenius_spectrum(self, k=None):
        """Eigenvalues of P sorted by decreasing modulus, and the right
        eigenvectors as columns."""
        self.perron_frobenius(k)
        _, evals, evecs = self._pf_cache[k]
        return evals, evecs

    def invariant_measure(self, k=None, atol=1e-12):
        """Probability vector rho* with P rho* = rho* (uniform for unitary vertices)."""
        evals, evecs = self.perron_frobenius_spectrum(k)
        idx = np.argmin(np.abs(evals - 1.0))
        if np.abs(evals[idx] - 1.0) > atol:
            warnings.warn(f"No eigenvalue within atol={atol:.1e} of 1. Closest: {evals[idx]}",
                          RuntimeWarning, stacklevel=2)
        vec = np.real_if_close(evecs[:, idx]).astype(float)
        if np.sum(vec) < 0:
            vec = -vec
        vec = np.where(vec < 0.0, 0.0, vec)
        return vec / np.sum(vec)

    def spectral_gap(self, k=None):
        """1 − |λ₂|, λ₂ the eigenvalue of second largest modulus (0 if |λ₂| = 1)."""
        evals, _ = self.perron_frobenius_spectrum(k)
        if evals.size < 2:
            return 0.0
        gap = float(1.0 - np.abs(evals[1]))
        return 0.0 if abs(gap) < 1e-12 else gap      # |λ2| = 1 up to round-off: no gap

    def is_ergodic(self, k=None, atol=1e-10):
        """True if the eigenvalue 1 is simple."""
        evals, _ = self.perron_frobenius_spectrum(k)
        return int(np.sum(np.abs(evals - 1.0) < atol)) == 1

    def is_mixing(self, k=None, atol=1e-10):
        """True if 1 is simple and every other eigenvalue has |λ| < 1."""
        evals, _ = self.perron_frobenius_spectrum(k)
        if int(np.sum(np.abs(evals - 1.0) < atol)) != 1:
            return False
        others = np.delete(evals, np.argmin(np.abs(evals - 1.0)))
        return others.size == 0 or bool(np.all(np.abs(others) < 1.0 - atol))

    def classical_summary(self, k=None, atol=1e-10):
        """dict: eigenvalues, right_eigenvectors, spectral_gap, is_ergodic,
        is_mixing, invariant_measure."""
        evals, evecs = self.perron_frobenius_spectrum(k)
        return {
            "eigenvalues": evals,
            "right_eigenvectors": evecs,
            "spectral_gap": self.spectral_gap(k),
            "is_ergodic": self.is_ergodic(k, atol=atol),
            "is_mixing": self.is_mixing(k, atol=atol),
            "invariant_measure": self.invariant_measure(k),
        }

    # ==================================================================
    # Probability vectors
    # ==================================================================

    def validate_rho(self, rho, normalize=False, atol=1e-14):
        """Check (and optionally normalise) a probability vector on the directed bonds."""
        rho = np.asarray(rho, dtype=float).reshape(-1)
        n = self.topology.n_directed
        if rho.shape != (n,):
            raise ValueError(f"rho must have shape ({n},), got {rho.shape}.")
        if np.min(rho) < -atol:
            raise ValueError(f"rho contains significantly negative entries (min={np.min(rho):.3e}).")
        rho = np.where(rho < 0.0, 0.0, rho)
        if normalize:
            s = np.sum(rho)
            if s <= 0:
                raise ValueError("Cannot normalise rho: sum is nonpositive.")
            rho = rho / s
        return rho

    def basis_state(self, index=None, dbond=None):
        """
        One-hot probability vector on one directed bond.

        Provide exactly one of `index` (integer) or `dbond` = (i, p, j, q), the
        directed bond from port (i, p) to port (j, q).
        """
        if (index is None) == (dbond is None):
            raise ValueError("Provide exactly one of 'index' or 'dbond'.")
        if dbond is not None:
            index = self.topology.directed(*dbond)
        rho0 = np.zeros(self.topology.n_directed, dtype=float)
        rho0[int(index)] = 1.0
        return rho0

    def evolve_rho(self, rho0, n_steps=1, k=None, trajectory=False, normalize_input=True):
        """rho_n = P^n rho0; with ``trajectory=True`` the array [rho_0, ..., rho_n],
        shape (n_steps + 1, 2B)."""
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0.")
        P = self.perron_frobenius(k)
        rho = self.validate_rho(rho0, normalize=normalize_input)
        traj = [rho]
        for _ in range(int(n_steps)):
            rho = P @ rho
            traj.append(rho)
        return np.asarray(traj, dtype=float) if trajectory else rho
