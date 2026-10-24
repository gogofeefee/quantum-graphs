#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics.py — DynamicsMixin for classical_graphs.

Perron-Frobenius operator, spectral analysis, invariant measure, ergodicity /
mixing tests, and rho evolution.
"""

import warnings
import numpy as np

from ._utils import _safe_logical_allclose_nonnegative


class DynamicsMixin:
    """
    Classical Perron-Frobenius dynamics on directed bonds.

    Depends on CoreMixin attributes: directed_bonds, directed_bond_index,
    vertex_scattering_matrix, virtual_port_map, valencies, n_directed_bonds.
    """

    # ==================================================================
    # Internal scattering matrix
    # ==================================================================

    def _build_sigma_alpha_alpha(self, k=None):
        """
        Build the internal directed-bond scattering matrix Sigma_aa.

        For each incoming directed bond src -> dst, the wave arrives at vertex
        v = dst[0] through port dst[1] and is scattered to all outgoing ports.

        Returns
        -------
        Sigma_aa : np.ndarray, shape (2B, 2B), complex
        """
        return np.array(self.model.block("aa", k))

    # ==================================================================
    # Perron-Frobenius operator
    # ==================================================================

    def build_perron_frobenius_operator(self, k=None, check_stochastic=True, atol=1e-12):
        """
        Build the classical Perron-Frobenius operator P = |Sigma_aa|².

        Parameters
        ----------
        k : float or None
            Optional spectral parameter for callable vertex matrices.
        check_stochastic : bool
            If True, validate that P is column-stochastic.
        atol : float
            Tolerance for the stochastic check.

        Returns
        -------
        P : np.ndarray, shape (2B, 2B), float
        """
        Sigma_aa = self._build_sigma_alpha_alpha(k=k)
        P = np.asarray(np.abs(Sigma_aa) ** 2, dtype=float)

        if check_stochastic:
            self._check_perron_frobenius_operator(P, atol=atol)

        self.Sigma_aa = Sigma_aa
        self.P = P
        return P

    def _check_perron_frobenius_operator(self, P, atol=1e-12):
        P = np.asarray(P, dtype=float)
        if P.shape != (self.n_directed_bonds, self.n_directed_bonds):
            raise ValueError(f"Invalid PF shape {P.shape}.")
        if not _safe_logical_allclose_nonnegative(P, atol=atol):
            raise ValueError(
                f"P contains significantly negative entries (min={np.min(P):.3e})."
            )
        col_sums = np.sum(P, axis=0)
        if not np.allclose(col_sums, 1.0, atol=atol, rtol=0.0):
            max_dev = np.max(np.abs(col_sums - 1.0))
            warnings.warn(
                f"P is not perfectly column-stochastic (max |col_sum - 1| = {max_dev:.3e}).",
                RuntimeWarning,
            )

    def get_perron_frobenius_operator(self, k=None, rebuild=False, **kwargs):
        """Return P, building it first if necessary."""
        if rebuild or not hasattr(self, "P"):
            return self.build_perron_frobenius_operator(k=k, **kwargs)
        return self.P

    # ==================================================================
    # Spectral analysis
    # ==================================================================

    def diagonalize_perron_frobenius(self, k=None, rebuild=False):
        """
        Eigendecompose P.

        Returns
        -------
        evals : np.ndarray — sorted by decreasing |eigenvalue|
        evecs_right : np.ndarray — right eigenvectors as columns
        """
        P = self.get_perron_frobenius_operator(k=k, rebuild=rebuild)
        evals, evecs_right = np.linalg.eig(P)
        order = np.argsort(-np.abs(evals))
        evals = evals[order]
        evecs_right = evecs_right[:, order]
        self.P_evals = evals
        self.P_evecs_right = evecs_right
        return evals, evecs_right

    def get_invariant_measure(self, k=None, rebuild=False, atol=1e-12, normalize=True):
        """
        Compute the invariant probability vector rho* with P @ rho* = rho*.

        Returns
        -------
        rho_star : np.ndarray, shape (2B,)
        """
        evals, evecs = self.diagonalize_perron_frobenius(k=k, rebuild=rebuild)
        idx = np.argmin(np.abs(evals - 1.0))
        if np.abs(evals[idx] - 1.0) > atol:
            warnings.warn(
                f"No eigenvalue within atol={atol:.1e} of 1. "
                f"Closest: {evals[idx]}",
                RuntimeWarning,
            )
        vec = np.real_if_close(evecs[:, idx]).astype(float)
        if np.sum(vec) < 0:
            vec = -vec
        vec = np.where(vec < 0.0, 0.0, vec)
        if normalize:
            s = np.sum(vec)
            if s <= 0:
                raise ValueError("Invariant measure normalization failed: nonpositive sum.")
            vec /= s
        self.rho_invariant = vec
        return vec

    def spectral_gap(self, k=None, rebuild=False):
        """
        Spectral gap = 1 - |lambda_2|, where lambda_2 is the second-largest eigenvalue.
        """
        evals, _ = self.diagonalize_perron_frobenius(k=k, rebuild=rebuild)
        if evals.size < 2:
            return 0.0
        gap = float(1.0 - np.abs(evals[1]))
        self.P_spectral_gap = gap
        return gap

    def is_ergodic(self, k=None, rebuild=False, atol=1e-10):
        """True if eigenvalue 1 is simple."""
        evals, _ = self.diagonalize_perron_frobenius(k=k, rebuild=rebuild)
        return int(np.sum(np.abs(evals - 1.0) < atol)) == 1

    def is_mixing(self, k=None, rebuild=False, atol=1e-10):
        """True if 1 is simple and all other eigenvalues satisfy |lambda| < 1."""
        evals, _ = self.diagonalize_perron_frobenius(k=k, rebuild=rebuild)
        evals = evals[np.argsort(-np.abs(evals))]
        if int(np.sum(np.abs(evals - 1.0) < atol)) != 1:
            return False
        others = np.delete(evals, np.argmin(np.abs(evals - 1.0)))
        return others.size == 0 or bool(np.all(np.abs(others) < 1.0 - atol))

    def classical_summary(self, k=None, rebuild=False, atol=1e-10):
        """
        Return a compact dict summary of the classical dynamics:
        eigenvalues, right_eigenvectors, spectral_gap, is_ergodic,
        is_mixing, invariant_measure.
        """
        evals, evecs = self.diagonalize_perron_frobenius(k=k, rebuild=rebuild)
        rho_star = self.get_invariant_measure(k=k, rebuild=False)
        gap = self.spectral_gap(k=k, rebuild=False)
        return {
            "eigenvalues": evals,
            "right_eigenvectors": evecs,
            "spectral_gap": gap,
            "is_ergodic": self.is_ergodic(k=k, rebuild=False, atol=atol),
            "is_mixing": self.is_mixing(k=k, rebuild=False, atol=atol),
            "invariant_measure": rho_star,
        }

    # ==================================================================
    # Probability-vector utilities
    # ==================================================================

    def validate_rho(self, rho, normalize=False, atol=1e-14):
        """
        Validate and optionally normalise a probability vector on directed bonds.

        Returns
        -------
        rho : np.ndarray, shape (2B,)
        """
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.shape != (self.n_directed_bonds,):
            raise ValueError(
                f"rho must have shape ({self.n_directed_bonds},), got {rho.shape}."
            )
        if np.min(rho) < -atol:
            raise ValueError(
                f"rho contains significantly negative entries (min={np.min(rho):.3e})."
            )
        rho = np.where(rho < 0.0, 0.0, rho)
        if normalize:
            s = np.sum(rho)
            if s <= 0:
                raise ValueError("Cannot normalise rho: sum is nonpositive.")
            rho = rho / s
        return rho

    def basis_state(self, index=None, dbond=None):
        """
        Return a one-hot probability vector concentrated on one directed bond.

        Provide exactly one of `index` (integer) or `dbond` (bond key).
        """
        if (index is None) == (dbond is None):
            raise ValueError("Provide exactly one of 'index' or 'dbond'.")
        if dbond is not None:
            index = self.directed_bond_index[dbond]
        rho0 = np.zeros(self.n_directed_bonds, dtype=float)
        rho0[int(index)] = 1.0
        return rho0

    def evolve_rho(self, rho0, n_steps=1, k=None, rebuild=False, normalize_input=True):
        """
        Evolve rho for n_steps: rho_{n+1} = P @ rho_n.

        Returns
        -------
        rho_n : np.ndarray, shape (2B,)
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0.")
        P = self.get_perron_frobenius_operator(k=k, rebuild=rebuild)
        rho = self.validate_rho(rho0, normalize=normalize_input)
        for _ in range(int(n_steps)):
            rho = P @ rho
        return rho

    def evolve_rho_trajectory(self, rho0, n_steps, k=None, rebuild=False,
                              normalize_input=True):
        """
        Return the full trajectory [rho_0, rho_1, ..., rho_n].

        Returns
        -------
        trajectory : np.ndarray, shape (n_steps+1, 2B)
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0.")
        P = self.get_perron_frobenius_operator(k=k, rebuild=rebuild)
        rho = self.validate_rho(rho0, normalize=normalize_input)
        traj = [rho.copy()]
        for _ in range(int(n_steps)):
            rho = P @ rho
            traj.append(rho.copy())
        return np.asarray(traj, dtype=float)

    def evolve_and_store(self, rho0, n_steps, k=None, rebuild=False,
                         normalize_input=True):
        """Compute trajectory and store in self.last_rho_trajectory."""
        trajectory = self.evolve_rho_trajectory(
            rho0=rho0, n_steps=n_steps, k=k,
            rebuild=rebuild, normalize_input=normalize_input,
        )
        self.last_rho_trajectory = trajectory
        return trajectory

    def perron_power(self, n, k=None, rebuild=False):
        """Return P^n."""
        if n < 0:
            raise ValueError("n must be >= 0.")
        P = self.get_perron_frobenius_operator(k=k, rebuild=rebuild)
        return np.linalg.matrix_power(P, int(n))
