#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralStatsMixin — Weyl law, unfolding, NNSD, Σ², Δ₃.
Module-level: NNSD_func (also re-exported from closed_graphs.__init__).
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def NNSD_func(s, wigner="GOE"):
    """
    Nearest-Neighbor Spacing Distribution (Wigner surmise).

    Parameters
    ----------
    s : float or array-like
    wigner : str
        'poissonian', 'GOE', 'GUE', or 'GSE' (case-insensitive).

    Returns
    -------
    float or ndarray
    """
    wigner = wigner.lower()
    if wigner == "poissonian":
        return np.exp(-s)
    elif wigner == "goe":
        return (np.pi / 2) * s * np.exp(-np.pi / 4 * s**2)
    elif wigner == "gue":
        return (32 / np.pi**2) * s**2 * np.exp(-4 / np.pi * s**2)
    elif wigner == "gse":
        return (2**18 / 3**6 / np.pi**3) * s**4 * np.exp(-64 / (9 * np.pi) * s**2)
    else:
        raise ValueError(
            "Invalid wigner type. Choose from 'poissonian', 'GOE', 'GUE', or 'GSE'."
        )


def _get_rmt_funcs():
    """Load sigma2_RMT and delta3_RMT from the wigner/ directory at the
    repository root (two levels above this package)."""
    wigner_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "wigner"))
    if wigner_path not in sys.path:
        sys.path.insert(0, wigner_path)
    from sigma2_RMT import sigma2_RMT
    from delta3_RMT import delta3_RMT
    return sigma2_RMT, delta3_RMT


class SpectralStatsMixin:

    # =========================================================================
    # Directional-graph detection
    # =========================================================================

    def _is_pure_directional(self):
        """Return True iff every level has deg == 2."""
        return (hasattr(self, "degeneracies") and
                self.degeneracies.size > 0 and
                np.all(self.degeneracies == 2))

    # =========================================================================
    # Spectrum selection
    # =========================================================================

    def _stats_selection(self, mode="auto"):
        """
        Choose spectrum and Weyl scale for spectral statistics.

        Parameters
        ----------
        mode : {"auto", "raw", "expanded", "directional"}

        Returns
        -------
        spectrum_used : np.ndarray
        weyl_scale : float
        note : str
        method : str
        """
        if mode == "raw":
            return (np.asarray(self.spectrum, float), 1.0,
                    "Using raw spectrum.", "raw")

        if mode == "expanded":
            return (np.asarray(self.degenerate_spectrum, float), 1.0,
                    "Using expanded (degenerate) spectrum.", "expanded")

        if mode == "directional":
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum. Run compute_spectrum() first.")
            return (np.asarray(self.spectrum, float), 0.5,
                    "Directional: Weyl × 1/2.", "directional")

        # auto
        if self._is_pure_directional():
            return (np.asarray(self.spectrum, float), 0.5,
                    "Pure directional: raw spectrum; Weyl × 1/2.", "directional")
        else:
            if hasattr(self, "degenerate_spectrum"):
                return (np.asarray(self.degenerate_spectrum, float), 1.0,
                        "Non-directional: expanded spectrum.", "expanded")
            return (np.asarray(self.spectrum, float), 1.0,
                    "No degeneracy info: raw spectrum.", "raw")

    # =========================================================================
    # Weyl law
    # =========================================================================

    def N_weyl(self, k):
        """
        Non-dispersive Weyl staircase:  N(k) = neff * L / π * k.
        """
        n_eff = self.neff(k) if callable(self.neff) else self.neff
        return n_eff * self.total_length / np.pi * k

    def N_weyl_scaled(self, k, length_scale=1.0, k0=None):
        """
        Dispersive Weyl staircase using the group index ng(k), with optional
        length scaling and anchoring.

        Parameters
        ----------
        k : float or array-like
        length_scale : float
            Multiply total_length by this factor (0.5 for directional graphs).
        k0 : float or None
            If provided, shift so that N_weyl_scaled(k0) = 0.

        Returns
        -------
        float or ndarray
        """
        L_eff = float(self.total_length) * float(length_scale)

        def eval_ng(kk):
            if callable(self.ng):
                return np.asarray(self.ng(kk), dtype=float)
            return np.full_like(np.asarray(kk, dtype=float), float(self.ng))

        def _integral(k_val):
            """∫_0^k ng(κ) dκ  (scalar)."""
            if callable(self.ng):
                Ng = 512
                ks = np.linspace(0.0, float(k_val), Ng)
                return np.trapezoid(eval_ng(ks), ks)
            return float(self.ng) * float(k_val)

        if np.isscalar(k):
            N = (L_eff / np.pi) * _integral(k)
            if k0 is not None:
                N -= (L_eff / np.pi) * _integral(k0)
            return N

        k_arr  = np.asarray(k, dtype=float)
        k_flat = k_arr.ravel()

        ks_full   = np.concatenate([k_flat, np.array([0.0])])
        ks_sorted = np.unique(ks_full)
        ng_sorted = eval_ng(ks_sorted)
        dks       = np.diff(ks_sorted)
        mids      = 0.5 * (ng_sorted[1:] + ng_sorted[:-1])
        F_sorted  = np.concatenate([[0.0], np.cumsum(mids * dks)])

        zero_idx = np.searchsorted(ks_sorted, 0.0)
        F_sorted -= F_sorted[zero_idx]

        idx    = np.searchsorted(ks_sorted, k_flat)
        F_at_k = F_sorted[idx]
        N_weyl = (L_eff / np.pi) * F_at_k

        if k0 is not None:
            F0 = (L_eff / np.pi) * _integral(k0)
            N_weyl -= F0

        return N_weyl.reshape(k_arr.shape)

    # =========================================================================
    # Unfolding
    # =========================================================================

    def _compute_unfolded_spectrum(self, spectrum=None, mode=None):
        """
        Unfold a spectrum using the Weyl law.

        Parameters
        ----------
        spectrum : ndarray or None
        mode : str or None

        Returns
        -------
        ndarray  — unfolded values N_weyl_scaled(k)
        """
        if spectrum is None:
            chosen_mode  = mode or getattr(self, "stats_mode", "auto")
            spec, weyl_scale, _, _ = self._stats_selection(mode=chosen_mode)
            spectrum = np.asarray(spec, dtype=float)
        else:
            spectrum    = np.asarray(spectrum, dtype=float)
            chosen_mode = mode or getattr(self, "stats_mode", "auto")
            _, weyl_scale, _, _ = self._stats_selection(mode=chosen_mode)

        spectrum = np.sort(np.real(spectrum))
        return self.N_weyl_scaled(spectrum, length_scale=weyl_scale)

    # =========================================================================
    # Σ² and Δ₃
    # =========================================================================

    def _compute_sigma2(self, eigvals, dE, L_max):
        """
        Number variance Σ²(L).

        Parameters
        ----------
        eigvals : ndarray   unfolded spectrum (sorted)
        dE : float          grid spacing
        L_max : float       maximum L

        Returns
        -------
        L_vals, sigma2_vals : ndarray
        """
        E0, Emax = eigvals[0], eigvals[-1]
        NR       = int(np.floor((Emax - E0) / dE))
        grid     = E0 + np.arange(1, NR + 1) * dE
        Ngrid    = np.searchsorted(eigvals, grid, side="left").astype(float)

        IR_max      = min(NR - 1, int(np.floor(L_max / dE)))
        L_vals      = np.arange(0, IR_max + 1) * dE
        sigma2_vals = np.empty_like(L_vals)

        for IR in range(1, IR_max + 1):
            counts         = Ngrid[IR:] - Ngrid[:-IR]
            L              = IR * dE
            sigma2_vals[IR] = np.mean((counts - L) ** 2)

        sigma2_vals[0] = 0.0
        return L_vals, sigma2_vals

    def _compute_delta3(self, eigvals, dE, L_max):
        """
        Spectral rigidity Δ₃(L).

        Parameters
        ----------
        eigvals : ndarray   unfolded spectrum (sorted)
        dE : float          grid spacing
        L_max : float       maximum window size

        Returns
        -------
        L_vals, delta3_vals : ndarray
        """
        NR     = int(np.floor((eigvals[-1] - eigvals[0]) / dE))
        IR_max = min(NR - 1, int(np.floor(L_max / dE)))
        L_vals      = np.arange(1, IR_max + 1) * dE
        delta3_vals = np.zeros(len(L_vals))

        for i, L in enumerate(L_vals):
            delta3 = []
            for j in range(len(eigvals)):
                if eigvals[j] + L > eigvals[-1]:
                    continue
                sub    = eigvals[(eigvals > eigvals[j]) & (eigvals <= eigvals[j] + L)]
                sub   -= eigvals[j] + L / 2
                count  = len(sub)

                sum1   = -1 / L**2 * np.sum(sub)**2
                sum2   =  3 * count / (2 * L**2) * np.sum(sub**2)
                sum3   = -3 / L**4 * np.sum(sub**2)**2
                int_arr = np.flip(np.arange(-count + 1, count + 1, step=2))
                sum4   =  1 / L * np.sum(int_arr * sub)
                delta3.append(count**2 / 16 + sum1 + sum2 + sum3 + sum4)

            delta3_vals[i] = np.mean(delta3)

        L_vals      = np.insert(L_vals, 0, 0.0)
        delta3_vals = np.insert(delta3_vals, 0, 0.0)
        return L_vals, delta3_vals
