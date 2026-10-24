#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralStatsMixin — Weyl law, unfolding, NNSD, Σ², Δ₃.
Module-level: NNSD_func (also re-exported from open_graphs.__init__).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Module-level NNSD function
# ---------------------------------------------------------------------------

def NNSD_func(s, wigner="GOE"):
    """
    Nearest-Neighbor Spacing Distribution based on the Wigner surmise.

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
        raise ValueError("Invalid wigner type. Choose from 'poissonian', 'GOE', 'GUE', or 'GSE'.")


def _get_rmt_funcs():
    """Return the (sigma2, delta3) RMT reference functions, signature f(L, wigner)."""
    from ..stats.rmt import sigma2_rmt, delta3_rmt
    return sigma2_rmt, delta3_rmt


class SpectralStatsMixin:

    @property
    def best_spectrum(self):
        """
        Return the best available resonance spectrum as a numpy array.

        Priority: complex numerical spectrum (``self.spectrum``) first,
        then experimental peaks (``self.exp_spectrum``), then ``None``.
        """
        if hasattr(self, "spectrum") and np.asarray(self.spectrum).size > 0:
            return self.spectrum
        if hasattr(self, "exp_spectrum") and np.asarray(self.exp_spectrum).size > 0:
            return self.exp_spectrum
        return None

    def _is_pure_directional(self):
        """Return True iff every level has deg == 2."""
        return (hasattr(self, "degeneracies") and
                self.degeneracies.size > 0 and
                np.all(self.degeneracies == 2))

    def _stats_selection(self, mode="auto", use_experimental=False):
        """
        Choose which spectrum and Weyl scaling to use.

        Returns (spectrum_used, weyl_scale, note, method).
        """
        if use_experimental:
            if not hasattr(self, "exp_spectrum"):
                raise AttributeError(
                    "use_experimental=True but exp_spectrum is missing. "
                    "Run find_experimental_resonances() first."
                )
            sp = np.asarray(self.exp_spectrum, float)
            if sp.size == 0:
                raise ValueError("exp_spectrum is empty.")
            if mode == "raw":
                return sp, 1.0, "Experimental raw spectrum (no duplication).", "raw"
            if mode == "expanded":
                return sp, 1.0, "Experimental: expanded = raw (no degeneracy info).", "expanded"
            if mode == "directional":
                return sp, 0.5, "Experimental directional assumption: Weyl × 1/2.", "directional"
            return sp, 1.0, "Experimental (auto): using expanded=raw (no degeneracy info).", "expanded"

        if mode == "raw":
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return np.real(np.asarray(self.spectrum)), 1.0, "Using raw spectrum (no duplication).", "raw"

        if mode == "expanded":
            if hasattr(self, "degenerate_spectrum"):
                return (np.real(np.asarray(self.degenerate_spectrum)),
                        1.0, "Using expanded (degenerate) spectrum.", "expanded")
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return (np.real(np.asarray(self.spectrum)),
                    1.0, "No degeneracy info: expanded=raw spectrum.", "expanded")

        if mode == "directional":
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return (np.real(np.asarray(self.spectrum)),
                    0.5, "Directional assumption: raw spectrum; Weyl × 1/2.", "directional")

        # AUTO
        if self._is_pure_directional():
            return (np.real(np.asarray(self.spectrum)),
                    0.5, "Detected pure directional case: raw spectrum; Weyl × 1/2.", "directional")
        if hasattr(self, "degenerate_spectrum"):
            return (np.real(np.asarray(self.degenerate_spectrum)),
                    1.0, "Non-directional: using expanded spectrum.", "expanded")
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum found. Run find_resonances() first.")
        return (np.real(np.asarray(self.spectrum)),
                1.0, "No degeneracy info: using raw spectrum.", "raw")

    # -------------------------------------------------------------------------
    # Weyl law
    # -------------------------------------------------------------------------

    def N_weyl_scaled2(self, k, length_scale=1.0):
        """Weyl staircase (non-dispersive) with optional length scale."""
        if callable(self.neff):
            n_eff = self.neff(np.real(k))
        else:
            n_eff = self.neff
        return n_eff * (self.total_length * length_scale) / np.pi * np.real(k)

    def N_weyl_scaled(self, k, length_scale: float = 1.0, k0: float = 0.0):
        """
        Dispersive Weyl staircase:
            N_Weyl(k; k0) = (L_eff / π) ∫_{k0}^{k} ng(κ) dκ
        """
        if np.isscalar(k):
            k_real = float(np.real(k))
        else:
            k_real = np.asarray(k, dtype=float).real

        k0_real = float(np.real(k0))
        L_eff   = float(self.total_length) * float(length_scale)

        def eval_ng(kk):
            if callable(self.ng):
                return np.asarray(self.ng(kk), dtype=float)
            return np.full_like(np.asarray(kk, dtype=float), float(self.ng), dtype=float)

        if np.isscalar(k_real):
            if callable(self.ng):
                ks       = np.linspace(k0_real, k_real, 512)
                integral = np.trapezoid(eval_ng(ks), ks)
            else:
                integral = float(self.ng) * (k_real - k0_real)
            return (L_eff / np.pi) * integral

        k_arr  = np.asarray(k_real, dtype=float)
        k_flat = k_arr.ravel()

        if callable(self.ng):
            ks_full   = np.concatenate([k_flat, np.array([k0_real], dtype=float)])
            ks_sorted = np.unique(ks_full)
            ng_sorted = eval_ng(ks_sorted)
            dks       = np.diff(ks_sorted)
            mids      = 0.5 * (ng_sorted[1:] + ng_sorted[:-1])
            F_sorted  = np.concatenate([[0.0], np.cumsum(mids * dks)])
            k0_idx    = np.searchsorted(ks_sorted, k0_real)
            F_sorted -= F_sorted[k0_idx]
            idx       = np.searchsorted(ks_sorted, k_flat)
            F_at_k    = F_sorted[idx]
            return ((L_eff / np.pi) * F_at_k).reshape(k_arr.shape)

        return (L_eff / np.pi) * float(self.ng) * (k_arr - k0_real)

    def rho_weyl_scaled(self, k, length_scale: float = 1.0):
        """Dispersive Weyl density: ρ_Weyl(k) = (L_eff / π) ng(k)."""
        k_real = float(np.real(k)) if np.isscalar(k) else np.asarray(k, dtype=float).real
        L_eff  = float(self.total_length) * float(length_scale)
        ng_val = self.ng(k_real) if callable(self.ng) else float(self.ng)
        return (L_eff / np.pi) * np.asarray(ng_val, dtype=float)

    # -------------------------------------------------------------------------
    # Unfolding
    # -------------------------------------------------------------------------

    def _compute_unfolded_spectrum(self, spectrum=None, mode=None, use_experimental=False):
        """Unfold spectrum via Weyl law anchored at k0 = spectrum[0]."""
        chosen_mode = mode or getattr(self, "stats_mode", "auto")

        if spectrum is None:
            spectrum_used, weyl_scale, _, _ = self._stats_selection(
                mode=chosen_mode, use_experimental=use_experimental
            )
            spectrum = np.asarray(spectrum_used, dtype=float)
        else:
            spectrum = np.asarray(spectrum, dtype=float)
            _, weyl_scale, _, _ = self._stats_selection(
                mode=chosen_mode, use_experimental=use_experimental
            )

        spectrum = np.sort(np.real(spectrum))
        if spectrum.size == 0:
            return spectrum

        k0 = float(spectrum[0])
        return self.N_weyl_scaled(spectrum, length_scale=weyl_scale, k0=k0)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_sigma2(self, eigvals, dE, L_max):
        """Compute number variance Σ²(L)."""
        E0    = eigvals[0]
        Emax  = eigvals[-1]
        NR    = int(np.floor((Emax - E0) / dE))
        grid  = E0 + np.arange(1, NR + 1) * dE
        Ngrid = np.searchsorted(eigvals, grid, side='left').astype(float)

        IR_max      = min(NR - 1, int(np.floor(L_max / dE)))
        L_vals      = np.arange(0, IR_max + 1) * dE
        sigma2_vals = np.empty_like(L_vals)

        for IR in range(1, IR_max + 1):
            counts          = Ngrid[IR:] - Ngrid[:-IR]
            sigma2_vals[IR] = np.mean((counts - IR * dE) ** 2)

        sigma2_vals[0] = 0.0
        return L_vals, sigma2_vals

    def _compute_delta3(self, eigvals, dE, L_max):
        """Compute spectral rigidity Δ₃(L)."""
        NR     = int(np.floor((eigvals[-1] - eigvals[0]) / dE))
        IR_max = min(NR - 1, int(np.floor(L_max / dE)))
        L_vals      = np.arange(1, IR_max + 1) * dE
        delta3_vals = np.zeros(len(L_vals))

        for i, L in enumerate(L_vals):
            delta3 = []
            for j in range(len(eigvals)):
                if eigvals[j] + L > eigvals[-1]:
                    continue
                sub   = eigvals[(eigvals > eigvals[j]) & (eigvals <= eigvals[j] + L)]
                sub  -= eigvals[j] + L / 2
                count = len(sub)
                sum1  = -1 / L**2 * np.sum(sub)**2
                sum2  = 3 * count / (2 * L**2) * np.sum(sub**2)
                sum3  = -3 / L**4 * np.sum(sub**2)**2
                int_array = np.flip(np.arange(-count + 1, count + 1, step=2))
                sum4  = 1 / L * np.sum(int_array * sub)
                total = count**2 / 16 + sum1 + sum2 + sum3 + sum4
                delta3.append(total)
            delta3_vals[i] = np.mean(delta3)

        L_vals      = np.insert(L_vals,      0, 0.0)
        delta3_vals = np.insert(delta3_vals, 0, 0.0)
        return L_vals, delta3_vals
