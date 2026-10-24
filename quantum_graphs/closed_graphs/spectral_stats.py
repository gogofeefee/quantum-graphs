#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralStatsMixin — Weyl law, unfolding, NNSD, Σ², Δ₃.
Module-level: NNSD_func (also re-exported from closed_graphs.__init__).
"""

import numpy as np

from ..stats.rmt import nnsd as _nnsd
from ..stats.spectral import number_variance, spectral_rigidity, weyl_staircase


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def NNSD_func(s, wigner="GOE"):
    """Nearest-neighbour spacing distribution: 'poissonian', 'GOE', 'GUE' or 'GSE'
    (see quantum_graphs.stats.nnsd)."""
    return _nnsd(s, wigner)


def _get_rmt_funcs():
    """Return the (sigma2, delta3) RMT reference functions, signature f(L, wigner)."""
    from ..stats.rmt import sigma2_rmt, delta3_rmt
    return sigma2_rmt, delta3_rmt


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
        """Dispersive Weyl staircase N(k) = (L_eff/π) ∫_{k0}^{k} ng dκ, L_eff = length_scale × L.

        ``k0=None`` integrates from 0 (see quantum_graphs.stats.weyl_staircase).
        """
        return weyl_staircase(k, self.total_length, self.ng, length_scale, 0.0 if k0 is None else k0)

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
        """Number variance Σ²(L) of an unfolded spectrum (quantum_graphs.stats.number_variance)."""
        return number_variance(eigvals, dE, L_max)

    def _compute_delta3(self, eigvals, dE, L_max, windows="level"):
        """Spectral rigidity Δ₃(L) of an unfolded spectrum (quantum_graphs.stats.spectral_rigidity)."""
        return spectral_rigidity(eigvals, dE, L_max, windows=windows)
