#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralStatsMixin — Weyl law, unfolding, NNSD, Σ², Δ₃.
Module-level: NNSD_func (also re-exported from open_graphs.__init__).
"""

import numpy as np

from ..stats.rmt import nnsd as _nnsd
from ..stats.spectral import number_variance, spectral_rigidity, weyl_staircase


# ---------------------------------------------------------------------------
# Module-level NNSD function
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


    def N_weyl_scaled(self, k, length_scale: float = 1.0, k0: float = 0.0):
        """Dispersive Weyl staircase N(k; k0) = (L_eff/π) ∫_{k0}^{k} ng(κ) dκ."""
        return weyl_staircase(k, self.total_length, self.ng, length_scale, k0)


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
        """Number variance Σ²(L) of an unfolded spectrum (quantum_graphs.stats.number_variance)."""
        return number_variance(eigvals, dE, L_max)

    def _compute_delta3(self, eigvals, dE, L_max, windows="level"):
        """Spectral rigidity Δ₃(L) of an unfolded spectrum (quantum_graphs.stats.spectral_rigidity)."""
        return spectral_rigidity(eigvals, dE, L_max, windows=windows)
