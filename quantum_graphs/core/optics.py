"""Wavelength / wavenumber conversions and refractive-index helpers."""

import numpy as np


def wavelength_to_k(lam_um):
    """Vacuum wavenumber k = 2π/λ (µm⁻¹) from the vacuum wavelength (µm)."""
    return 2 * np.pi / np.asarray(lam_um, dtype=float)


def k_to_wavelength(k):
    """Vacuum wavelength λ = 2π/k (µm) from the wavenumber (µm⁻¹)."""
    return 2 * np.pi / np.asarray(k, dtype=float)


def group_index_function(neff):
    """Group index n_g(k) = n_eff(k) + k dn_eff/dk.

    Returns a float for a constant index, otherwise a callable that
    differentiates ``neff`` numerically (numdifftools, adaptive step).
    """
    if not callable(neff):
        return float(neff)
    import numdifftools as nd
    derivative = nd.Derivative(neff)

    def ng_func(k):
        return neff(k) + k * derivative(k)

    ng_func.__name__ = f"ng_from_{getattr(neff, '__name__', 'neff')}"
    return ng_func
