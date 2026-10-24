#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpb_dispersion.py — MPB dispersion helpers shared by the config_nfunc examples.

Functions
---------
load_mpb_indices(neff_npz, ng_npz)
    Load neff(k) and ng(k) from MPB .npz files and build cubic interpolation callables.

plot_mpb_indices_vs_k(k_neff, neff_raw, neff_func, k_ng, ng_raw, ng_func)
    Two-panel figure: neff(k) top, ng(k) bottom.

apply_index_scaling(neff_in, ng_in, scale)
    Multiply neff and ng (constants or callables) by a global scale factor.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_mpb_indices(neff_npz_path: str, ng_npz_path: str):
    """
    Load neff(k_vac) and ng(k_vac) from MPB .npz files.

    Expected array keys
    -------------------
    neff file : ``k_vac``, ``neff``
    ng file   : ``k_vac``, ``ng``

    Returns
    -------
    k_neff, neff_raw, neff_func, k_ng, ng_raw, ng_func
    """
    for p in (neff_npz_path, ng_npz_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"MPB npz not found: {p}")

    neff_data = np.load(neff_npz_path)
    if "k_vac" not in neff_data or "neff" not in neff_data:
        raise KeyError('neff npz must contain keys "k_vac" and "neff".')

    ng_data = np.load(ng_npz_path)
    if "k_vac" not in ng_data or "ng" not in ng_data:
        raise KeyError('ng npz must contain keys "k_vac" and "ng".')

    def _clean_interp(k, y, name):
        k = np.asarray(k, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if k.size != y.size:
            raise ValueError(f"{name}: k and y must have the same length.")
        mask = np.isfinite(k) & np.isfinite(y)
        k, y = k[mask], y[mask]
        if k.size == 0:
            raise ValueError(f"{name}: no finite samples after cleaning.")
        order = np.argsort(k)
        k, y = k[order], y[order]
        k_u, idx = np.unique(k, return_index=True)
        y_u = y[idx]
        kind = "cubic" if k_u.size >= 4 else "linear"
        f = interp1d(k_u, y_u, kind=kind, fill_value="extrapolate")
        return k_u, y_u, f

    k_neff, neff_raw, neff_func = _clean_interp(
        neff_data["k_vac"], neff_data["neff"], "neff")
    k_ng, ng_raw, ng_func = _clean_interp(
        ng_data["k_vac"], ng_data["ng"], "ng")

    print("✓ Loaded MPB indices: neff(k) and ng(k).")
    return k_neff, neff_raw, neff_func, k_ng, ng_raw, ng_func


def plot_mpb_indices_vs_k(k_neff, neff_raw, neff_func,
                           k_ng, ng_raw, ng_func,
                           n_plot=10_000, show=True):
    """
    Two-panel figure: neff(k) on top, ng(k) on bottom, shared k-axis.
    """
    k_min = float(min(np.min(k_neff), np.min(k_ng)))
    k_max = float(max(np.max(k_neff), np.max(k_ng)))
    k_plot = np.linspace(k_min, k_max, int(n_plot))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    ax1.plot(k_neff, neff_raw, "o", color="gray", alpha=0.5, label="MPB data")
    ax1.plot(k_plot, neff_func(k_plot), "-", color="black", lw=2, label="Interpolated")
    ax1.set_ylabel(r"$n_\mathrm{eff}(k)$", fontsize=22)

    ax2.plot(k_ng, ng_raw, "o", color="gray", alpha=0.5, label="MPB data")
    ax2.plot(k_plot, ng_func(k_plot), "-", color="black", lw=2, label="Interpolated")
    ax2.set_xlabel(r"$k$ (µm$^{-1}$)", fontsize=22)
    ax2.set_ylabel(r"$n_g(k)$", fontsize=22)

    for ax in (ax1, ax2):
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in",
                       length=6, width=2, labelsize=20, top=True, right=True)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=3, width=1.5, top=True, right=True)
        ax.legend(fontsize=18, frameon=False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


def apply_index_scaling(neff_in, ng_in, scale: float = 1.0):
    """
    Return (neff_scaled, ng_scaled) with both multiplied by ``scale``.
    """
    scale = float(scale)

    def _wrap(f):
        def g(k):
            return scale * f(k)
        g.__name__ = getattr(f, "__name__", "index_func") + f"_x{scale:.6f}"
        return g

    neff_out = _wrap(neff_in) if callable(neff_in) else scale * float(neff_in)
    ng_out   = _wrap(ng_in)   if callable(ng_in)   else scale * float(ng_in)
    return neff_out, ng_out
