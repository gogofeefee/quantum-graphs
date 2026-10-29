#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring (closed) — constant neff
==============================

Geometry
--------
One physical 4-port vertex (directional coupler) with two self-loops,
forming a figure-eight (double-ring) topology:

    vertex 0:
        port 0 ↔ port 2  →  ring of circumference L1 = 2π R1
        port 1 ↔ port 3  →  ring of circumference L2 = 2π R2

The ratio R1/R2 = √2 is irrational, breaking the arithmetic degeneracy
and giving generic (GUE) spectral statistics for a time-reversal-broken graph.

Analysis pipeline:
    1. Graph topology plot
    2. Spectrum computation
    3. Zeta-B plot
    4. Wavefunction for a few modes
    5. Entropy and IPR
    6. Spectral statistics: staircase, NNSD, Σ²/Δ₃
    7. Length spectrum

Each section is guarded by ``if True / False`` — flip to enable/disable.
See also: config_nfunc.py
"""

# ============================================================================
#  Paths & imports
# ============================================================================

import os
import sys
import numpy as np

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _argv0 = sys.argv[0] if sys.argv else ""
    _here  = (os.path.dirname(os.path.abspath(_argv0))
              if os.path.isfile(_argv0) else os.getcwd())


# Use the installed package (pip install -e .) or, failing that, this checkout.
try:
    import quantum_graphs  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..", "..")))

from quantum_graphs.closed_graphs import DirectionalGraphs


# ============================================================================
#  Ring geometry
# ============================================================================

R1 = 100.0                     # radius of ring 1 (µm)
R2 = R1 / np.sqrt(2)           # radius of ring 2 (irrational ratio)
L1 = 2.0 * np.pi * R1         # circumference of ring 1 (µm)
L2 = 2.0 * np.pi * R2         # circumference of ring 2 (µm)

print(f"Ring geometry: R1 = {R1:.2f} µm  →  L1 = {L1:.3f} µm")
print(f"               R2 = {R2:.4f} µm  →  L2 = {L2:.3f} µm")

# One real vertex at the origin.
# port 0 ↔ port 2 : first self-loop (ring 1)
# port 1 ↔ port 3 : second self-loop (ring 2)
vertex_positions = [(0.0, 0.0)]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 0, 2, L1),
    (0, 1, 0, 3, L2),
]


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 2.41
wavelength_range = (1.48, 1.64)   # µm
C                = 0.5            # coupling constant


# ============================================================================
#  Instantiate DirectionalGraphs
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions = vertex_positions,
    bonds            = bonds,
    vertex_couplings = C,
    neff             = neff,
    wavelength_range = wavelength_range,
    name             = f"DoubleRing_nconst_R1={R1:.0f}_R2={R2:.2f}_neff{neff:.3f}_C{C:.2f}",
    base_dir         = _here,
)

print(f"\nGraph summary:")
print(f"  Vertices     : {graph.n_vertices}")
print(f"  Bonds        : {graph.n_bonds}")
print(f"  Total length : {graph.total_length:.3f} µm")
print(f"  Save path    : {graph.save_path}\n")


# ============================================================================
#  1. Topology plot
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True,  save_fig=True, file_format="pdf")
    graph.plot_graph(show_lengths=False, save_fig=True, file_format="pdf")


# ============================================================================
#  2. Spectrum computation
# ============================================================================

if True:
    spectrum, degeneracies = graph.compute_spectrum(
        N_scan        = 1_000_000,
        tol           = 1e-16,
        show_progress = True,
        save_spectrum = True,
        overwrite     = False,
        maxiter       = 500,
        nullity_tol   = 1e-4,
    )
    print(f"\nSpectrum: {len(spectrum)} eigenvalues found.")
    for idx, (k, d) in enumerate(zip(spectrum[:10], degeneracies[:10])):
        print(f"  {idx:4d}  k = {k:.10f}  deg = {d}")


# ============================================================================
#  3. Zeta-B
# ============================================================================

if True:
    graph.plot_zeta_B(
        wavelength_range = None,
        N_plot   = 500_000,
        spectrum = None,
        save_fig = True,
        file_format = "pdf",
    )
    xlim_zoom = (1.51, 1.53)
    graph.plot_zeta_B(
        wavelength_range = xlim_zoom,
        N_plot   = 100_000,
        spectrum = spectrum,
        save_fig = True,
        file_format = "pdf",
        xlim = xlim_zoom,
    )


# ============================================================================
#  4. Wavefunction
# ============================================================================

if True:
    for mode_idx in [0, 5]:
        a_in = graph.compute_wavefunction(spectrum[mode_idx], tol=1e-4)
        graph.plot_wavefunction(a_in=a_in, k=spectrum[mode_idx],
                                save_fig=True, file_format="pdf")


# ============================================================================
#  5. Entropy and IPR
# ============================================================================

if True:
    graph.plot_entropy(wavelength_range=None, save_fig=True, file_format="pdf")
    graph.plot_ipr(wavelength_range=None,     save_fig=True, file_format="pdf")


# ============================================================================
#  6. Spectral statistics
# ============================================================================

if True:
    wavelength_range_stats = None

    graph.plot_staircase_function(
        plot_weyl              = True,
        weyl_diff              = True,
        mode                   = None,
        wavelength_range_stats = wavelength_range_stats,
        save_fig               = True,
        file_format            = "pdf",
    )
    graph.short_range_correlations(
        width                  = 0.1,
        xlim                   = (0, 3),
        plot_wigner            = ["goe", "gue", "poissonian"],
        color_list             = ["r-", "g--", "b-."],
        cumulative_spacing     = True,
        mode                   = None,
        wavelength_range_stats = wavelength_range_stats,
        yscale                 = "linear",
        save_fig               = True,
        file_format            = "pdf",
    )
    graph.long_range_correlations(
        L_max                  = 10,
        L_step                 = 0.1,
        plot_wigner            = ["goe", "gue", "poissonian"],
        color_list             = ["r-", "g--", "b-."],
        correlation_type       = "both",
        mode                   = None,
        wavelength_range_stats = wavelength_range_stats,
        save_fig               = True,
        file_format            = "pdf",
    )


# ============================================================================
#  7. Length spectrum
# ============================================================================

if True:
    graph.plot_length_spectrum(
        L_max    = 3000,
        n_points = 100_000,
        save_fig = True,
        file_format = "pdf",
    )
