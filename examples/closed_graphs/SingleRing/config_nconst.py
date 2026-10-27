#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SingleRing (closed) — constant neff
=====================================

Geometry
--------
One physical 2-port vertex (Neumann/Kirchhoff BC) with one self-loop:

    vertex 0:
        port 0 ↔ port 1  →  ring of circumference L = 2π R

The Neumann vertex matrix is:
    σ = [[0, 1],
         [1, 0]]

which enforces perfect transmission (no reflection): a wave entering
port 0 exits port 1 and vice versa.

Because the graph has only a single bond and a single vertex with perfect
transmission, the spectrum consists of equally spaced eigenvalues
k_n = 2π n / (neff * L), and all spectral statistics are Poissonian.

Analysis pipeline:
    1. Graph topology plot
    2. Spectrum computation
    3. Zeta-B plot
    4. Wavefunction for a few modes
    5. IPR (= 1: one bond)
    6. Spectral statistics: staircase, NNSD, Σ²/Δ₃
    7. Length spectrum

Each section is guarded by ``if True / False`` — flip to enable/disable.
See also: config_nfunc.py, ../DoubleRing/config_nconst.py
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

from quantum_graphs.closed_graphs import ClosedGraphs


# ============================================================================
#  Ring geometry
# ============================================================================

R  = 100.0                   # ring radius (µm)
L  = 2.0 * np.pi * R        # circumference (µm)

print(f"Ring geometry: R = {R:.2f} µm  →  L = {L:.3f} µm")

# One real vertex at the origin.
# port 0 ↔ port 1 : single self-loop
vertex_positions = [(0.0, 0.0)]

port_map = [
    [(0, 1),   # port 0 → port 1
     (0, 0)],  # port 1 → port 0
]

lengths = {
    ((0, 0), (0, 1)): L,
}


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 2.41
wavelength_range = (1.48, 1.64)   # µm


# ============================================================================
#  Vertex scattering matrix (Neumann/Kirchhoff BC)
# ============================================================================

# 2-port Neumann vertex: S = 2/d * J - I_d  with d=2  →  off-diagonal ones
S_vertex = np.array([[0.0, 1.0],
                     [1.0, 0.0]], dtype=complex)

vertex_matrices = {0: S_vertex}


# ============================================================================
#  Instantiate ClosedGraphs
# ============================================================================

graph = ClosedGraphs(
    vertex_positions = vertex_positions,
    port_map         = port_map,
    lengths          = lengths,
    vertex_matrices  = vertex_matrices,
    neff             = neff,
    wavelength_range = wavelength_range,
    name             = f"SingleRing_nconst_R={R:.0f}_neff{neff:.3f}",
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
#  5. IPR
# ============================================================================

if True:
    # one physical bond: IPR = 1 for every mode, and the normalised entropy
    # S / ln N_b is undefined (plot_entropy would show NaN)
    graph.plot_ipr(wavelength_range=None, save_fig=True, file_format="pdf")


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
