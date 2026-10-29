#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SingleRing (closed) — dispersive neff(k)
==========================================

Same single-ring topology as config_nconst.py but with neff(k) and ng(k)
loaded from MPB simulation data.

Geometry
--------
One physical 2-port vertex (Neumann/Kirchhoff BC) with one self-loop:

    vertex 0:
        port 0 ↔ port 1  →  ring of circumference L = 2π R

Analysis pipeline:
    1. Dispersion plot
    2. Graph topology plot
    3. Spectrum computation
    4. Zeta-B plot
    5. Wavefunction for a few modes
    6. IPR (= 1: one bond)
    7. Spectral statistics: staircase, NNSD, Σ²/Δ₃
    8. Length spectrum

Each section is guarded by ``if True / False`` — flip to enable/disable.
See also: config_nconst.py, ../DoubleRing/config_nfunc.py
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
from quantum_graphs.dispersion import load_mpb_indices, plot_mpb_indices_vs_k, apply_index_scaling


# ============================================================================
#  Ring geometry
# ============================================================================

R  = 100.0                   # ring radius (µm)
L  = 2.0 * np.pi * R        # circumference (µm)

print(f"Ring geometry: R = {R:.2f} µm  →  L = {L:.3f} µm")

# One real vertex at the origin.
# port 0 ↔ port 1 : single self-loop
vertex_positions = [(0.0, 0.0)]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um)
bonds = [
    (0, 0, 0, 1, L),
]


# ============================================================================
#  Vertex scattering matrix (Neumann/Kirchhoff BC)
# ============================================================================

# 2-port Neumann vertex: S = 2/d * J - I_d  with d=2  →  off-diagonal ones
S_vertex = np.array([[0.0, 1.0],
                     [1.0, 0.0]], dtype=complex)

vertex_matrices = {0: S_vertex}


# ============================================================================
#  MPB dispersion data
# ============================================================================

_mpb_dir = os.path.abspath(os.path.join(
    _here, "..", "..", "..", "data", "mpb", "shallow_etched_si_on_sio2_w0.40_h0.22_rib0.08"))
neff_npz = os.path.join(_mpb_dir, "neff_vs_kvac_band1_full.npz")
ng_npz   = os.path.join(_mpb_dir, "group_index_vs_kvac_band1_full.npz")

k_neff, neff_raw, neff_func, k_ng, ng_raw, ng_func = load_mpb_indices(neff_npz, ng_npz)


# ============================================================================
#  1. Dispersion plot
# ============================================================================

if True:
    plot_mpb_indices_vs_k(k_neff, neff_raw, neff_func,
                           k_ng,   ng_raw,   ng_func,
                           n_plot=10_000, show=True)


# ============================================================================
#  Optical parameters
# ============================================================================

index_multiplier = 1.0
neff, ng = apply_index_scaling(neff_func, ng_func, scale=index_multiplier)
if index_multiplier != 1.0:
    print(f"✓ Index multiplier: ×{index_multiplier:.6f}")

wavelength_range = (1.48, 1.64)


# ============================================================================
#  Instantiate ClosedGraphs
# ============================================================================

graph = ClosedGraphs(
    vertex_positions = vertex_positions,
    bonds            = bonds,
    vertex_matrices  = vertex_matrices,
    neff             = neff,
    ng               = ng,
    wavelength_range = wavelength_range,
    name = (
        f"SingleRing_nfunc_R={R:.0f}"
        + (f"_nx{index_multiplier:.5f}" if index_multiplier != 1.0 else "")
    ),
    base_dir = _here,
)

print(f"\nGraph summary:")
print(f"  Vertices     : {graph.n_vertices}")
print(f"  Bonds        : {graph.n_bonds}")
print(f"  Total length : {graph.total_length:.3f} µm")
print(f"  Save path    : {graph.save_path}\n")


# ============================================================================
#  2. Topology plot
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True,  save_fig=True, file_format="pdf")
    graph.plot_graph(show_lengths=False, save_fig=True, file_format="pdf")


# ============================================================================
#  3. Spectrum
# ============================================================================

if True:
    spectrum, degeneracies = graph.compute_spectrum(
        show_progress = True,
        save_spectrum = True,
        overwrite     = False,
    )
    print(f"\nSpectrum: {len(spectrum)} eigenvalues found.")


# ============================================================================
#  4. Zeta-B
# ============================================================================

if True:
    graph.plot_zeta_B(wavelength_range=None, N_plot=500_000, spectrum=None,
                      save_fig=True, file_format="pdf")
    xlim_zoom = (1.51, 1.53)
    graph.plot_zeta_B(wavelength_range=xlim_zoom, N_plot=100_000, spectrum=spectrum,
                      save_fig=True, file_format="pdf", xlim=xlim_zoom)


# ============================================================================
#  5. Wavefunction
# ============================================================================

if True:
    for mode_idx in [0, 5]:
        a_in = graph.compute_wavefunction(spectrum[mode_idx], tol=1e-4)
        graph.plot_wavefunction(a_in=a_in, k=spectrum[mode_idx],
                                save_fig=True, file_format="pdf")


# ============================================================================
#  6. IPR
# ============================================================================

if True:
    # one physical bond: IPR = 1 for every mode, and the normalised entropy
    # S / ln N_b is undefined (plot_entropy would show NaN)
    graph.plot_ipr(wavelength_range=None, save_fig=True, file_format="pdf")


# ============================================================================
#  7. Spectral statistics
# ============================================================================

if True:
    wavelength_range_stats = None

    graph.plot_staircase_function(
        plot_weyl              = True,
        weyl_diff              = True,
        wavelength_range       = wavelength_range_stats,
        save_fig               = True,
        file_format            = "pdf",
    )
    graph.short_range_correlations(
        width                  = 0.1,
        xlim                   = (0, 3),
        plot_wigner            = ["goe", "gue", "poissonian"],
        color_list             = ["r-", "g--", "b-."],
        cumulative_spacing     = True,
        wavelength_range       = wavelength_range_stats,
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
        wavelength_range       = wavelength_range_stats,
        save_fig               = True,
        file_format            = "pdf",
    )


# ============================================================================
#  8. Length spectrum
# ============================================================================

if True:
    graph.plot_length_spectrum(
        L_max    = 3000,
        n_points = 100_000,
        save_fig = True,
        file_format = "pdf",
    )
