#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tetrahedron (closed) — constant neff
=====================================

Geometry
--------
Complete graph K4 (tetrahedron): 4 vertices, every pair connected exactly
once, so each vertex has degree 3.

    vertex 0 -- vertex 1
    vertex 0 -- vertex 2
    vertex 0 -- vertex 3
    vertex 1 -- vertex 2
    vertex 1 -- vertex 3
    vertex 2 -- vertex 3

Each vertex is a 3-port junction. The DirectionalGraphs 4-port coupler
model (constant C) does not apply here (it requires valency 4 exactly), so
every vertex uses the symmetric Neumann/Kirchhoff scattering matrix

    S = 2/3 * J - I_3

(J = all-ones 3x3 matrix), which is the natural equal-power-splitting
("50/50"-style) analogue of a constant coupler for a 3-port vertex.

Bond lengths come from an earlier "Foam3" device geometry
(lA=60, lB=sqrt(2)*lA, lC=sqrt(3)*lA um, l_pad=2.5 um, Dx=0), giving

    0-1 : 60.000 um   (straight, length lA)
    0-2 : 84.853 um   (straight, length lB)
    0-3 : 103.923 um  (straight, length lC)
    1-2 : 202.050 um  (arc)
    1-3 : 227.993 um  (arc)
    2-3 : 261.801 um  (arc)

Analysis pipeline:
    1. Graph topology plot
    2. Spectrum computation
    3. Zeta-B plot
    4. Wavefunction for a few modes
    5. Entropy and IPR
    6. Spectral statistics: staircase, NNSD, Sigma^2/Delta_3
    7. Length spectrum

Each section is guarded by ``if True / False`` — flip to enable/disable.
See also: ../SingleRing/config_nconst.py, ../DoubleRing/config_nconst.py
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
#  Tetrahedron (K4) geometry
# ============================================================================

lA = 60.0                  # um
lB = np.sqrt(2) * lA       # um
lC = np.sqrt(3) * lA       # um

# Vertex layout (for plotting only — bond lengths below are set explicitly
# and do not need to match Euclidean distances between these positions).
vertex_positions = [
    (0.0, 0.0),
    (0.0, lA),
    (-np.sqrt(3) / 2 * lB, -0.5 * lB),
    ( np.sqrt(3) / 2 * lC, -0.5 * lC),
]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 1, 0, 60.000000),   # 0-1, straight, lA
    (0, 1, 2, 0, 84.852814),   # 0-2, straight, lB
    (0, 2, 3, 0, 103.923048),   # 0-3, straight, lC
    (1, 2, 2, 1, 202.050456),   # 1-2, arc
    (1, 1, 3, 2, 227.993154),   # 1-3, arc
    (2, 2, 3, 1, 261.801408),   # 2-3, arc
]

print(f"Tetrahedron geometry: total length = {sum(b[4] for b in bonds):.3f} um")


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 2.41
wavelength_range = (1.48, 1.64)   # um


# ============================================================================
#  Vertex scattering matrices — symmetric Neumann/Kirchhoff 3-port
# ============================================================================

# 3-port Neumann vertex: S = 2/d * J - I_d  with d=3  (equal power splitting)
J3 = np.ones((3, 3), dtype=complex)
I3 = np.eye(3, dtype=complex)
S_vertex = (2.0 / 3.0) * J3 - I3

vertex_matrices = {i: S_vertex for i in range(4)}


# ============================================================================
#  Instantiate ClosedGraphs
# ============================================================================

graph = ClosedGraphs(
    vertex_positions = vertex_positions,
    bonds            = bonds,
    vertex_matrices  = vertex_matrices,
    neff             = neff,
    wavelength_range = wavelength_range,
    name             = f"Tetrahedron_nconst_neff{neff:.3f}",
    base_dir         = _here,
)

print(f"\nGraph summary:")
print(f"  Vertices     : {graph.n_vertices}")
print(f"  Bonds        : {graph.n_bonds}")
print(f"  Total length : {graph.total_length:.3f} um")
print(f"  Save path    : {graph.save_path}\n")


# ============================================================================
#  1. Topology plot
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True,  save_fig=True, file_format="pdf")
    graph.plot_graph(show_lengths=False, save_fig=True, file_format="pdf")

    # Bold thesis version: no lengths, large vertex bubbles/numbers and
    # thick bonds, legible even when the figure is shrunk down in print.
    graph.plot_graph(
        show_lengths=False,
        save_fig=True,
        file_format="pdf",
        vertex_radius_frac=0.12,
        vertex_fontsize=46,
        bond_linewidth=9,
        filename_suffix="thesis_bold",
    )


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
        weyl_diff               = True,
        wavelength_range        = wavelength_range_stats,
        save_fig                = True,
        file_format              = "pdf",
    )
    graph.short_range_correlations(
        width                   = 0.1,
        xlim                    = (0, 3),
        plot_wigner             = ["goe", "gue", "poissonian"],
        color_list               = ["r-", "g--", "b-."],
        cumulative_spacing       = True,
        wavelength_range        = wavelength_range_stats,
        yscale                   = "linear",
        save_fig                 = True,
        file_format               = "pdf",
    )
    graph.long_range_correlations(
        L_max                    = 10,
        L_step                   = 0.1,
        plot_wigner              = ["goe", "gue", "poissonian"],
        color_list                = ["r-", "g--", "b-."],
        correlation_type          = "both",
        wavelength_range        = wavelength_range_stats,
        save_fig                  = True,
        file_format                = "pdf",
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
