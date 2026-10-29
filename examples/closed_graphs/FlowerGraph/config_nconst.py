#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowerGraph (closed) — constant neff, constant C = 0.5
======================================================

Geometry
--------
Five 4-port directional couplers (C = 0.5) placed on a pentagon. Each pair of
neighbouring vertices is joined by TWO bonds of different lengths (a short
inner arc and a long outer arc), so that the graph looks like a five-petal
flower:

    10 bonds, 20 directed bonds, total length L_tot = 1599.62 um.

The vertex positions and bond lengths below are those of a fabricated
silicon device (typical length 65 um, coupler length 3.5 um), written out
explicitly; the vertex positions are only used for drawing.

Multiple bonds between the same pair of vertices are ordinary bonds of the
list, drawn as arcs; ``bond_spacing`` only controls how far apart they are
drawn.

Analysis pipeline: same as the DoubleRing example (topology, spectrum,
secular function, wavefunctions, entropy / IPR, spectral statistics, length
spectrum). Spectral statistics are computed twice, on the full window and on
a sub-window, to show the ``wavelength_range`` option of the statistics.
"""

# ============================================================================
#  Paths & imports
# ============================================================================

import os
import sys
import numpy as np

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # e.g. run cell-by-cell in an IDE
    _here = os.getcwd()


# Use the installed package (pip install -e .) or, failing that, this checkout.
try:
    import quantum_graphs  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..", "..")))

from quantum_graphs.closed_graphs import DirectionalGraphs


# ============================================================================
#  Geometry
# ============================================================================

# (x, y) in um -- for drawing only
vertex_positions = [
    (  0.000,  73.539),   # vertex 0
    (-64.244,  20.874),   # vertex 1
    (-33.058, -45.500),   # vertex 2
    ( 35.379, -48.695),   # vertex 3
    ( 78.195,  25.407),   # vertex 4
]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 4, 1, 123.830),   # petal 0-4
    (0, 2, 4, 3, 224.798),
    (0, 1, 1, 0, 112.068),   # petal 0-1
    (0, 3, 1, 2, 215.516),
    (1, 1, 2, 0, 100.260),   # petal 1-2
    (1, 3, 2, 2, 207.531),
    (2, 1, 3, 0, 92.901),   # petal 2-3
    (2, 3, 3, 2, 183.485),
    (3, 1, 4, 0, 116.961),   # petal 3-4
    (3, 3, 4, 2, 222.271),
]


# ============================================================================
#  Optical parameters
# ============================================================================

neff                   = 3.97           # constant index (here the group index of the waveguide)
wavelength_range       = (1.40, 1.70)   # vacuum wavelength window (um)
wavelength_range_sub   = (1.48, 1.55)   # sub-window used for a second set of statistics
C_all                  = 0.5            # same coupling for all 5 couplers

# Other possibilities, for later:
#   C_all = {0: 0.5, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.5}   # one coupling per vertex
#   C_all = lambda k: 0.5 + 0.1 * np.cos(3 * k)         # wavelength-dependent coupling
#   neff  = lambda k: 2.4 + 0.1 * (k - 4.0)             # dispersive index


# ============================================================================
#  Build the graph
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions       = vertex_positions,
    bonds                  = bonds,
    couplings       = C_all,
    neff                   = neff,
    wavelength_range       = wavelength_range,
    bond_spacing           = [40, -40, -40, -40, -40],   # drawing only
    name                   = f"FlowerGraph_neff{neff:.3f}_C{C_all:.2f}",
    base_dir               = _here,
)

print("\nGraph summary:")
print(f"  Vertices                 : {graph.n_vertices}")
print(f"  Bonds                    : {graph.n_bonds}")
print(f"  Total length             : {graph.total_length:.3f} um")
print(f"  Save path                : {graph.save_path}\n")


# ============================================================================
#  1. Topology plot
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True, save_fig=True, file_format="pdf")
    graph.plot_graph(show_lengths=True, show_port_markers=True,
                     save_fig=True, file_format="pdf")


# ============================================================================
#  2. Spectrum
# ============================================================================

# ~1600 eigenvalues in this window. The first run takes ~9 minutes; the
# result is then cached and re-loaded instantly.
spectrum, degeneracies = graph.compute_spectrum(
    show_progress = True,
    save_spectrum = True,
    overwrite     = False,
)
print(f"\nSpectrum: {len(spectrum)} eigenvalues found.")
print(f"Degeneracies present: {sorted(set(np.asarray(degeneracies).tolist()))}")


# ============================================================================
#  3. Secular function, zoomed
# ============================================================================

if True:
    xlim_zoom = (1.51, 1.52)
    graph.plot_zeta_B(wavelength_range=xlim_zoom, N_plot=50_000,
                      spectrum=spectrum, xlim=xlim_zoom,
                      save_fig=True, file_format="pdf")


# ============================================================================
#  4. Wavefunctions
# ============================================================================

if True:
    for mode_idx in [0, 100]:
        a_in = graph.compute_wavefunction(spectrum[mode_idx], tol=1e-4)
        graph.plot_wavefunction(a_in=a_in, k=spectrum[mode_idx],
                                save_fig=True, file_format="pdf")


# ============================================================================
#  5. Entropy and IPR of every eigenmode
# ============================================================================

if True:
    graph.plot_entropy(wavelength_range=None, save_fig=True, file_format="pdf")
    graph.plot_ipr(wavelength_range=None,     save_fig=True, file_format="pdf")


# ============================================================================
#  6. Spectral statistics -- full window, then a sub-window
# ============================================================================

if True:
    for wr in (None, wavelength_range_sub):
        print(f"\n--- statistics on {wr if wr else 'the full window'} ---")
        graph.plot_staircase_function(
            plot_weyl=True, weyl_diff=True, 
            wavelength_range=wr, save_fig=True, file_format="pdf")
        graph.short_range_correlations(
            width=0.1, xlim=(0, 3), plot_wigner=["goe", "gue", "poissonian"],
            color_list=["r-", "g--", "b-."], cumulative_spacing=True,
            wavelength_range=wr, yscale="linear",
            save_fig=True, file_format="pdf")
        graph.long_range_correlations(
            L_max=10, L_step=0.1, plot_wigner=["goe", "gue", "poissonian"],
            color_list=["r-", "g--", "b-."], correlation_type="both",
            wavelength_range=wr,
            save_fig=True, file_format="pdf")


# ============================================================================
#  7. Length spectrum
# ============================================================================

if True:
    graph.plot_length_spectrum(L_max=3000, n_points=100_000,
                               save_fig=True, file_format="pdf")

print("\nDone.")
