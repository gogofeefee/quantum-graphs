#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BowTieGraph (closed) — constant neff, constant C = 0.5
======================================================

Geometry
--------
Five 4-port directional couplers (C = 0.5) joined by 10 bonds. Two pairs of
vertices (1-4 and 2-3) are joined by TWO bonds of different lengths, the
other pairs by a single bond:

    10 bonds, 20 directed bonds, total length L_tot = 1597.07 um.

The vertex positions and bond lengths below are those of a fabricated
silicon device (typical length 85 um, coupler length 2.5 um), written out
explicitly; the vertex positions are only used for drawing. The measured
device also has a sixth, access coupler connected to two input/output leads:
see examples/open_graphs/BowTieGraph for the open version of the same graph.
Here the access coupler is removed and its two bonds are merged into the
single bond 3-4 (251.257 um), which closes the graph.

The two double bonds are ordinary bonds of the list, drawn as arcs.

Analysis pipeline: same as the FlowerGraph example (topology, spectrum,
secular function, wavefunctions, entropy / IPR, spectral statistics, length
spectrum).
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
    (  0.000,   0.000),   # vertex 0
    ( 83.283,  48.083),   # vertex 1
    (-76.500,  44.167),   # vertex 2
    (-63.693, -36.773),   # vertex 3
    ( 68.166, -39.356),   # vertex 4
]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 3, 2, 79.153),
    (0, 1, 2, 3, 95.085),
    (0, 2, 4, 0, 84.717),
    (0, 3, 1, 1, 103.523),
    (1, 0, 4, 1, 96.144),   # double bond 1-4
    (1, 2, 4, 3, 262.852),
    (1, 3, 2, 1, 279.690),
    (2, 0, 3, 1, 255.817),   # double bond 2-3
    (2, 2, 3, 3, 88.833),
    (3, 0, 4, 2, 251.257),   # merged bond (former access coupler)
]


# ============================================================================
#  Optical parameters
# ============================================================================

neff                   = 3.97           # constant index (here the group index of the waveguide)
wavelength_range       = (1.40, 1.70)   # vacuum wavelength window (um)
wavelength_range_sub   = (1.48, 1.57)   # sub-window used for a second set of statistics
C_all                  = 0.5            # same coupling for all 5 couplers


# ============================================================================
#  Build the graph
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions       = vertex_positions,
    bonds                  = bonds,
    vertex_couplings       = C_all,
    neff                   = neff,
    wavelength_range       = wavelength_range,
    name                   = f"BowTieGraph_neff{neff:.3f}_C{C_all:.2f}",
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

# ~1600 eigenvalues in this window. The first run takes ~6 minutes; the
# result is then cached and re-loaded instantly.
spectrum, degeneracies = graph.compute_spectrum(
    N_scan        = 3_000_000,
    tol           = 1e-16,
    show_progress = True,
    save_spectrum = True,
    overwrite     = False,
    maxiter       = 500,
    nullity_tol   = 1e-4,
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
            plot_weyl=True, weyl_diff=True, mode=None,
            wavelength_range_stats=wr, save_fig=True, file_format="pdf")
        graph.short_range_correlations(
            width=0.1, xlim=(0, 3), plot_wigner=["goe", "gue", "poissonian"],
            color_list=["r-", "g--", "b-."], cumulative_spacing=True,
            mode=None, wavelength_range_stats=wr, yscale="linear",
            save_fig=True, file_format="pdf")
        graph.long_range_correlations(
            L_max=10, L_step=0.1, plot_wigner=["goe", "gue", "poissonian"],
            color_list=["r-", "g--", "b-."], correlation_type="both",
            mode=None, wavelength_range_stats=wr,
            save_fig=True, file_format="pdf")


# ============================================================================
#  7. Length spectrum
# ============================================================================

if True:
    graph.plot_length_spectrum(L_max=3000, n_points=100_000,
                               save_fig=True, file_format="pdf")

print("\nDone.")
