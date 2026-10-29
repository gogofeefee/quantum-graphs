#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pentagon K5 (closed) — constant neff, constant C = 0.5
=======================================================

Geometry
--------
Complete graph K5: five 4-port directional couplers (C = 0.5), every pair of
vertices joined by exactly one bond (10 bonds, 20 directed bonds). The bond
lengths were drawn once uniformly in [80, 120] um (seed 42) and are written
out below; the vertex positions (a regular pentagon) are only used for
drawing.

Why this graph is interesting
-----------------------------
Every coupler matrix σ is symmetric (reciprocal vertices), yet the graph has
no time-reversal symmetry. A ray entering a coupler on one arm leaves on the
other arm, and with this wiring the directed bonds split into TWO
independent sectors of 10 directed bonds each (``graph.sectors``): a wave
turning "clockwise" never couples to the reversed waves. Reciprocity maps one
sector onto the other, so

* every level is exactly doubly degenerate (the two partner sectors have the
  same spectrum; ``compute_spectrum`` computes it once);
* inside one sector there is no antiunitary symmetry left: the desymmetrised
  spectrum follows the GUE, not the GOE, although σ is symmetric
  (unidirectional graphs, che2022 mechanism).

The Weyl law of one sector uses its own directed length,
L_s = 1038.572 um = the total length of the graph (not twice it).

Constant n_eff and C over a broad window (0.8–2.0 um) are an idealisation
chosen to have ~1900 levels per sector, enough for long-range statistics.

Analysis pipeline:
    1. Topology plot (lengths, ports)
    2. Sectors
    3. Spectrum (one sector per pair of partners) — a few seconds
    4. One eigenmode
    5. Spectral statistics vs GUE: staircase, NNSD, Σ² / Δ₃
    6. Length spectrum

Each section is guarded by ``if True / False`` — flip to enable/disable.
See also: ../DoubleRing/config_nconst.py (another graph of two partner sectors)
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

# Regular pentagon of radius 100 um, vertex 0 at the top, counter-clockwise
# (x, y) in um -- for drawing only
R = 100.0
angles = 2 * np.pi * np.arange(5) / 5
vertex_positions = [(-R * np.sin(a), R * np.cos(a)) for a in angles]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 4, 2, 110.958242),
    (0, 1, 1, 3,  97.555138),
    (0, 2, 2, 0, 114.343917),
    (0, 3, 3, 0, 107.894721),
    (1, 0, 3, 2,  83.767094),
    (1, 1, 2, 3, 119.024894),
    (1, 2, 4, 1, 110.445588),
    (2, 1, 3, 3, 111.442572),
    (2, 2, 4, 0,  85.124545),
    (3, 1, 4, 3,  98.015438),
]


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 2.45          # constant effective index (= group index)
wavelength_range = (0.8, 2.0)    # vacuum wavelength window (um)
C                = 0.5           # power cross-coupling of every coupler


# ============================================================================
#  Build the graph
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions = vertex_positions,
    bonds            = bonds,
    couplings        = C,
    neff             = neff,
    wavelength_range = wavelength_range,
    name             = f"Pentagon_nconst_neff{neff:.3f}_C{C:.2f}",
    base_dir         = _here,
)

print("\nGraph summary:")
print(f"  Vertices     : {graph.n_vertices}")
print(f"  Bonds        : {graph.n_bonds}")
print(f"  Total length : {graph.total_length:.3f} um")
print(f"  Save path    : {graph.save_path}\n")


# ============================================================================
#  1. Topology plot
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True, save_fig=True, file_format="pdf")
    graph.plot_graph(show_port_markers=True, save_fig=True, file_format="pdf")


# ============================================================================
#  2. Sectors
# ============================================================================

# Two partner sectors of 10 directed bonds, L_s = 1038.572 um each, GUE.
print(graph.sectors)
print(f"Expected ensemble of the desymmetrised spectrum: "
      f"{graph.spectral_ensemble()}")


# ============================================================================
#  3. Spectrum
# ============================================================================

spectrum, degeneracies = graph.compute_spectrum(
    show_progress = True,
    save_spectrum = True,
    overwrite     = False,
)
print(f"\nSpectrum: {len(spectrum)} distinct levels, "
      f"multiplicities {sorted(set(np.asarray(degeneracies).tolist()))}")
print(f"Levels of one sector: {len(graph.levels())}, "
      f"Weyl law: {graph.weyl_count():.1f}")


# ============================================================================
#  4. One eigenmode
# ============================================================================

if True:
    mode_idx = 500
    a_in = graph.compute_wavefunction(spectrum[mode_idx], tol=1e-4)
    graph.plot_wavefunction(a_in=a_in, k=spectrum[mode_idx],
                            save_fig=True, file_format="pdf")


# ============================================================================
#  5. Spectral statistics of the desymmetrised spectrum (one sector)
# ============================================================================

if True:
    graph.plot_staircase_function(
        plot_weyl=True, weyl_diff=True,
        save_fig=True, file_format="pdf")
    graph.short_range_correlations(
        width=0.1, xlim=(0, 3), plot_wigner=["goe", "gue", "poissonian"],
        color_list=["r-", "g--", "b-."], cumulative_spacing=True,
        yscale="linear", save_fig=True, file_format="pdf")
    graph.long_range_correlations(
        L_max=10, L_step=0.1, plot_wigner=["goe", "gue", "poissonian"],
        color_list=["r-", "g--", "b-."], correlation_type="both",
        save_fig=True, file_format="pdf")


# ============================================================================
#  6. Length spectrum
# ============================================================================

if True:
    graph.plot_length_spectrum(L_max=1500, n_points=100_000,
                               save_fig=True, file_format="pdf")

print("\nDone.")
