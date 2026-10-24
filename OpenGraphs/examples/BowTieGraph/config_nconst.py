#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BowTieGraph (open) — constant neff, constant couplings
======================================================

Geometry
--------
The bow-tie graph of ClosedGraphs/examples/BowTieGraph (five 4-port
directional couplers, 10 bonds, two of them doubled), as fabricated: an
access coupler (vertex 0) sits on the bond between vertices 4 and 5. Ports 0
and 2 of the access coupler are external leads (``None`` in ``port_map``):

    6 vertices, 11 bonds, 2 leads, total internal length L_tot = 1597.07 um.

The vertex positions and bond lengths are those of a fabricated silicon
device (typical length 85 um, coupler length 2.5 um), written out explicitly;
the vertex positions are only used for drawing.

Model
-----
  - neff = constant float.
  - C_internal (5 graph couplers) and C_external (access coupler) = constants.

Analysis pipeline (each section is guarded by ``if True`` — flip to disable):
    1. Graph topology plot
    2. Loss estimation (sets the imaginary part of k used for S(k))
    3. Complex resonances (poles of S) from a 2D scan of σ_min(I − U(k))
    4. Resonances detected on the transmission |S_10|² ("experimental" method)
    5. S-matrix: amplitude, amplitude + phase, Fourier transform, length
       spectrum, complex-plane trajectory
    6. Energy distribution on the graph at one resonance
    7. Resonance wavefunctions, entropy and IPR of every resonance
    8. Spectral statistics: staircase, NNSD, Σ²/Δ₃

Runtime: ~10-15 minutes on one CPU core the
first time; every expensive result is cached next to this script and
re-loaded instantly afterwards.
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

# open_graphs/ lives two levels up (OpenGraphs/open_graphs)
_open_graphs = os.path.abspath(os.path.join(_here, "..", ".."))
if _open_graphs not in sys.path:
    sys.path.insert(0, _open_graphs)

from open_graphs import DirectionalGraphs


# ============================================================================
#  Geometry
# ============================================================================

# (x, y) in um -- for drawing only
vertex_positions = [
    (  0.000, -91.942),   # vertex 0: access coupler
    (  0.000,   0.000),   # vertex 1
    ( 83.283,  48.083),   # vertex 2
    (-76.500,  44.167),   # vertex 3
    (-63.693, -36.773),   # vertex 4
    ( 68.166, -39.356),   # vertex 5
]

# port_map[i][p] = (j, q): port p of vertex i <-> port q of vertex j.
# None = external lead. Ports (0, 1) are one arm of the coupler, (2, 3) the other.
port_map = [
    [None,   (4, 0), None,   (5, 2)],   # vertex 0: leads on ports 0 and 2
    [(4, 2), (3, 3), (5, 0), (2, 1)],   # vertex 1
    [(5, 1), (1, 3), (5, 3), (3, 1)],   # vertex 2
    [(4, 1), (2, 3), (4, 3), (1, 1)],   # vertex 3
    [(0, 1), (3, 0), (1, 0), (3, 2)],   # vertex 4
    [(1, 2), (2, 0), (0, 3), (2, 2)],   # vertex 5
]

# Bond lengths in um
lengths = {
    ((0, 1), (4, 0)): 124.697,   # access coupler -> vertex 4
    ((0, 3), (5, 2)): 126.560,   # access coupler -> vertex 5
    ((1, 0), (4, 2)):  79.153,
    ((1, 1), (3, 3)):  95.085,
    ((1, 2), (5, 0)):  84.717,
    ((1, 3), (2, 1)): 103.523,
    ((2, 0), (5, 1)):  96.144,   # double bond 2-5
    ((2, 2), (5, 3)): 262.852,
    ((2, 3), (3, 1)): 279.690,
    ((3, 0), (4, 1)): 255.817,   # double bond 3-4
    ((3, 2), (4, 3)):  88.833,
}

# Direction (degrees) in which each lead is drawn
lead_orientations = {
    (0, 0): -150,   # input lead
    (0, 2):  -30,   # output lead
}


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 3.97           # constant index (here the group index of the waveguide)
wavelength_range = (1.48, 1.64)   # vacuum wavelength window (um)
C_internal       = 0.5            # the 5 couplers of the graph
C_external       = 0.3            # the access coupler


# ============================================================================
#  Build the graph
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions       = vertex_positions,
    port_map               = port_map,
    lengths                = lengths,
    internal_coupling      = C_internal,
    external_coupling      = C_external,
    neff                   = neff,
    wavelength_range       = wavelength_range,
    lead_orientations      = lead_orientations,
    name = (f"BowTieGraph_open_neff{neff:.3f}"
            f"_Cin{C_internal:.2f}_Cext{C_external:.2f}"),
    base_dir = _here,
)

print("\nGraph summary:")
print(f"  Vertices     : {graph.n_vertices}")
print(f"  Bonds        : {graph.n_bonds}")
print(f"  Leads        : {graph.n_leads}")
print(f"  Total length : {graph.total_length:.3f} um")
print(f"  Save path    : {graph.save_path}\n")


# ============================================================================
#  1. Topology plot
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True,  show_port_markers=False,
                     save_fig=True, file_format="pdf")
    graph.plot_graph(show_lengths=True,  show_port_markers=True,
                     save_fig=True, file_format="pdf")


# ============================================================================
#  2. Loss estimation
# ============================================================================
#  Propagation loss (dB/cm) and a fractional loss per coupler are converted
#  into an imaginary part of k, used below to evaluate S(k + i k_imag).

alpha_dB_per_cm   = 1.0
coupler_loss_frac = 0.05

k_imag_abs = np.log(10) / (1e4 * 20 * neff) * alpha_dB_per_cm
k_imag_cpl = (-graph.n_vertices * np.log(1 - coupler_loss_frac)
              / (2 * neff * graph.total_length))
k_imag     = k_imag_abs + k_imag_cpl

print("=== Loss contributions ===")
print(f"  Material absorption  k_imag_abs = {k_imag_abs:.3e} um^-1")
print(f"  Coupler loss         k_imag_cpl = {k_imag_cpl:.3e} um^-1")
print(f"  Total                k_imag     = {k_imag:.3e} um^-1\n")


# ============================================================================
#  3. Complex resonances
# ============================================================================
#  σ_min(I − U(k)) is evaluated on an N_re × N_im grid of the complex k-plane,
#  local minima are refined, deduplicated and accepted below eps_accept.

if True:
    spectrum, degeneracies = graph.find_resonances(
        N_re = 200_000,
        N_im = 15,
        im_min = -1e-4,
        im_max =  0.0,
        peak_prominence = 0.0,
        neighbor_rows   = 1,
        box_re   = 3,
        box_im   = 3,
        tol_step = 1e-13,
        maxiter  = 500,
        eps_accept  = 1e-4,
        nullity_tol = 1e-4,
        grid_dtype         = "float32",
        overwrite_grid     = False,
        overwrite_refined  = False,
        overwrite_spectrum = False,
        plot    = True,
        verbose = True,
    )
    print(f"\n{len(spectrum)} resonances accepted.")


# ============================================================================
#  4. Resonances detected on the transmission spectrum
# ============================================================================
#  Same method as for measured spectra: peaks of d²T/dk² with T = |S_10|².

if True:
    graph.find_experimental_resonances(
        n_points      = 500_000,
        component     = (1, 0),
        k_imag        = 2e-7,
        prominence    = None,
        amp_threshold = 9e7,
        distance      = 5,
        save          = True,
        overwrite     = False,
        plot_figures  = True,
        verbose       = True,
        file_format   = "pdf",
    )


# ============================================================================
#  5. S-matrix
# ============================================================================

S_kwargs = dict(
    lambda_min = wavelength_range[0],
    lambda_max = wavelength_range[1],
    n_points   = 500_000,
    components = (1, 0),
    k_imag     = k_imag,
    save       = True,
    file_format = "pdf",
)

if True:
    graph.plot_S_amplitude(spectrum=graph.best_spectrum, x_axis="frequency", **S_kwargs)
    graph.plot_S_amplitude_phase(spectrum=graph.best_spectrum, x_axis="frequency", **S_kwargs)
    graph.plot_S_fourier(xlim_um=(0, 5000), **S_kwargs)
    graph.plot_length_spectrum(L_max=5000, n_points=100_000,
                               save_fig=True, file_format="pdf")
    graph.plot_S_complex_plane(spectrum=graph.best_spectrum, **S_kwargs)


# ============================================================================
#  6. Energy distribution at the first resonance, injecting in lead (0, 0)
# ============================================================================

if True:
    test_k = float(np.real(graph.best_spectrum[0])) + 1j * k_imag
    graph.plot_energy_distribution(k=test_k, injection_port=0, show_leads=True,
                                   save=True, file_format="pdf")


# ============================================================================
#  7. Resonance wavefunctions (null space of I − U(k) at a complex pole),
#     entropy and IPR of every resonance
# ============================================================================

if True:
    graph.plot_resonance_wavefunction(index=0, save_fig=True, file_format="pdf")
    graph.plot_entropy(save_fig=True, file_format="pdf")
    graph.plot_ipr(save_fig=True, file_format="pdf")


# ============================================================================
#  8. Spectral statistics of the resonances
# ============================================================================

if True:
    graph.plot_staircase_function(
        plot_weyl=True, weyl_diff=True, mode=None,
        save_fig=True, file_format="pdf")
    graph.short_range_correlations(
        width=0.1, xlim=(0, 3), plot_wigner=["goe", "gue", "poissonian"],
        color_list=["r-", "g--", "b-."], cumulative_spacing=True,
        mode=None, yscale="linear", save_fig=True, file_format="pdf")
    graph.long_range_correlations(
        L_max=10, L_step=0.1, plot_wigner=["goe", "gue", "poissonian"],
        color_list=["r-", "g--", "b-."], correlation_type="both",
        mode=None, save_fig=True, file_format="pdf")

print("\nDone.")
