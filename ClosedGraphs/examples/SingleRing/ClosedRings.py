#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosedSingleRing.py

Constructs a ClosedGraphs object based on a single-ring (loop) geometry.
This script:
- Defines one ring (a loop made by connecting two ports of a single vertex)
- Defines optical parameters
- Explicitly sets the 2×2 vertex scattering matrix at the only vertex
- Instantiates a ClosedGraphs object
- (Optionally) plots and saves the graph with bond lengths

Author: Hugo Girin
Date: July 2025
"""

# ========================================================================
#                           Imports and Paths
# ========================================================================

import os
import sys
import numpy as np

# Path to ClosedGraphs
closedgraphs_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/ClosedGraphs')
if closedgraphs_path not in sys.path:
    sys.path.append(closedgraphs_path)

# Import ClosedGraphs class (NOT the DirectionalGraphs child)
try:
    from ClosedGraphs import ClosedGraphs
    print("✓ Successfully imported ClosedGraphs.")
except ImportError as e:
    print(f"✗ Failed to import ClosedGraphs: {e}")
    raise

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

# Ring radius [µm]
R = 100.0

# One real vertex at the origin (the loop attaches here)
vertex_positions = [(0.0, 0.0)]  # vertex index 0

# We form one loop by connecting two distinct ports (0 and 1) of the same vertex:
#   port_map[i][p] = (j, q) means "port p of vertex i connects to port q of vertex j"
# Here: (0,0) ↔ (0,1)
port_map = [
    [ (0, 1),  # port 0 connects to port 1 (loop)
      (0, 0) ] # port 1 connects to port 0 (loop)
]

# Bond lengths dict using local-port addressing:
# key: ((i, p), (j, q)) for an undirected bond between directed ports (i,p) and (j,q)
# For a ring, the loop length is the circumference: L = 2πR
L = 2.0 * np.pi * R

lengths = {
    ((0, 0), (0, 1)): L,  # single loop
}

# Optional: a directory suffix for saved figures/outputs
save_dir_name = 'SingleRing_R={:.1f}'.format(R)

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

# Effective index (can also be a function of k)
neff = 2.45

# Wavelength range in microns (vacuum)
wavelength_range = (1.48, 1.64)

# Explicit vertex scattering matrix at the only vertex:
# Indexing: ports (0, 1). For a degree-2 Kirchhoff/Neumann vertex:
#   S = 2/d * J - I_d  with d=2  ⇒  S = [[0, 1],
#                                        [1, 0]]
# This enforces perfect transmission between the two ports: (0→1) and (1→0).
S_vertex = np.array([[0.0, 1.0],
                     [1.0, 0.0]], dtype=complex)

vertex_matrices = {0: S_vertex}

# ========================================================================
#                      Construct the ClosedGraphs Object
# ========================================================================

graph = ClosedGraphs(
    vertex_positions=vertex_positions,
    port_map=port_map,
    lengths=lengths,
    vertex_matrices=vertex_matrices,   # explicitly provided at vertex 0
    neff=neff,
    wavelength_range=wavelength_range,
    save_dir_name=save_dir_name,
    name='SingleRing'
)

# ---- Plotting the graph with lengths -----------------------------------
graph.plot_graph(show_lengths=True, save_fig=True)
graph.plot_graph(show_lengths=False, save_fig=True)


# ========================================================================
#                   Spectrum computation and plotting
# ========================================================================

# --- Compute the spectrum -----------------------------------------------
spectrum = graph.compute_spectrum(
    N_scan=1_000_000,        # match the resolution of old code
    tol=1e-16,
    show_progress=True,
    save_spectrum=True,
    overwrite=False
)

# --- Plot the ζ_B function to verify eigenvalues -------------------------
if True:  # set to False if you don't want the plot
    xlim = (1.51, 1.53)

    graph.plot_zeta_B(
        wavelength_range=None,  # from config definition
        N_plot=500_000,
        spectrum=None, #spectrum,
        save_fig=True,
        file_format='pdf',
    )
    
    graph.plot_zeta_B(
        wavelength_range=xlim,  # from config definition
        N_plot=100_000,
        spectrum=None, #spectrum,
        save_fig=True,
        file_format='pdf',
        xlim=xlim
    )
    
    graph.plot_zeta_B(
        wavelength_range=xlim,  # from config definition
        N_plot=100_000,
        spectrum=spectrum,
        save_fig=True,
        file_format='pdf',
        xlim=xlim
    )


# ========================================================================
#                        Plot of the wavefunctions
# ========================================================================

if True:  # set to True when you actually want to generate the plot
    mode_idx = 51  # index of the mode to visualize

    # --- Compute the wavefunction coefficients ----------------------------
    a_in = graph.compute_wavefunction(
        spectrum[mode_idx],
        tol=1e-4   # slightly tighter tolerance for good accuracy
    )

    # --- Plot the wavefunction energy density -----------------------------
    graph.plot_wavefunction(
        a_in=a_in,
        k=spectrum[mode_idx],
        save_fig=True,
        file_format='pdf'
    )

# ========================================================================
#                          Spectral statistics
# ========================================================================
if True:
    # Plot staircase function N(k) and Weyl law deviation
    graph.plot_staircase_function(
        plot_weyl=True,
        weyl_diff=True,
        save_fig=True,
        file_format="pdf"
    )
    
    # Plot nearest-neighbor spacing distribution (NNSD)
    graph.short_range_correlations(
        width=0.1,
        xlim=(0, 3),
        plot_wigner=['goe', 'gue', 'poissonian'],
        color_list=['r-', 'g--', 'b-.', 'orange-', 'purple--'],
        cumulative_spacing=True,
        save_fig=True,
        file_format="pdf"
    )
    
    # Plot long-range spectral correlations: Σ²(L) and Δ₃(L)
    graph.long_range_correlations(
        L_max=10,
        L_step=0.1,
        plot_wigner=['goe', 'gue', 'poissonian'],
        color_list=['r-', 'g--', 'b-.', 'orange-', 'purple--'],
        correlation_type="both",
        save_fig=True,
        file_format="pdf"
    )
    
    

