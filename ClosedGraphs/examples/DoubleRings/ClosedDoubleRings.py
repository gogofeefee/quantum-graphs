#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosedDoubleRings.py

Constructs a ClosedGraphs object based on Double Rings geometry.
This script:
- Defines two conjugated rings (figure-eight: two loops sharing one vertex)
- Defines optical and scattering parameters
- Instantiates a ClosedGraphs/DirectionalGraphs object
- Plots and saves the graph with bond lengths

Author: Hugo Girin
Date: July 2025
"""

# ========================================================================
#                           Imports and Paths
# ========================================================================

import os
import sys
import numpy as np

# Path to ClosedGraphs / DirectionalGraphs
opengraphs_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/ClosedGraphs')
if opengraphs_path not in sys.path:
    sys.path.append(opengraphs_path)

# Import DirectionalGraphs class (ClosedGraphs for closed/directional vertices)
try:
    from ClosedGraphs import DirectionalGraphs
    print("✓ Successfully imported DirectionalGraphs.")
except ImportError as e:
    print(f"✗ Failed to import DirectionalGraphs: {e}")
    raise

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

# Radii of the two rings [µm]
R1 = 100
R2 = R1 / np.sqrt(2)

# One real vertex at the origin (both loops attach here)
vertex_positions = [(0.0, 0.0)]  # vertex index 0

# The closed/directional vertex has (at least) 4 ports.
# We form two loops:
#   - Loop 1 uses ports (0, 1) on vertex 0
#   - Loop 2 uses ports (2, 3) on vertex 0
#
# port_map[i][p] = (j, q) means "port p of vertex i connects to port q of vertex j"
# Loops are made by connecting two distinct ports of the SAME vertex.

port_map = [
    [ (0, 2),  # port 0 connects to port 2 (loop 1)
      (0, 3),  # port 1 connects to port 3 (loop 2)
      (0, 0),  # port 2 connects to port 0 (loop 1)
      (0, 1) ] # port 3 connects to port 1 (loop 2)
]

# Bond lengths dict using local-port addressing:
# key: ((i, p), (j, q)) for an undirected bond between directed ports (i,p) and (j,q)
# For a loop, length is the circumference: L = 2πR
L1 = 2.0 * np.pi * R1
L2 = 2.0 * np.pi * R2

lengths = {
    ((0, 0), (0, 2)): L1,  # loop 1
    ((0, 1), (0, 3)): L2,  # loop 2
}

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

# Effective index (can also be a function of k)
neff = 2.45

# Wavelength range in microns (vacuum)
wavelength_range = (1.48, 1.64)

# Coupling constant for the (only) vertex
# Your class should interpret this as the internal scattering parameter
# for a 4-port directional vertex (ports 0..3).
coupling_constant = 0.5

# Optional: a directory suffix for saved figures/outputs
save_dir_name = f"DoubleRings_R1={R1:.1f}_R2={R2:.1f}_C={coupling_constant:.2f}"

# ========================================================================
#                      Construct the ClosedGraphs Object
# ========================================================================

graph = DirectionalGraphs(
    vertex_positions=vertex_positions,
    port_map=port_map,
    lengths=lengths,
    vertex_couplings=coupling_constant,
    neff=neff,
    wavelength_range=wavelength_range,
    save_dir_name=save_dir_name,
    name='DoubleRings'
)

# ---- Plotting the graph with lengths -----------------------------------
graph.plot_graph(show_lengths=True, save_fig=True, show_port_arrows=False)
graph.plot_graph(show_lengths=False, save_fig=True, show_port_arrows=True, head_length_frac=0.12, head_width_frac=0.09, head_offset_frac=0.24)
graph.plot_graph(show_lengths=True, save_fig=True, show_port_arrows=True, head_length_frac=0.12, head_width_frac=0.09, head_offset_frac=0.24)


# ========================================================================
#                   Spectrum computation and plotting
# ========================================================================

# --- Compute the spectrum -----------------------------------------------
spectrum, degeneracies = graph.compute_spectrum(
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
    mode_idx = 54  # index of the mode to visualize

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
        mode=None,
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
        mode=None,
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
        mode=None,
        save_fig=True,
        file_format="pdf"
    )
    
    

