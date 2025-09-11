#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosedPentagon.py

Constructs a ClosedGraphs object based on a "Pentagon" geometry.
This script:
- Defines the complete graph K5 with 5 directional 4-port vertices
- Assigns one port at each vertex to each of its 4 neighbors
- Sets random edge lengths in [80,120] µm (with fixed RNG seed for reproducibility)
- Defines optical and scattering parameters
- Instantiates a ClosedGraphs/DirectionalGraphs object
- Plots and saves the graph with bond lengths

Author: Hugo Girin
Date: August 2025
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

# Import DirectionalGraphs class
try:
    from ClosedGraphs import DirectionalGraphs
    print("✓ Successfully imported DirectionalGraphs.")
except ImportError as e:
    print(f"✗ Failed to import DirectionalGraphs: {e}")
    raise

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

# Define positions of the 5 vertices (regular pentagon for plotting only)
R = 100.0  # radius of circumcircle, arbitrary for drawing [µm]
angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 angles
vertex_positions = [(-R*np.sin(a), R*np.cos(a)) for a in angles]

# Port map: each vertex has 4 ports, one to each of its neighbors
# Format: port_map[i][p] = (j, q)
port_map = [
    # Vertex 0
    [ (4, 2), (1, 3), (2, 0), (3, 0) ],
    # Vertex 1
    [ (3, 2), (2, 3), (4, 1), (0, 1) ],
    # Vertex 2
    [ (0, 2), (3, 3), (4, 0), (1, 1) ],
    # Vertex 3
    [ (0, 3), (4, 3), (1, 0), (2, 1) ],
    # Vertex 4
    [ (2, 2), (1, 2), (0, 0), (3, 1) ]
]

# ========================================================================
#                        Define Bond Lengths
# ========================================================================

# Generate reproducible random lengths between 80 and 120 µm
rng = np.random.default_rng(seed=42)

lengths = {}
for i, ports in enumerate(port_map):
    for p, (j, q) in enumerate(ports):
        if (j, q) < (i, p):  # avoid duplicating edges
            continue
        L = rng.uniform(80.0, 120.0)
        lengths[((i, p), (j, q))] = L

print("✓ Assigned random bond lengths (µm).")

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

# Effective index
neff = 2.45

# Wavelength range [µm]
wavelength_range = (0.7, 7.0) #(1.48, 1.64)

# Coupling constant for all 4-port vertices
coupling_constant = 0.5

# Optional: directory suffix for saved figures/outputs
save_dir_name = f"Pentagon_lambdamin={wavelength_range[0]:.1f}_lambdamax={wavelength_range[1]:.1f}_C={coupling_constant:.2f}_seed=42"

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
    name='Pentagon'
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
    N_scan=5_000_000,        # match the resolution of old code
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
        N_plot=1_000_000,
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
        k=graph.spectrum[mode_idx],
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
    
    

