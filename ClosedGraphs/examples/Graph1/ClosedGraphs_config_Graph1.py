#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph1_config_closed.py

Constructs a ClosedGraphs object based on the physical layout from Graph1 (GDS design).
This script:
- Imports the layout from Graph1.py
- Extracts vertex positions, port map, and bond lengths
- Defines optical and scattering parameters
- Instantiates a ClosedGraphs object
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
from pathlib import Path

# --- Set working directory manually (adapt as needed) ---
project_dir = Path('~/Documents/Research/Simul/Graphs/ClosedGraphs/Graph1').expanduser()
os.chdir(project_dir)
print(f"✓ Working directory set to: {Path.cwd()}")

# --- Path to Graph1.py (GDS layout) ---
graph1_path = Path('~/Documents/Research/CleanRoom/Samples/Foam_v2/litho/gds/Codes').expanduser()
if str(graph1_path) not in sys.path:
    sys.path.append(str(graph1_path))

# --- Path to ClosedGraphs/DirectionalGraphs module ---
closedgraphs_path = Path('~/Documents/Research/Simul/Graphs/ClosedGraphs').expanduser()
if str(closedgraphs_path) not in sys.path:
    sys.path.append(str(closedgraphs_path))

# --- Import Graph1 from layout ---
try:
    from Graph1 import Graph1
    print("✓ Successfully imported Graph1.")
except ImportError as e:
    print(f"✗ Failed to import Graph1: {e}")
    raise

# --- Import ClosedGraphs class ---
try:
    from ClosedGraphs import DirectionalGraphs
    print("✓ Successfully imported DirectionalGraphs.")
except ImportError as e:
    print(f"✗ Failed to import DirectionalGraphs: {e}")
    raise

    
# ========================================================================
#                          Helper functions
# ========================================================================

def convert_open_to_closed(vertex_positions, port_map, lengths):
    """
    Transform an open graph (with vertex 0 having two leads) into a closed graph:
    - Removes vertex 0 (lead vertex)
    - Merges its two internal bonds into one long bond
    - Reindexes all vertices and updates port_map and lengths

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Positions of all vertices (including vertex 0).
    port_map : list of list
        Connectivity map (including leads).
    lengths : dict
        Lengths dict {((i,pi), (j,pj)): L}

    Returns
    -------
    new_positions : list of (float, float)
    new_port_map : list of list
    new_lengths : dict
    """

    # ----------------------------
    # 1. Identify the two bonds to merge
    # ----------------------------
    # Bonds: (0,1)->(vA, pA) and (0,3)->(vB, pB)
    bonds_from_v0 = []
    for port_idx, conn in enumerate(port_map[0]):
        if conn is not None:  # skip leads (None)
            bonds_from_v0.append(((0, port_idx), conn))

    if len(bonds_from_v0) != 2:
        raise ValueError(f"Expected 2 internal bonds from vertex 0, found {len(bonds_from_v0)}")

    # Extract endpoints
    (endA, endB) = bonds_from_v0
    (_, pA), (vA, portA) = endA
    (_, pB), (vB, portB) = endB

    # ----------------------------
    # 2. Compute merged bond length
    # ----------------------------
    lengthA = lengths.get(tuple(sorted([endA[0], endA[1]])))
    lengthB = lengths.get(tuple(sorted([endB[0], endB[1]])))
    merged_length = lengthA + lengthB

    # ----------------------------
    # 3. Build new vertex_positions
    # ----------------------------
    new_positions = vertex_positions[1:]  # remove vertex 0

    # ----------------------------
    # 4. Reindex vertices in port_map
    # ----------------------------
    new_port_map = []
    for old_idx, ports in enumerate(port_map):
        if old_idx == 0:
            continue  # skip vertex 0
        updated_ports = []
        for conn in ports:
            if conn is None:
                updated_ports.append(None)
            else:
                j, pj = conn
                # decrement index if > 0
                updated_ports.append((j-1, pj) if j > 0 else None)
        new_port_map.append(updated_ports)

    # After this, vertex indices are shifted (old1→new0, old2→new1, etc.)

    # ----------------------------
    # 5. Update lengths
    # ----------------------------
    new_lengths = {}
    for (a, b), L in lengths.items():
        i, pi = a
        j, pj = b
        if i == 0 or j == 0:
            continue  # skip bonds touching vertex 0
        # decrement indices > 0
        new_a = (i-1, pi)
        new_b = (j-1, pj)
        new_lengths[tuple(sorted([new_a, new_b]))] = L

    # ----------------------------
    # 6. Insert merged bond
    # ----------------------------
    # IMPORTANT: after reindexing, vA and vB are shifted by -1
    new_vA = vA - 1
    new_vB = vB - 1
    new_lengths[tuple(sorted([(new_vA, portA), (new_vB, portB)]))] = merged_length

    # ----------------------------
    # 7. Patch port_map connectivity
    # ----------------------------
    # - Remove connections to the removed vertex
    # - Add the merged bond endpoints
    new_port_map[new_vA][portA] = (new_vB, portB)
    new_port_map[new_vB][portB] = (new_vA, portA)

    return new_positions, new_port_map, new_lengths

# ========================================================================
#                        Define Geometry Parameters
# ========================================================================

params = {
    'wg_width': 0.385,              # Waveguide width
    'typical_length': 90.0,         # Typical length between graph nodes (ONLY MODIFY THIS)
    'routing_radius': 20.0,         
    'coupler_orientation': 60.0,
    'coupler_gap': 0.2,
    'coupler_length': 3.4,
    'coupler_width': 0.385,
    'coupler_radius': 8.0,
    'coupler_angle': 45.0,
    'access_coupler_gap': 0.2,
    'access_coupler_length': 6.0,
    'access_coupler_width': 0.385,
    'access_coupler_radius': 8.0,
    'access_coupler_angle': 45.0,
    'taper_length': 20.0,
    'taper_width': 0.2,
    'access_length': 460.0
}

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

device, lengths, output_params = Graph1(params=params)

vertex_positions = output_params['vertex_positions']
port_map = output_params['port_map']

# Transforming the graph from open to closed
new_vertex_positions, new_port_map, new_lengths = convert_open_to_closed(vertex_positions, port_map, lengths)

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

# Effective refractive index (can also be a function of k)
neff = 2.45

# Wavelength range in microns (vacuum)
wavelength_range = (0.7, 7.0) #(1.48, 1.64)

# Coupling constant for all vertices (since all are internal for ClosedGraphs)
coupling_constant = 0.5

# ========================================================================
#                      Construct the ClosedGraphs Object
# ========================================================================

graph = DirectionalGraphs(
    vertex_positions=new_vertex_positions,
    port_map=new_port_map,
    lengths=new_lengths,
    vertex_couplings=coupling_constant,
    neff=neff,
    wavelength_range=wavelength_range,
    virtual_vertex_spacing=None,
    name=f"Graph1_TypicalLength={params['typical_length']:.1f}"
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
    N_scan=10_000_000,        # match the resolution of old code
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
#                   Plot of the wavefunctions and entropy
# ========================================================================
# Plotting the wavefunction
if True:  # set to True when you actually want to generate the plot
    mode_idx = 4009 #1146  # index of the mode to visualize

    # --- Compute the wavefunction coefficients ----------------------------
    a_in = graph.compute_wavefunction(
        graph.spectrum[mode_idx],
        tol=1e-4   # slightly tighter tolerance for good accuracy
    )

    # --- Plot the wavefunction energy density -----------------------------
    graph.plot_wavefunction(
        a_in=a_in,
        k=spectrum[mode_idx],
        cmap=None,
        save_fig=True,
        file_format='pdf'
    )

if True:
    # plotting the entropy for each mode
    graph.plot_entropy(
        wavelength_range=None,   # Only consider modes with λ between 1.50 µm and 1.60 µm
        show_smooth=True,                # Overlay the moving-average entropy curve (in red)
        smooth_window=100,                 # Size of the moving average window (number of points, must be odd)
        save_fig=True,                   # Save the figure to graph.save_path
        file_format="pdf"                # File format for saving the figure ("pdf", "png", "svg", etc.)
    )


# ========================================================================
#                   Plot of the wavefunctions and entropy
# ========================================================================
if True:
    # --- Plot the length spectrum of a closed graph ---
    graph.plot_length_spectrum(
        L_max=4999,               # maximum length (ℓ) in µm for the x-axis (default = 4999)
        n_points=100_000,         # number of points in the ℓ grid (lower than default for faster plotting)
        save=True,                # save the figure in graph.save_path
        file_format="png",        # file format for saving: "pdf", "png", "svg", etc.
        ignore_zero_peak=True,    # exclude ℓ≈0 spike when setting ylim (recommended)
        zero_window_um=100        # width of the exclusion window near ℓ=0 (µm); None = automatic
    )



# ========================================================================
#                          Spectral statistics
# ========================================================================
if True:
    # Wavelength range(s) for the spectral statistics
    lam_ranges = None
    
    # Plot staircase function N(k) and Weyl law deviation
    graph.plot_staircase_function(
        plot_weyl=True,
        weyl_diff=True,
        mode=None,
        wavelength_ranges=lam_ranges,
        show_legend=False,
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
        wavelength_ranges=lam_ranges,
        show_legend=False,
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
        wavelength_ranges=lam_ranges,
        show_legend=False,
        save_fig=True,
        file_format="pdf"
    )
    
    

