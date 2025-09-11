#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph1_config_closed.py

Constructs a ClosedGraphs object based on the physical layout from Graph1 (GDS design).
This script:
- Imports the layout from Graph1.py
- Extracts vertex positions, port map, and bond lengths
- Loads neff and group index data from MPB simulations
- Creates interpolation functions for neff(k) and ng(k)
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
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d   # for neff(k) and ng(k) interpolation

# Path to Graph1.py (GDS layout)
graph1_path = os.path.expanduser('~/Documents/Research/CleanRoom/Samples/Foam_v2/litho/gds/Codes')
if graph1_path not in sys.path:
    sys.path.append(graph1_path)

# Path to OpenGraphs.py (now also contains ClosedGraphs)
opengraphs_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/ClosedGraphs')
if opengraphs_path not in sys.path:
    sys.path.append(opengraphs_path)

# Import Graph1 from layout
try:
    from Graph1 import Graph1
    print("✓ Successfully imported Graph1.")
except ImportError as e:
    print(f"✗ Failed to import Graph1: {e}")
    raise

# Import ClosedGraphs class
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
#                 Load neff(k) and ng(k) from MPB Results
# ========================================================================

# Path to MPB dispersion results
mpb_path = "/home/hugo/Documents/Research/Simul/MPB/sio2_si_hsq/dispersion_sio2_si_hsq/k0.0_to_10.0_res512_numk500_width0.39_height0.22"

# --- Load effective index ---
neff_data = np.load(os.path.join(mpb_path, "neff_vs_kvac_band1_full.npz"))
k_neff = neff_data["k_vac"]
neff_vals = neff_data["neff"]

# Interpolation function for neff(k)
neff_func = interp1d(k_neff, neff_vals, kind='cubic', fill_value="extrapolate")

# --- Load group index ---
ng_data = np.load(os.path.join(mpb_path, "group_index_vs_kvac_band1_full.npz"))
k_ng = ng_data["k_vac"]
ng_vals = ng_data["ng"]

# Interpolation function for ng(k)
ng_func = interp1d(k_ng, ng_vals, kind='cubic', fill_value="extrapolate")

print("✓ Loaded neff(k) and ng(k) from MPB data.")

# ========================================================================
#                     Plot neff(k) and ng(k) from MPB
# ========================================================================

# --- Build a smooth k array for plotting ---
k_plot = np.linspace(min(k_neff.min(), k_ng.min()), 
                     max(k_neff.max(), k_ng.max()), 
                     10000)

# 1) ---- Plot neff(k) --------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(k_neff, neff_vals, 'o', color='gray', alpha=0.5, label='MPB data')
ax1.plot(k_plot, neff_func(k_plot), '-', color='black', linewidth=2, label='Interpolated')

# --- Axis labels ---
ax1.set_xlabel(r"$k$ (µm$^{-1}$)", fontsize=22)
ax1.set_ylabel(r"$n_\mathrm{eff}(k)$", fontsize=22)

# --- Ticks formatting ---
ax1.minorticks_on()
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                labelsize=20, top=True, right=True)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                top=True, right=True)

# --- Legend ---
ax1.legend(fontsize=18, frameon=False)

plt.tight_layout()
plt.show()

# 2) ---- Plot ng(k) --------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(k_ng, ng_vals, 'o', color='gray', alpha=0.5, label='MPB data')
ax2.plot(k_plot, ng_func(k_plot), '-', color='black', linewidth=2, label='Interpolated')

# --- Axis labels ---
ax2.set_xlabel(r"$k$ (µm$^{-1}$)", fontsize=22)
ax2.set_ylabel(r"$n_g(k)$", fontsize=22)

# --- Ticks formatting ---
ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                labelsize=20, top=True, right=True)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                top=True, right=True)

# --- Legend ---
ax2.legend(fontsize=18, frameon=False)

plt.tight_layout()
plt.show()

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

# Wavelength range in microns (vacuum)
wavelength_range = (1.48, 1.55) # (1.14, 3.1)

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
    neff=neff_func,
    ng=ng_func,
    wavelength_range=wavelength_range,
    name=f"Graph1_TypicaLength={params['typical_length']:.1f}"
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
    N_scan=2_000_000,        # match the resolution of old code
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
    mode_idx = 104  # index of the mode to visualize

    # --- Compute the wavefunction coefficients ----------------------------
    a_in = graph.compute_wavefunction(
        graph.spectrum[mode_idx],
        tol=1e-4   # slightly tighter tolerance for good accuracy
    )

    # --- Plot the wavefunction energy density -----------------------------
    graph.plot_wavefunction(
        a_in=a_in,
        k=spectrum[mode_idx],
        save_fig=True,
        file_format='svg'
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
    
    

