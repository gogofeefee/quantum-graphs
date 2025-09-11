#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosedTetrahedron.py

Constructs a ClosedGraphs object based on a tetrahedron (complete graph K4).
This script:
- Uses 4 non-directional vertices (degree 3 each)
- Your port_map connects every pair of vertices exactly once
- Draws reproducible random edge lengths in [80,120] µm (fixed RNG seed)
- Lets you choose per-vertex δ-type parameters (lambda) for the 3×3 scattering
  matrices: S_i = 2/(3 + i*lambda_i) * J - I_3  (unitary for real lambda_i)
- Builds and returns a ClosedGraphs object
- Optionally plots & saves the graph with bond lengths

Author: Hugo Girin
Date: August 2025
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
#                     User Parameters (edit here)
# ========================================================================

# Optical parameters
neff = 2.45
wavelength_range = (0.7, 7.0)     # [µm] (vacuum)

# δ-coupling parameters (one per vertex)
# S_i = 2/(d + i*lambda_i) * J - I_d with d=3
lambda_params = [0.0, 0.0, 0.0, 0.0]   # you can change per-vertex values here

# Random-lengths settings
seed = 1234                 # fixed RNG seed (appears in save_dir_name)
L_min, L_max = 80.0, 120.0  # [µm]

# Plot/save options
file_format = "pdf"

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

# Vertex positions:
# - Vertex 0 at the center
# - Vertices 1,2,3 on a circle for a clear tetrahedron drawing
R = 120.0  # radius for layout (purely for visualization) [µm]
vertex_positions = [
    (0.0, 0.0),                                         # vertex 0 (center)
    (R*np.cos(np.deg2rad(90.0)),  R*np.sin(np.deg2rad(90.0))),   # vertex 1 (top)
    (R*np.cos(np.deg2rad(210.0)), R*np.sin(np.deg2rad(210.0))),  # vertex 2 (bottom-left)
    (R*np.cos(np.deg2rad(330.0)), R*np.sin(np.deg2rad(330.0))),  # vertex 3 (bottom-right)
]

# Port map provided by you
# port_map[i][p] = (j, q)  means "port p of vertex i connects to port q of vertex j"
port_map = [
    # Vertex 0  (ports: to 1, 2, 3)
    [ (1, 0), (2, 0), (3, 0) ],
    # Vertex 1  (ports: to 0, 3, 2)
    [ (0, 0), (3, 2), (2, 1) ],
    # Vertex 2  (ports: to 0, 1, 3)
    [ (0, 1), (1, 2), (3, 1) ],
    # Vertex 3  (ports: to 0, 2, 1)
    [ (0, 2), (2, 2), (1, 1) ]
]

# ========================================================================
#                        Edge Lengths (reproducible)
# ========================================================================

rng = np.random.default_rng(seed)
lengths = {}
for i, ports in enumerate(port_map):
    for p, (j, q) in enumerate(ports):
        # ensure we only set each undirected edge once
        if (j, q) < (i, p):
            continue
        L = rng.uniform(L_min, L_max)
        lengths[((i, p), (j, q))] = L
print("✓ Assigned random bond lengths (µm).")

# ========================================================================
#                Vertex Scattering Matrices (3×3 each)
# ========================================================================

def delta_coupling_S_deg3(lam: float) -> np.ndarray:
    """
    Symmetric δ-coupling for a degree-3 vertex:
        S = 2/(3 + i*lam) * J - I_3
    where J is the 3×3 all-ones matrix and I_3 is identity.
    For real lam, S is unitary.
    """
    J = np.ones((3, 3), dtype=complex)
    I = np.eye(3, dtype=complex)
    coeff = 2.0 / (3.0 + 1j * lam)
    return coeff * J - I

vertex_matrices = {i: delta_coupling_S_deg3(lambda_params[i]) for i in range(4)}

# ========================================================================
#                      Construct the ClosedGraphs Object
# ========================================================================

# Save-dir name encodes wavelength range, seed, and lambda values
lam_tag = "lam=[" + ",".join(f"{x:g}" for x in lambda_params) + "]"
lamrange_tag = f"lamrange={wavelength_range[0]:.3f}-{wavelength_range[1]:.3f}"
save_dir_name = f"Tetrahedron_{lamrange_tag}_seed={seed}_{lam_tag}"

graph = ClosedGraphs(
    vertex_positions=vertex_positions,
    port_map=port_map,
    lengths=lengths,
    vertex_matrices=vertex_matrices,   # explicit 3×3 matrices per vertex
    neff=neff,
    wavelength_range=wavelength_range,
    save_dir_name=save_dir_name,
    name='Tetrahedron'
)

# ---- Plotting the graph with lengths -----------------------------------
graph.plot_graph(show_lengths=True, save_fig=True)
graph.plot_graph(show_lengths=False, save_fig=True)


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
    
    

