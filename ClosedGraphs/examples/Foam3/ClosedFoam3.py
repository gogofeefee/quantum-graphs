#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosedTetrahedron_Foam3Lengths.py

Tetrahedron (complete graph K4) built with the legacy Foam3 bond-length
construction (3 straight bonds + 3 arc bonds) adapted to ClosedGraphs.

- Vertex 0 is at the center; vertices 1,2,3 are arranged around it
- Uses your provided port_map (non-directional vertices, degree 3)
- Bond lengths are computed from (lA, lB, lC), Dx and l_pad (no randomness)
- Vertex scattering matrices are 3×3 δ-type with user-chosen lambda_i
- Wavelength range is 0.70–7.00 µm; n_eff = 1.0

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

# Import ClosedGraphs (NOT the DirectionalGraphs child)
try:
    from ClosedGraphs import ClosedGraphs
    print("✓ Successfully imported ClosedGraphs.")
except ImportError as e:
    print(f"✗ Failed to import ClosedGraphs: {e}")
    raise

# ========================================================================
#                         Optical / Vertex Parameters
# ========================================================================

# Effective index and wavelength span (vacuum) [µm]
neff = 1.0
wavelength_range = (0.70, 7.00)  # as requested: 0.7 → 7.0 µm

# δ-coupling parameters per vertex (edit freely)
# For degree d=3:  S = 2/(d + i*lambda) * J - I_3  (unitary for real lambda)
lambda_params = [0.0, 0.0, 0.0, 0.0]

# ========================================================================
#                 Legacy "Foam3" Bond Lengths (no randomness)
# ========================================================================

def compute_bond_lengths(l, Dx=np.zeros(3), l_pad=2.5):
    """
    Compute the six bond lengths (3 central straights, 3 outer arcs)
    according to the legacy Foam3 construction.

    Inputs
    ------
    l : array-like, shape (3,)
        Base straight lengths for the three center-to-outer bonds:
        l[0] ↔ (0–1), l[1] ↔ (0–2), l[2] ↔ (0–3)
    Dx : array-like, shape (3,), optional
        Displacement for each arc construction (default 0)
    l_pad : float, optional
        Padding used in arc-length construction (default 2.5 µm)

    Returns
    -------
    bond_length : ndarray, shape (6,)
        [ L_01, L_02, L_03, L_12, L_23, L_13 ]
        i.e., first the three center bonds, then the three outer edges.
    """
    l = np.asarray(l, dtype=float)
    Dx = np.asarray(Dx, dtype=float)

    bond_length = np.zeros(6, dtype=float)

    # Geometry used in the old code
    x1, y1 = -np.sqrt(3) / 2 * l, -0.5 * l
    x2, y2 =  np.sqrt(3) / 2 * np.roll(l, -1), -0.5 * np.roll(l, -1)

    Dx_max = np.abs(x2 - x1 - np.abs(y2 - y1))
    if np.any(Dx >= Dx_max):
        raise ValueError("Dx is too large for the arc construction.")

    R_12 = 0.5 * (x2 - x1 + y1 - y2 - Dx)
    R_21 = 0.5 * (x2 - x1 + y2 - y1 - Dx)

    # Central straight bonds
    bond_length[:3] = l
    # Outer arc bonds (order matches the np.roll: (1–2), (2–3), (3–1))
    bond_length[3:] = Dx + 2 * l_pad + (np.pi / 2.0) * (R_12 + R_21)

    return bond_length

# Set the Foam3 parameters (units: µm)
lA = 60.0
lB = np.sqrt(2.0) * lA
lC = np.sqrt(3.0) * lA
l_vec  = np.array([lA, lB, lC], dtype=float)
l_pad  = 2.5
Dx_vec = np.array([0.0, 0.0, 0.0], dtype=float)

# Compute the six bond lengths
L01, L02, L03, L12, L23, L13 = compute_bond_lengths(l_vec, Dx=Dx_vec, l_pad=l_pad)
print("✓ Foam3 bond lengths [µm]:",
      f"L01={L01:.3f}, L02={L02:.3f}, L03={L03:.3f}, L12={L12:.3f}, L23={L23:.3f}, L13={L13:.3f}")

# ========================================================================
#                   Geometry (positions) and port connectivity
# ========================================================================

# Positions from your legacy code (vertex 0 central; 1,2,3 around)
pos = {
    0: (0.0, 0.0),
    1: (0.0, lA),
    2: (-np.sqrt(3) / 2.0 * lB, -0.5 * lB),
    3: ( np.sqrt(3) / 2.0 * lC, -0.5 * lC),
}
vertex_positions = [pos[i] for i in range(4)]

# Your port_map (non-directional vertices, degree 3)
# port_map[i][p] = (j, q)  meaning: port p of vertex i connects to port q of vertex j
port_map = [
    # Vertex 0
    [ (1, 0), (2, 0), (3, 0) ],
    # Vertex 1
    [ (0, 0), (3, 2), (2, 1) ],
    # Vertex 2
    [ (0, 1), (1, 2), (3, 1) ],
    # Vertex 3
    [ (0, 2), (2, 2), (1, 1) ]
]

# ========================================================================
#                 Build the lengths dict for ClosedGraphs
# ========================================================================

# Map undirected vertex pairs → physical lengths from Foam3
edge_length_by_vertices = {
    (0, 1): L01,
    (0, 2): L02,
    (0, 3): L03,
    (1, 2): L12,
    (2, 3): L23,
    (1, 3): L13,
}

def undirected_key(i, p, j, q):
    """Return a canonical (vertex,port)-(vertex,port) key with sorted endpoints."""
    a, b = (i, p), (j, q)
    return (a, b) if a <= b else (b, a)

def vertex_pair(i, j):
    """Return sorted undirected vertex pair (vi, vj)."""
    return (i, j) if i <= j else (j, i)

lengths = {}
for i, ports in enumerate(port_map):
    for p, (j, q) in enumerate(ports):
        # only add each undirected connection once
        key = undirected_key(i, p, j, q)
        if key in lengths:
            continue
        vi, vj = vertex_pair(i, j)
        L = edge_length_by_vertices[(vi, vj)]
        lengths[key] = float(L)

print("✓ Constructed lengths dict for ClosedGraphs.")

# ========================================================================
#            Vertex scattering matrices (δ-coupling, deg 3)
# ========================================================================

def delta_coupling_S_deg3(lam: float) -> np.ndarray:
    """
    Symmetric δ-coupling for a degree-3 vertex:
        S = 2/(3 + i*lam) * J - I_3
    where J is the all-ones 3×3 matrix. For real lam, S is unitary.
    """
    J = np.ones((3, 3), dtype=complex)
    I = np.eye(3, dtype=complex)
    return (2.0 / (3.0 + 1j * lam)) * J - I

vertex_matrices = {i: delta_coupling_S_deg3(lambda_params[i]) for i in range(4)}

# ========================================================================
#                    Instantiate ClosedGraphs & (optional) plot
# ========================================================================

lam_tag = "lam=[" + ",".join(f"{x:g}" for x in lambda_params) + "]"
lamrange_tag = f"lamrange={wavelength_range[0]:.2f}-{wavelength_range[1]:.2f}"
save_dir_name = f"Tetrahedron_Foam3_{lamrange_tag}_{lam_tag}"

graph = ClosedGraphs(
    vertex_positions=vertex_positions,
    port_map=port_map,
    lengths=lengths,
    vertex_matrices=vertex_matrices,
    neff=neff,
    wavelength_range=wavelength_range,
    save_dir_name=save_dir_name,
    name='Foam3'
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
        width=0.08,
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
    
    

