#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph1_config_closed_with_coupling_func.py

Constructs a ClosedGraphs (DirectionalGraphs) object based on the physical layout from Graph1 (GDS design),
but with a coupling constant provided as a function of k (µm⁻¹) interpolated from a measured npz file.

This script:
- Imports the layout from Graph1.py
- Extracts vertex positions, port map, and bond lengths
- Converts the open graph (with leads at vertex 0) to a closed graph by merging the two bonds at v=0
- Loads a measured coupling constant vs wavelength (nm), converts to k (µm⁻¹), and builds coupling_func(k)
- Plots coupling vs k similarly to your neff/ng plots
- Instantiates a DirectionalGraphs object using coupling_func(k)
- Plots and (optionally) saves the graph with bond lengths

Author: Hugo Girin
Date: August 2025
"""

# ========================================================================
#                           Imports and Paths
# ========================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Path to Graph1.py (GDS layout)
graph1_path = os.path.expanduser('~/Documents/Research/CleanRoom/Samples/Foam_v2/litho/gds/Codes')
if graph1_path not in sys.path:
    sys.path.append(graph1_path)

# Path to ClosedGraphs / DirectionalGraphs
opengraphs_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/ClosedGraphs')
if opengraphs_path not in sys.path:
    sys.path.append(opengraphs_path)

# Import Graph1 from layout
try:
    from Graph1 import Graph1
    print('✓ Successfully imported Graph1.')
except ImportError as e:
    print(f'✗ Failed to import Graph1: {e}')
    raise

# Import ClosedGraphs class (DirectionalGraphs)
try:
    from ClosedGraphs import DirectionalGraphs
    print('✓ Successfully imported DirectionalGraphs.')
except ImportError as e:
    print(f'✗ Failed to import DirectionalGraphs: {e}')
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
    # 1) Identify the two internal bonds at vertex 0
    bonds_from_v0 = []
    for port_idx, conn in enumerate(port_map[0]):
        if conn is not None:
            bonds_from_v0.append(((0, port_idx), conn))

    if len(bonds_from_v0) != 2:
        raise ValueError(f'Expected 2 internal bonds from vertex 0, found {len(bonds_from_v0)}')

    endA, endB = bonds_from_v0
    (_, pA), (vA, portA) = endA
    (_, pB), (vB, portB) = endB

    # 2) Compute merged bond length
    keyA = tuple(sorted([endA[0], endA[1]]))
    keyB = tuple(sorted([endB[0], endB[1]]))
    try:
        lengthA = lengths[keyA]
        lengthB = lengths[keyB]
    except KeyError:
        raise KeyError('Could not find one of the v0 bond lengths in the lengths dict. '
                       f'Missing keys: {keyA} or {keyB}')
    merged_length = float(lengthA) + float(lengthB)

    # 3) New vertex positions: drop vertex 0
    new_positions = vertex_positions[1:]

    # 4) Reindex port_map (shift indices > 0 by -1)
    new_port_map = []
    for old_idx, ports in enumerate(port_map):
        if old_idx == 0:
            continue
        updated_ports = []
        for conn in ports:
            if conn is None:
                updated_ports.append(None)
            else:
                j, pj = conn
                updated_ports.append((j - 1, pj) if j > 0 else None)
        new_port_map.append(updated_ports)

    # 5) Update lengths (remove bonds touching old v=0, shift others)
    new_lengths = {}
    for (a, b), L in lengths.items():
        i, pi = a
        j, pj = b
        if i == 0 or j == 0:
            continue
        new_a = (i - 1, pi)
        new_b = (j - 1, pj)
        new_lengths[tuple(sorted([new_a, new_b]))] = float(L)

    # 6) Insert merged bond between reindexed vA and vB
    new_vA = vA - 1
    new_vB = vB - 1
    new_lengths[tuple(sorted([(new_vA, portA), (new_vB, portB)]))] = merged_length

    # 7) Patch connectivity to connect new_vA:portA <-> new_vB:portB
    new_port_map[new_vA][portA] = (new_vB, portB)
    new_port_map[new_vB][portB] = (new_vA, portA)

    return new_positions, new_port_map, new_lengths


def build_coupling_func_from_npz(npz_path):
    """
    Load a coupling constant measured vs wavelength (nm), convert to k (µm⁻¹),
    and return a callable coupling_func(k) built by interpolation.

    Parameters
    ----------
    npz_path : str
        Absolute path to the npz file with arrays: lambda_nm, coupling_constant

    Returns
    -------
    coupling_func : callable
        Interpolation function f(k) with k in µm⁻¹ (cubic, extrapolates).
    k_raw : ndarray
        The k samples (µm⁻¹) derived from the input wavelengths.
    c_raw : ndarray
        The corresponding coupling constants.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f'Coupling npz not found: {npz_path}')

    data = np.load(npz_path)
    if 'lambda_nm' not in data or 'coupling_constant' not in data:
        raise KeyError('npz must contain keys "lambda_nm" and "coupling_constant".')

    lambda_nm = np.asarray(data['lambda_nm']).astype(float)
    c_raw = np.asarray(data['coupling_constant']).astype(float)

    if lambda_nm.ndim != 1 or c_raw.ndim != 1 or lambda_nm.size != c_raw.size:
        raise ValueError('lambda_nm and coupling_constant must be 1D arrays of the same length.')

    # Convert wavelengths in nm to k in µm⁻¹: λ_µm = λ_nm / 1000 ; k = 2π / λ_µm
    lambda_um = lambda_nm / 1000.0
    k_raw = 2.0 * np.pi / lambda_um

    # Sort by increasing k (monotonic x for interpolation)
    order = np.argsort(k_raw)
    k_sorted = k_raw[order]
    c_sorted = c_raw[order]

    # Remove NaNs and potential duplicates in k
    mask = np.isfinite(k_sorted) & np.isfinite(c_sorted)
    k_clean = k_sorted[mask]
    c_clean = c_sorted[mask]

    # If there are exact duplicate k values, keep the first occurrence (or average)
    # Here: keep the first occurrence
    if k_clean.size == 0:
        raise ValueError('No valid (finite) coupling samples after cleaning.')

    unique_k, uniq_idx = np.unique(k_clean, return_index=True)
    unique_c = c_clean[uniq_idx]

    # Build interpolation function in k (µm⁻¹)
    coupling_func = interp1d(unique_k, unique_c, kind='cubic', fill_value='extrapolate')

    return coupling_func, unique_k, unique_c


def plot_coupling_vs_k(k_raw, c_raw, coupling_func, n_plot=8000, show=True):
    """
    Plot raw coupling points and the interpolated coupling curve versus k.

    Parameters
    ----------
    k_raw : ndarray
        Raw (cleaned/unique) k samples in µm⁻¹.
    c_raw : ndarray
        Raw coupling constants at k_raw.
    coupling_func : callable
        Interpolation function f(k).
    n_plot : int
        Number of points for the smooth curve.
    show : bool
        If True, call plt.show() at the end.
    """
    k_min = float(np.min(k_raw))
    k_max = float(np.max(k_raw))
    k_plot = np.linspace(k_min, k_max, int(n_plot))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_raw, c_raw, 'o', alpha=0.5, label='Measured (from npz)')
    ax.plot(k_plot, coupling_func(k_plot), '-', linewidth=2, label='Interpolated')

    ax.set_xlabel(r'$k$ (µm$^{-1}$)', fontsize=22)
    ax.set_ylabel('Coupling constant', fontsize=22)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                   labelsize=20, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                   top=True, right=True)

    ax.legend(fontsize=18, frameon=False)
    plt.tight_layout()
    if show:
        plt.show()


# ========================================================================
#                        Define Geometry Parameters
# ========================================================================

params = {
    'wg_width': 0.385,              # Waveguide width
    'typical_length': 90.0,         # Typical length between graph nodes
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
    'access_length': 460.0,
}

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

device, lengths, output_params = Graph1(params=params)
vertex_positions = output_params['vertex_positions']
port_map = output_params['port_map']

# Transforming the graph from open to closed
new_vertex_positions, new_port_map, new_lengths = convert_open_to_closed(
    vertex_positions, port_map, lengths
)

# ========================================================================
#                 Load coupling(k) from npz and build callable
# ========================================================================

coupling_npz_path = '/home/hugo/Documents/Research/Exp/Xavier/foam_v2/dose5000/barrette2/use_coupler/ring_R=100/test_coupler4_2/coupling_constant.npz'
coupling_func, k_samples, c_samples = build_coupling_func_from_npz(coupling_npz_path)
print('✓ Loaded coupling constant and built coupling_func(k).')

# Optional: plot coupling vs k similarly to your neff/ng figures
plot_coupling_vs_k(k_samples, c_samples, coupling_func, n_plot=10000, show=True)

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

# Effective refractive index (constant here; can be a callable if desired)
neff = 4.1

# Wavelength range in microns (vacuum)
wavelength_range = (1.48, 1.64)  # adjust if needed

# ========================================================================
#                      Construct the ClosedGraphs Object
# ========================================================================

graph = DirectionalGraphs(
    vertex_positions=new_vertex_positions,
    port_map=new_port_map,
    lengths=new_lengths,
    vertex_couplings=coupling_func,     # <--- function of k (µm⁻¹)
    neff=neff,                          # can also be a func if you want
    wavelength_range=wavelength_range,
    virtual_vertex_spacing=None,
    name=f"Graph1_TypicalLength={params['typical_length']:.1f}"
)

print('✓ DirectionalGraphs object created with coupling_func(k).')

# ---- Plotting the graph with lengths -----------------------------------
graph.plot_graph(show_lengths=True, save_fig=True, show_port_arrows=False)
graph.plot_graph(show_lengths=False, save_fig=True, show_port_arrows=True, head_length_frac=0.12, head_width_frac=0.09, head_offset_frac=0.24)
graph.plot_graph(show_lengths=True, save_fig=True, show_port_arrows=True, head_length_frac=0.12, head_width_frac=0.09, head_offset_frac=0.24)


# ========================================================================
#                   Spectrum computation and plotting
# ========================================================================

# --- Compute the spectrum -----------------------------------------------
spectrum, degeneracies = graph.compute_spectrum(
    N_scan=3_000_000,        # match the resolution of old code
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
    mode_idx = 100  # index of the mode to visualize

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
    
    

