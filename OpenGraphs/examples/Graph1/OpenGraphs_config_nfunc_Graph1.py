#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph1_config.py

Constructs an OpenGraphs object based on the physical layout from Graph1 (GDS design).
This script:
- Imports the layout from Graph1.py
- Extracts vertex positions, port map, and bond lengths
- Loads neff and group index data from MPB simulations
- Creates interpolation functions for neff(k) and ng(k)
- Defines optical and scattering parameters
- Instantiates an OpenGraphs object
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

# Path to Graph1.py
graph1_path = os.path.expanduser('~/Documents/Research/CleanRoom/Samples/Foam_v2/litho/gds/Codes')
if graph1_path not in sys.path:
    sys.path.append(graph1_path)

# Path to OpenGraphs.py
opengraphs_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/OpenGraphs')
if opengraphs_path not in sys.path:
    sys.path.append(opengraphs_path)

# Import Graph1 from layout
try:
    from Graph1 import Graph1
    print("✓ Successfully imported Graph1.")
except ImportError as e:
    print(f"✗ Failed to import Graph1: {e}")
    raise

# Import OpenGraphs class
try:
    from OpenGraphs import DirectionalGraphs
    print("✓ Successfully imported DirectionalGraphs.")
except ImportError as e:
    print(f"✗ Failed to import DirectionalGraphs: {e}")
    raise
    

# ========================================================================
#                        Define Geometry Parameters
# ========================================================================

# These parameters control the GDS layout of the graph
# Only 'typical_length' should be modified for geometry scaling
params = {
    'wg_width': 0.385,              # Waveguide width
    'typical_length': 90.0,         # Typical length between graph nodes (modifiable)
    'routing_radius': 20.0,         # Radius used for automatic routing
    'coupler_orientation': 60.0,    # Angle of rotation for couplers
    'coupler_gap': 0.2,             # Gap in the coupler
    'coupler_length': 3.4,          # Coupling section length
    'coupler_width': 0.385,         # Width in coupler region
    'coupler_radius': 8.0,          # Euler bend minimum radius
    'coupler_angle': 45.0,          # Angle of Euler bends
    'access_coupler_gap': 0.2,      # Gap in access coupler
    'access_coupler_length': 6.0,   # Length of access coupler section
    'access_coupler_width': 0.385,  # Width in access coupler region
    'access_coupler_radius': 8.0,   # Euler bend radius in access
    'access_coupler_angle': 45.0,   # Euler bend angle in access
    'taper_length': 20.0,           # Taper length
    'taper_width': 0.2,             # Minimum width of taper
    'access_length': 460.0          # Total length of the access waveguides
}

# ========================================================================
#                  Geometrical Construction of the Graph
# ========================================================================

# Generate the GDS layout and extract geometry
device, lengths, output_params = Graph1(params=params)

# Extract necessary data from the layout
vertex_positions = output_params['vertex_positions']
port_map = output_params['port_map']

# Orientation of the leads
lead_orientations = {
    (0, 0): -150,   # Lead 0 → -150°
    (0, 2): -30     # Lead 1 → -30°
}

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
wavelength_range = (1.48, 1.64)

# Coupling constant of the directional graph
C_internal = 0.5  # Coupling for internal (non-lead) vertices
C_external = 0.1  # Coupling for external (lead-connected) vertices

# ========================================================================
#                      Construct the DirectionalGraphs Object
# ========================================================================

graph = DirectionalGraphs(
    vertex_positions=vertex_positions,
    port_map=port_map,
    lengths=lengths,
    internal_coupling=C_internal,
    external_coupling=C_external,
    neff=neff_func,          # <-- use interpolated function
    ng=ng_func,     # <-- requires DirectionalGraphs to accept this arg
    lead_orientations=lead_orientations,
    wavelength_range=wavelength_range,
    virtual_vertex_spacing=35,
    name=f"Graph1_TypicaLength={params['typical_length']:.1f}"
)

# ---- Plotting the graph with lengths -----------------------------------
graph.plot_graph(show_lengths=True, save_fig=True, show_port_arrows=False)
graph.plot_graph(show_lengths=False, save_fig=True, show_port_arrows=True, head_length_frac=0.15, head_width_frac=0.12, head_offset_frac=0.28)
graph.plot_graph(show_lengths=True, save_fig=True, show_port_arrows=True, head_length_frac=0.15, head_width_frac=0.12, head_offset_frac=0.28)


# ========================================================================
#                         Define the losses
# ========================================================================

# --- Evaluation of the losses ---
alpha_abs = 1.0             # Absorption of the silicon (dB/cm)
coupler_losses = 0.05       # Losses for a single coupler (fraction, 0.05 = 5%)

# --- Compute mid-wavenumber ---
lambda_min, lambda_max = wavelength_range
k_min = 2 * np.pi / lambda_max
k_max = 2 * np.pi / lambda_min
k_mid = 0.5 * (k_min + k_max)

# --- Evaluate neff at k_mid ---
neff_mid = float(neff_func(k_mid))

# --- Contributions ---
k_imag_abs = np.log(10) / (1e4 * 20 * neff_mid) * alpha_abs
k_imag_cpl = - graph.n_vertices * np.log(1 - coupler_losses) / (2 * neff_mid * graph.total_length)

k_imag = k_imag_abs + k_imag_cpl

# --- Print breakdown ---
print("=== Loss Contributions ===")
print(f"k_mid:                           {k_mid:.3f} µm⁻¹")
print(f"neff(k_mid):                     {neff_mid:.4f}")
print(f"k_imag_abs (material absorption): {k_imag_abs:.3e} µm⁻¹")
print(f"k_imag_cpl (coupler loss):        {k_imag_cpl:.3e} µm⁻¹")
print("---------------------------------")
print(f"Total k_imag:                     {k_imag:.3e} µm⁻¹")


# ========================================================================
#                     Resonance finding and plotting
# ========================================================================

if False:
    # --- Computation of the resonance spectrum (σ_min method) ---
    spectrum, degeneracies = graph.find_resonances_sigma2d(
        N_re=1_000_000,       # Number of grid points along the real k-axis
        N_im=25,           # Number of grid points along the imaginary k-axis
        im_min=-4e-5,       # Minimum imaginary part of k (negative, search depth)
        im_max=0.0,         # Maximum imaginary part of k (default: 0, real axis)
        eps_accept=1e-4,    # Acceptance threshold on σ_min at refined resonance
        tol_step=1e-12,     # Convergence tolerance for 2D refinement steps
        maxiter=300,        # Maximum iterations for each refinement
        refine_box_mult_re=2,   # Refinement box half-width in Re bins
        refine_box_mult_im=2,   # Refinement box half-width in Im bins
        nullity_tol=1e-4,   # SVD threshold for degeneracy (nullity) evaluation
        grid_dtype="float32",   # Storage dtype for σ_min grid (faster/smaller on disk)
        save_grid=True,     # Save σ_min grid to disk for resuming/reuse
        overwrite_grid=False,   # Force recomputation of σ_min grid if True
        save_results=True,  # Save final spectrum and degeneracies to .npz
        overwrite_results=False, # Force recomputation of spectrum/deg if True
        ensure_grid=False,   # Forcing the computation of the grid
        plot=True,          # Display heatmap of σ_min with resonances marked
        verbose=True        # Print detailed progress and warnings
    )

    # --- Plot the resonances in the complex k-plane (σ_min method) ---
    graph.plot_resonances_sigma(
        lam_min=None,            # use self.wavelength_range lower bound if None
        lam_max=None,            # use self.wavelength_range upper bound if None
        N_re=1_000_000,            # number of samples along Re(k) for grid lookup
        N_im=25,                # number of samples along Im(k) for grid lookup
        im_min=-4e-5,            # minimum Im(k) bound (negative)
        im_max=0.0,              # maximum Im(k) bound (usually 0)
        plot_heatmap=True,       # show σ_min grid heatmap if available/compute it
        overwrite_grid=False,    # recompute the σ_min grid even if cached
        grid_dtype="float32",    # dtype of σ_min grid storage
        save=True,               # save figure in graph.save_path
        file_format="pdf",       # save format: "pdf", "png", "svg", ...
        verbose=True             # print progress/status messages
    )
    
    
# ========================================================================
#                     Experimental resonance finding and plotting
# ========================================================================
# Experimental resonances are the (real) resonances we extract from the 
# transmission spectra by looking for dips in |S_ij|² for a given component.
# A small imaginary part is added to k to broaden the resonances slightly 
# so they can be sampled efficiently.

if True:
    # --- Manually specified peaks (in λ, microns) ---
    manual_lambda_peaks = [
        # Example: 1.5532, 1.5578, 1.5621
        # Add wavelengths here that the automatic algorithm struggles to detect
        1.4907756, 1.493149, 1.6191387
    ]
    
    # --- Run experimental resonance search ---
    graph.find_experimental_resonances(
        n_points=5_000_000,           # Number of sampling points between λ_min and λ_max
        component=(1, 0),             # S-matrix element (i, j) to analyze
        k_imag=1e-8,            # Small imaginary part added to k for smoothing
        prominence=0.0005,            # Minimum dip prominence for detection (None = take all minima)
        manual_lambda_peaks=None,     # Manually added peaks in λ (µm)
        save=True,                    # Save the resulting resonance .npy and figures
        overwrite=False,              # Force recomputation even if file already exists
        plot_zoom=True,               # Plot zoomed-in spectrum views if λ-range > 0.2 µm
        verbose=True,                 # Print progress while computing
        file_format="pdf"             # Format for saved plots ("pdf", "svg", "png", etc.)
    )

    
# ========================================================================
#                      Transmission and length spectrum
# ========================================================================
if True:
    # ---- Plot single S-matrix component -------------------------------------
    graph.plot_S_components(lambda_min=1.48,
                            lambda_max=1.64,
                            k_imag=k_imag,
                            n_points=1_000_000,
                            components=(1, 0),
                            spectrum=None,  # in k (µm⁻¹)
                            plot_fft=True,
                            save=False, 
                            file_format="pdf")
    
    graph.plot_S_components(lambda_min=1.484,
                            lambda_max=1.492,
                            k_imag=k_imag,
                            n_points=100_000,
                            components=(1, 0),
                            spectrum=graph.exp_spectrum,  # in k (µm⁻¹)
                            plot_fft=False,
                            save=True, 
                            file_format="pdf")
    
    graph.plot_S_components(lambda_min=1.484,
                            lambda_max=1.492,
                            k_imag=k_imag,
                            n_points=100_000,
                            components=(1, 0),
                            spectrum=None,  # in k (µm⁻¹)
                            plot_fft=False,
                            save=True, 
                            file_format="pdf")

if False:
    # ---- Plot the length spectrum -------------------------------------------
    graph.plot_length_spectrum(L_max=4999, 
                               n_points=100000, 
                               save=True, 
                               file_format="pdf")


# ========================================================================
#                        Graph and the wavefunction
# ========================================================================
  
if True:
    # ---- Plot the energy distribution ---------------------------------------
    # Choose a wavenumber in your wavelength range (1.55 µm here as an example)
    # test_lambda = 1.56  # µm
    # test_k = 2 * np.pi / test_lambda  + k_imag * 1j # vacuum wavenumber
    test_k = graph.exp_spectrum[100] + k_imag * 1j # vacuum wavenumber
    
    # Choose an injection lead
    # You can either pass an index into graph.leads or (vertex, port)
    # For example, inject from the first lead:
    injection_lead = 0  # you could also do injection_lead = graph.leads[0]
    
    # Run the plot
    graph.plot_energy_distribution(
        k=test_k,
        injection_port=injection_lead,
        show_leads=True,      # show injection/reflection/transmission arrows
        save=True,            # save figure to graph.save_path
        file_format="pdf"     # can also use "png" or "svg"
    )
    
    print("✓ plot_energy_distribution test completed.")
    

# ========================================================================
#                          Spectral statistics
# ========================================================================
if True:
    # Use experimental spectrum for the spectral statistics
    use_experimental = True
    
    # Plot staircase function N(k) and Weyl law deviation
    graph.plot_staircase_function(
        plot_weyl=True,
        weyl_diff=True,
        mode=None,
        use_experimental=use_experimental,
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
        use_experimental=use_experimental,
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
        use_experimental=use_experimental,
        save_fig=True,
        file_format="pdf"
    )
