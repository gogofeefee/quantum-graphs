#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 18:02:43 2025

@author: hugo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring_config.py

Constructs an OpenGraphs object for a *single vertex ring* with:
- one self-loop,
- two leads attached to the same vertex.

We define:
- Geometry and bond lengths,
- Optical parameters,
- Vertex scattering matrix.

Author: Hugo Girin
Date: July 2025
"""

# ========================================================================
#                           Imports and Paths
# ========================================================================

import os
import sys
import numpy as np

# Path to OpenGraphs.py
opengraphs_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/OpenGraphs')
if opengraphs_path not in sys.path:
    sys.path.append(opengraphs_path)

# Import DirectionalGraphs subclass
try:
    from OpenGraphs import DirectionalGraphs
    print("Successfully imported DirectionalGraphs.")
except ImportError as e:
    print(f"Failed to import DirectionalGraphs: {e}")
    raise

# ========================================================================
#                        Define Geometry Parameters
# ========================================================================

R = 100.0       # radius in microns
L_ring = 2 * np.pi * R  # circumference of the ring

# One vertex at the origin
vertex_positions = [(0.0, 0.0)]

# Port map for the vertex:
# - port 0 → lead
# - port 1 → lead
# - ports 2–3 → self-loop
port_map = [[None, (0, 3), None, (0, 1)]]

# Define bond lengths dictionary (self-loop)
lengths = {((0, 1), (0, 3)): L_ring}

# ========================================================================
#                 Define Optical and Scattering Parameters
# ========================================================================

neff = 2.45  # Effective refractive index (constant)
wavelength_range = (1.48, 1.64)  # Vacuum wavelength range
C_ext = 0.1  # Coupling constant
losses = np.log(1 - C_ext) / (2 * neff * L_ring)  # Losses (Im(k_res) in µm^-1)

# ========================================================================
#                      Construct the DirectionalGraphs Object
# ========================================================================

graph = DirectionalGraphs(
    vertex_positions=vertex_positions,
    port_map=port_map,
    lengths=lengths,
    external_coupling=C_ext,     # This sets the 4x4 scattering matrix automatically
    neff=neff,
    wavelength_range=wavelength_range,
    name="RingGraph"
)

# ========================================================================
#                         Define the losses
# ========================================================================

# --- Evaluation of the losses ---
alpha_abs = 1              # Absorption of the silicon (dB/cm)
coupler_losses = 0.1       # Losses for a single coupler (fraction, 0.05 = 5%)

# --- Contributions ---
k_imag_abs = np.log(10) / (1e4 * 20 * neff) * alpha_abs
k_imag_cpl = - graph.n_vertices * np.log(1 - coupler_losses) / (2 * neff * graph.total_length)

k_imag = k_imag_abs + k_imag_cpl

# --- Print breakdown ---
print("=== Loss Contributions ===")
print(f"k_imag_abs (material absorption): {k_imag_abs:.3e} µm⁻¹")
print(f"k_imag_cpl (coupler loss):        {k_imag_cpl:.3e} µm⁻¹")
print("---------------------------------")
print(f"Total k_imag:                     {k_imag:.3e} µm⁻¹")


# ========================================================================
#                     Resonance finding and plotting
# ========================================================================

if True:
    # --- Computation of the resonance spectrum ---
    resonances = graph.find_resonances(
        min_Q=10000,        # Minimum quality factor (Q) of resonances to include
        tol=1e-16,          # Convergence tolerance for Newton refinement
        save=True,          # Whether to save resonance data and plots to disk
        overwrite=False,    # Force recomputation even if a saved file exists
        re_points=2000,     # Number of grid points along the real k-axis
        im_points=3000,     # Number of grid points along the imaginary k-axis
        max_iter=50,        # Maximum Newton iterations for refining each root
        plot=True,          # Display heatmap of -log10|det| with found resonances
        verbose=True,       # Print progress updates during computation
        threshold=None      # Only keep local maxima of -log10|det| above this value (None = keep all)
    )
    
if False:
    # --- Plot the resonances in the complex k-plane ---
    graph.plot_resonances(
        lambda_range=None,       # use the default wavelength range from the graph
        plot_secular=True,       # show the secular determinant heatmap
        plot_resonances=True,    # overlay the computed resonances
        save=True,               # save the figure in graph.save_path
        file_format="pdf",       # save as PDF (can also use "png", "svg", etc.)
        verbose=True             # print progress messages
        )
    
    
# ========================================================================
#                      Transmission and length spectrum
# ========================================================================
if False:
    # ---- Plot single S-matrix component -------------------------------------
    graph.plot_S_components(lambda_min=1.48,
                            lambda_max=1.64,
                            k_imag=k_imag,
                            n_points=100000,
                            components=(0, 1),
                            spectrum=graph.resonances,  # in k (µm⁻¹)
                            plot_fft=True,
                            save=False, 
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
if False:
    # ---- Plot and save the graph --------------------------------------------
    graph.plot_graph(show_lengths=True, save_fig=True, file_format="pdf")
    
if False:
    # ---- Plot the energy distribution ---------------------------------------
    # Choose a wavenumber in your wavelength range (1.55 µm here as an example)
    test_lambda = 1.56  # µm
    # test_k = 2 * np.pi / test_lambda  + k_imag * 1j # vacuum wavenumber
    test_k = graph.resonances[50].real + 0.01  + k_imag * 1j # vacuum wavenumber
    
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
        cumulative_spacing=True,
        save_fig=True,
        file_format="pdf"
    )
    
    # Plot long-range spectral correlations: Σ²(L) and Δ₃(L)
    graph.long_range_correlations(
        L_max=10,
        L_step=0.2,
        plot_wigner=['goe', 'gue', 'poissonian'],
        correlation_type="both",
        save_fig=True,
        file_format="pdf"
    )


