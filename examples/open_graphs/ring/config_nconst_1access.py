#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring — single access coupler, constant neff
===========================================

Geometry
--------
1 physical 4-port vertex (single access coupler):
    vertex 0 : access coupler
        port 0 → open lead
        port 1 → internal ring (self-loop)
        port 2 → open lead
        port 3 → internal ring (self-loop)

The ring is a self-loop of total length L_ring = 2πR connecting ports 1 and 3
of vertex 0: an ordinary bond of the bond list, drawn as a teardrop.

Lead orientations
-----------------
    (0, 0) → -150°
    (0, 2) →  -30°

Analysis pipeline:
    1. Graph topology plot
    2. Loss estimation
    3. Complex resonance finding
    4. Experimental resonance finding
    5. S-matrix: amplitude, amplitude+phase, Fourier, complex plane
    6. Energy distribution
    7. Spectral statistics: staircase, NNSD, Σ²/Δ₃

Each section is guarded by ``if True / False`` — flip to enable/disable.
See also: config_nfunc_1access.py, config_nconst_2access.py,
          config_nfunc_2access.py
"""

# ============================================================================
#  Paths & imports
# ============================================================================

import os
import sys
import numpy as np

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _argv0 = sys.argv[0] if sys.argv else ""
    _here  = (os.path.dirname(os.path.abspath(_argv0))
              if os.path.isfile(_argv0) else os.getcwd())


# Use the installed package (pip install -e .) or, failing that, this checkout.
try:
    import quantum_graphs  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..", "..")))

from quantum_graphs.open_graphs import DirectionalGraphs


# ============================================================================
#  Ring geometry
# ============================================================================

R      = 100.0               # radius (µm)
L_ring = 2.0 * np.pi * R    # total circumference (µm)

vertex_positions = [(0.0, 0.0)]

# Self-loop: port 1 ↔ port 3 (both at vertex 0)
# Bond (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um)
bonds = [(0, 1, 0, 3, L_ring)]

# External leads (vertex, port), numbered in this order in the S-matrix
leads = [(0, 0), (0, 2)]

lead_orientations = {
    (0, 0): -150.0,
    (0, 2):  -30.0,
}

print(f"Ring geometry: R = {R:.1f} µm, L = {L_ring:.3f} µm")


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 2.41
wavelength_range = (1.48, 1.64)
C_internal       = 0.5    # coupling constant of the (unused) internal vertex
C_external       = 0.3    # coupling constant of the access coupler


# ============================================================================
#  Instantiate DirectionalGraphs
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions  = vertex_positions,
    bonds             = bonds,
    leads             = leads,
    couplings = C_internal,
    access_couplings = C_external,
    neff              = neff,
    wavelength_range  = wavelength_range,
    lead_orientations = lead_orientations,
    name = f"Ring_1access_R{R:.0f}_neff{neff:.3f}_C{C_external:.2f}",
    base_dir = _here,
)

print(f"\nGraph summary:")
print(f"  Vertices  : {graph.n_vertices}")
print(f"  Bonds     : {graph.n_bonds}")
print(f"  Leads     : {graph.n_leads}")
print(f"  Total length: {graph.total_length:.3f} µm")
print(f"  Save path : {graph.save_path}\n")


# ============================================================================
#  1. Topology plot
# ============================================================================

if True:
    graph.plot_graph(
        show_lengths      = True,
        show_port_markers = False,
        save_fig          = True,
        file_format       = "pdf",
    )
    graph.plot_graph(
        show_lengths      = False,
        show_port_markers = True,
        save_fig          = True,
        file_format       = "pdf",
    )


# ============================================================================
#  2. Loss estimation
# ============================================================================

alpha_dB_per_cm   = 1.0
coupler_loss_frac = 0.05

lambda_min, lambda_max = wavelength_range
k_mid    = 0.5 * (2 * np.pi / lambda_max + 2 * np.pi / lambda_min)
neff_mid = float(neff)

k_imag_abs = np.log(10) / (1e4 * 20 * neff_mid) * alpha_dB_per_cm
k_imag_cpl = (
    -graph.n_vertices
    * np.log(1 - coupler_loss_frac)
    / (2 * neff_mid * graph.total_length)
)
k_imag = k_imag_abs + k_imag_cpl

print("=== Loss Contributions ===")
print(f"  k_mid                          = {k_mid:.4f} µm⁻¹")
print(f"  neff                           = {neff_mid:.5f}")
print(f"  Material absorption  k_imag_abs = {k_imag_abs:.3e} µm⁻¹")
print(f"  Coupler loss         k_imag_cpl = {k_imag_cpl:.3e} µm⁻¹")
print(f"  Total                k_imag     = {k_imag:.3e} µm⁻¹\n")


# ============================================================================
#  3. Complex resonance finding
# ============================================================================

if True:
    spectrum, degeneracies = graph.find_resonances(
        N_re = 1_000_000,
        N_im = 25,
        im_min = -1e-4,
        im_max =  0.0,
        peak_prominence = 0.0,
        neighbor_rows   = 1,
        box_re   = 3,
        box_im   = 3,
        tol_step = 1e-13,
        maxiter  = 500,
        eps_accept  = 1e-4,
        nullity_tol = 1e-4,
        grid_dtype         = "float32",
        overwrite_grid     = False,
        overwrite_refined  = False,
        overwrite_spectrum = False,
        plot    = True,
        verbose = True,
    )

    print(f"\nAccepted resonances ({len(spectrum)} total):")
    for idx, (k, d) in enumerate(zip(spectrum, degeneracies)):
        print(f"  {idx:4d}  Re(k) = {np.real(k):.10e}  "
              f"Im(k) = {np.imag(k):.10e}  deg = {d}")


# ============================================================================
#  4. Experimental resonance finding
# ============================================================================

if True:
    graph.find_experimental_resonances(
        n_points      = 2_000_000,
        component     = (1, 0),
        k_imag        = 8e-8,
        prominence    = None,
        amp_threshold = 0,
        distance      = 5,
        manual_lambda_peaks  = [],
        exclude_lambda_peaks = [],
        exclude_tol_nm       = 2e-4,
        save_fig     = True,
        overwrite    = False,
        plot         = True,
        verbose      = True,
        file_format  = "pdf",
    )


# ============================================================================
#  5a. S-matrix amplitude
# ============================================================================

if True:
    graph.plot_S_amplitude(
        lambda_min = wavelength_range[0],
        lambda_max = wavelength_range[1],
        n_points   = 2_000_000,
        components = (1, 0),
        spectrum   = None,
        k_imag     = k_imag,
        x_axis     = "frequency",
        save_fig   = True,
        file_format = "pdf",
    )
    graph.plot_S_amplitude(
        lambda_min = wavelength_range[0],
        lambda_max = wavelength_range[1],
        n_points   = 2_000_000,
        components = (1, 0),
        spectrum   = graph.best_spectrum,
        k_imag     = k_imag,
        x_axis     = "frequency",
        save_fig   = True,
        file_format       = "pdf",
        reuse_if_included = True,
    )


# ============================================================================
#  5b. S-matrix amplitude + phase (two-panel)
# ============================================================================

if True:
    graph.plot_S_amplitude_phase(
        lambda_min = wavelength_range[0],
        lambda_max = wavelength_range[1],
        n_points   = 2_000_000,
        components = (1, 0),
        spectrum   = graph.best_spectrum,
        k_imag     = k_imag,
        x_axis     = "frequency",
        save_fig   = True,
        file_format       = "pdf",
        reuse_if_included = True,
    )


# ============================================================================
#  5c. Length analysis: Fourier TF and length spectrum
#  — plot_S_fourier : Fourier transform of |S(λ)|² vs optical path length
#  — plot_length_spectrum : periodic-orbit sum ρ̂(ℓ) = |Σ_j exp(-i k_j neff ℓ)|
# ============================================================================

if True:
    graph.plot_S_fourier(
        lambda_min = wavelength_range[0],
        lambda_max = wavelength_range[1],
        n_points   = 2_000_000,
        components = (1, 0),
        k_imag     = k_imag,
        xlim_um    = (0, 3000),
        save_fig   = True,
        file_format       = "pdf",
        reuse_if_included = True,
    )

    graph.plot_length_spectrum(
        L_max    = 3000,
        n_points = 100_000,
        save_fig = True,
        file_format = "pdf",
        # use_experimental = True,
    )


# ============================================================================
#  5d. S-matrix complex-plane trajectory
# ============================================================================

if True:
    graph.plot_S_complex_plane(
        lambda_min = wavelength_range[0],
        lambda_max = wavelength_range[1],
        n_points   = 2_000_000,
        components = (1, 0),
        spectrum   = graph.best_spectrum,
        k_imag     = k_imag,
        save_fig   = True,
        file_format = "pdf",
    )


# ============================================================================
#  6. Energy distribution
# ============================================================================

if True:
    test_k = float(np.real(graph.best_spectrum[0])) + 1j * k_imag

    graph.plot_energy_distribution(
        k              = test_k,
        injection_port = 0,
        show_leads     = True,
        save_fig       = True,
        file_format    = "pdf",
    )


# ============================================================================
#  7. Spectral statistics
# ============================================================================

if True:
    use_experimental       = True
    wavelength_range_stats = None

    graph.plot_staircase_function(
        plot_weyl              = True,
        weyl_diff              = True,
        mode                   = None,
        use_experimental       = use_experimental,
        wavelength_range_stats = wavelength_range_stats,
        save_fig               = True,
        file_format            = "pdf",
    )

    graph.short_range_correlations(
        width                  = 0.1,
        xlim                   = (0, 3),
        plot_wigner            = ["goe", "gue", "poissonian"],
        color_list             = ["r-", "g--", "b-."],
        cumulative_spacing     = True,
        mode                   = None,
        use_experimental       = use_experimental,
        wavelength_range_stats = wavelength_range_stats,
        yscale                 = "linear",
        save_fig               = True,
        file_format            = "pdf",
    )

    graph.long_range_correlations(
        L_max                  = 10,
        L_step                 = 0.1,
        plot_wigner            = ["goe", "gue", "poissonian"],
        color_list             = ["r-", "g--", "b-."],
        correlation_type       = "both",
        mode                   = None,
        use_experimental       = use_experimental,
        wavelength_range_stats = wavelength_range_stats,
        save_fig               = True,
        file_format            = "pdf",
    )
