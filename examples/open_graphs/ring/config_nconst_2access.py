#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring — two access couplers, constant neff
==========================================

Geometry
--------
2 physical 4-port vertices:
    vertex 0 : bottom access coupler  — position (0, -R)
    vertex 1 : top access coupler     — position (0, +R)

Each coupler:
    port 0 → open lead
    port 1 → upper half-ring (shared with partner)
    port 2 → open lead
    port 3 → lower half-ring (shared with partner)

Two internal half-ring bonds:
    (0,1) ↔ (1,1)  [upper half, length πR]
    (0,3) ↔ (1,3)  [lower half, length πR]

Lead orientations
-----------------
    Bottom: (0,0)→-150°, (0,2)→-30°
    Top:    (1,0)→+150°, (1,2)→+30°

S-matrix components of interest:
    (1,0) — through-port (same-side output)
    (3,2) — cross-port   (opposite-side output)

Analysis pipeline:
    1. Graph topology plot
    2. Loss estimation
    3. Complex resonance finding
    4. Experimental resonance finding
    5. S-matrix: amplitude, amplitude+phase, Fourier, complex plane
       for both (1,0) and (3,2)
    6. Energy distribution (both injection leads)
    7. Spectral statistics: staircase, NNSD, Σ²/Δ₃

See also: config_nconst_1access.py, config_nfunc_1access.py,
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

R      = 100.0
L_half = np.pi * R   # each half-ring

vertex_positions = [
    (0.0, -R),   # bottom access coupler
    (0.0,  R),   # top access coupler
]

port_map = [
    [None, (1, 1), None, (1, 3)],   # vertex 0 (bottom)
    [None, (0, 1), None, (0, 3)],   # vertex 1 (top)
]

lengths = {
    tuple(sorted([(0, 1), (1, 1)])): L_half,
    tuple(sorted([(0, 3), (1, 3)])): L_half,
}

lead_orientations = {
    (0, 0): -150.0,
    (0, 2):  -30.0,
    (1, 0):  150.0,
    (1, 2):   30.0,
}

print(f"Ring geometry: R = {R:.1f} µm, L_half = {L_half:.3f} µm, L_ring = {2*L_half:.3f} µm")


# ============================================================================
#  Optical parameters
# ============================================================================

neff             = 2.41
wavelength_range = (1.48, 1.64)
C_internal       = 0.5   # unused here (all vertices are access couplers)
C_external       = 0.3   # coupling at each access coupler


# ============================================================================
#  Instantiate DirectionalGraphs
# ============================================================================

graph = DirectionalGraphs(
    vertex_positions  = vertex_positions,
    port_map          = port_map,
    lengths           = lengths,
    internal_coupling = C_internal,
    external_coupling = C_external,
    neff              = neff,
    wavelength_range  = wavelength_range,
    lead_orientations = lead_orientations,
    name = f"Ring_2access_R{R:.0f}_neff{neff:.3f}_C{C_external:.2f}",
    base_dir = _here,
)

print(f"\nGraph summary:")
print(f"  Vertices  : {graph.n_vertices}")
print(f"  Bonds     : {graph.n_bonds}")
print(f"  Leads     : {graph.n_leads}")
print(f"  Total length: {graph.total_length:.3f} µm")
print(f"  Save path : {graph.save_path}\n")

if hasattr(graph, "leads"):
    print("Lead order:")
    for i, ld in enumerate(graph.leads):
        print(f"  lead {i}: {ld}")


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
        save         = True,
        overwrite    = False,
        plot_figures = True,
        verbose      = True,
        file_format  = "pdf",
    )


# ============================================================================
#  5. S-matrix plots — both port pairs
#  — plot_S_fourier : Fourier transform of |S(λ)|² vs optical path length
#  — plot_length_spectrum : periodic-orbit sum ρ̂(ℓ) = |Σ_j exp(-i k_j neff ℓ)|
# ============================================================================

if True:
    for comp in [(1, 0), (3, 2)]:
        graph.plot_S_amplitude(
            lambda_min = wavelength_range[0],
            lambda_max = wavelength_range[1],
            n_points   = 2_000_000,
            components = comp,
            spectrum   = None,
            k_imag     = k_imag,
            x_axis     = "frequency",
            save       = True,
            file_format = "pdf",
        )
        graph.plot_S_amplitude_phase(
            lambda_min = wavelength_range[0],
            lambda_max = wavelength_range[1],
            n_points   = 2_000_000,
            components = comp,
            spectrum   = graph.best_spectrum,
            k_imag     = k_imag,
            x_axis     = "frequency",
            save       = True,
            file_format       = "pdf",
            reuse_if_included = True,
        )
        graph.plot_S_fourier(
            lambda_min = wavelength_range[0],
            lambda_max = wavelength_range[1],
            n_points   = 2_000_000,
            components = comp,
            k_imag     = k_imag,
            xlim_um    = (0, 3000),
            save       = True,
            file_format       = "pdf",
            reuse_if_included = True,
        )
        graph.plot_S_complex_plane(
            lambda_min = wavelength_range[0],
            lambda_max = wavelength_range[1],
            n_points   = 2_000_000,
            components = comp,
            spectrum   = graph.best_spectrum,
            k_imag     = k_imag,
            save       = True,
            file_format = "pdf",
        )

    graph.plot_length_spectrum(
        L_max    = 3000,
        n_points = 100_000,
        save_fig = True,
        file_format = "pdf",
        # use_experimental = True,
    )


# ============================================================================
#  6. Energy distribution (both injection leads)
# ============================================================================

if True:
    for inj_port in [0, 2]:
        test_k = float(np.real(graph.best_spectrum[0])) + 1j * k_imag
        graph.plot_energy_distribution(
            k              = test_k,
            injection_port = inj_port,
            show_leads     = True,
            save           = True,
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
