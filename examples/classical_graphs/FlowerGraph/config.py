#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowerGraph — classical Perron-Frobenius dynamics
=================================================

Same closed graph as examples/closed_graphs/FlowerGraph (five 4-port
directional couplers, C = 0.5, each neighbouring pair joined by two bonds),
used here for the classical (ray / probability-flow) dynamics instead of the
quantum spectrum.

The Perron-Frobenius operator P = |Σ_αα|² acts on probability vectors ρ
defined on the directed bonds; ρ_{n+1} = P ρ_n conserves total probability.

Pipeline (each section is guarded by ``if True`` — flip to disable):
    1. Classical summary: spectral gap, ergodicity, mixing
    2. Graph topology plots
    3. ρ after n steps, starting from one directed bond (one PDF per step)
    4. Animation of the probability flow (.mp4, needs ffmpeg; use
       writer="gif" otherwise)
"""

# ============================================================================
#  Paths & imports
# ============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

from quantum_graphs.classical_graphs import DirectionalClassicalGraphs


def directional_coupler_matrix(C):
    """4×4 scattering matrix of a lossless directional coupler (power coupling C).

    Ports (0, 1) form one arm and ports (2, 3) the other; through amplitude
    sqrt(1 - C), cross amplitude i sqrt(C), no back-reflection.
    """
    t, c = np.sqrt(1.0 - C), 1j * np.sqrt(C)
    return np.array([[0, 0, t, c],
                     [0, 0, c, t],
                     [t, c, 0, 0],
                     [c, t, 0, 0]], dtype=complex)


# ============================================================================
#  Geometry
# ============================================================================

# (x, y) in um -- for drawing only
vertex_positions = [
    (  0.000,  73.539),   # vertex 0
    (-64.244,  20.874),   # vertex 1
    (-33.058, -45.500),   # vertex 2
    ( 35.379, -48.695),   # vertex 3
    ( 78.195,  25.407),   # vertex 4
]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 4, 1, 123.830),   # petal 0-4
    (0, 2, 4, 3, 224.798),
    (0, 1, 1, 0, 112.068),   # petal 0-1
    (0, 3, 1, 2, 215.516),
    (1, 1, 2, 0, 100.260),   # petal 1-2
    (1, 3, 2, 2, 207.531),
    (2, 1, 3, 0, 92.901),   # petal 2-3
    (2, 3, 3, 2, 183.485),
    (3, 1, 4, 0, 116.961),   # petal 3-4
    (3, 3, 4, 2, 222.271),
]

C_all = 0.5


# ============================================================================
#  Build the graph
# ============================================================================

graph = DirectionalClassicalGraphs(
    vertex_positions       = vertex_positions,
    bonds                  = bonds,
    vertex_matrices        = directional_coupler_matrix(C_all),
    bond_spacing           = [40, -40, -40, -40, -40],   # drawing only
    name                   = f"FlowerGraph_classical_C{C_all:.2f}",
    base_dir               = _here,
)

print("\nGraph summary:")
print(f"  Vertices       : {graph.n_vertices}")
print(f"  Bonds          : {graph.n_bonds} ({graph.topology.n_directed} directed)")
print(f"  Save path      : {graph.save_path}\n")


# ============================================================================
#  1. Classical summary
# ============================================================================

if True:
    graph.build_perron_frobenius_operator()
    summary = graph.classical_summary()
    print("=== Classical summary ===")
    print(f"  Spectral gap        : {summary['spectral_gap']:.6f}")
    print(f"  Ergodic             : {summary['is_ergodic']}")
    print(f"  Mixing              : {summary['is_mixing']}")
    print(f"  Leading eigenvalues : {summary['eigenvalues'][:6]}")


# ============================================================================
#  2. Topology plots
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True,  save_fig=True, file_format="pdf")
    graph.plot_graph(show_lengths=False, show_port_markers=True,
                     save_fig=True, file_format="pdf")


# ============================================================================
#  3. ρ after n steps, starting on directed bond 0
# ============================================================================

rho0    = graph.basis_state(index=0)
n_steps = 15

if True:
    for i in range(n_steps + 1):
        _, fig_i, _ = graph.plot_rho_after_steps(
            rho0    = rho0,
            n_steps = i,
            title   = f"FlowerGraph — rho after {i} step{'s' if i != 1 else ''}",
            rho_color             = "#b22222",
            bond_offset_ratio_min = 0.02,
            bond_offset_ratio_max = 0.04,
            arrow_length_mode     = "constant",
            save_fig = True,
            filename = f"rho_step_{i:02d}.pdf",
        )
        plt.close(fig_i)


# ============================================================================
#  4. Animation of the probability flow
# ============================================================================

if True:
    graph.animate_rho(
        rho0      = rho0,
        n_steps   = n_steps,
        style     = "flow",
        fps       = 20,
        n_subframes_move = 24,
        n_subframes_fade = 12,
        rho_color = "#b22222",
        bond_offset_ratio_min = 0.02,
        bond_offset_ratio_max = 0.04,
        arrow_width_min   = 3.0,
        arrow_width_max   = 10.0,
        arrow_length_mode = "constant",
        rho_max_mode      = "global",
        title             = None,
        save_animation    = True,
        writer            = "mp4",          # "gif" if ffmpeg is not installed
        filename          = f"rho_animation_FlowerGraph_{n_steps}steps.mp4",
    )
    plt.close("all")

print("\nDone.")
