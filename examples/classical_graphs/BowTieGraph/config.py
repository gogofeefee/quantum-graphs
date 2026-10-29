#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BowTieGraph — classical Perron-Frobenius dynamics
=================================================

Same closed graph as examples/closed_graphs/BowTieGraph (five 4-port
directional couplers, C = 0.5, 10 bonds, two of them doubled),
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


# ============================================================================
#  Geometry
# ============================================================================

# (x, y) in um -- for drawing only
vertex_positions = [
    (  0.000,   0.000),   # vertex 0
    ( 83.283,  48.083),   # vertex 1
    (-76.500,  44.167),   # vertex 2
    (-63.693, -36.773),   # vertex 3
    ( 68.166, -39.356),   # vertex 4
]

# Bonds (i, p, j, q, L): port p of vertex i <-> port q of vertex j, length L (um).
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
bonds = [
    (0, 0, 3, 2, 79.153),
    (0, 1, 2, 3, 95.085),
    (0, 2, 4, 0, 84.717),
    (0, 3, 1, 1, 103.523),
    (1, 0, 4, 1, 96.144),   # double bond 1-4
    (1, 2, 4, 3, 262.852),
    (1, 3, 2, 1, 279.690),
    (2, 0, 3, 1, 255.817),   # double bond 2-3
    (2, 2, 3, 3, 88.833),
    (3, 0, 4, 2, 251.257),
]

C_all = 0.5


# ============================================================================
#  Build the graph
# ============================================================================

graph = DirectionalClassicalGraphs(
    vertex_positions       = vertex_positions,
    bonds                  = bonds,
    couplings              = C_all,
    name                   = f"BowTieGraph_classical_C{C_all:.2f}",
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
    graph.perron_frobenius()
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
            title   = f"BowTieGraph — rho after {i} step{'s' if i != 1 else ''}",
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
        filename          = f"rho_animation_BowTieGraph_{n_steps}steps.mp4",
    )
    plt.close("all")

print("\nDone.")
