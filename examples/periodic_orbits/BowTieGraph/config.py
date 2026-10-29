#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BowTieGraph — periodic-orbit enumeration
========================================

Same closed graph as examples/closed_graphs/BowTieGraph (five 4-port
directional couplers, 10 bonds, two of them doubled).

Every real vertex is a directional coupler: a ray entering on one arm
(ports 0, 1) leaves on the other arm (ports 2, 3) and vice versa, so the
periodic orbits follow from the topology and the bond lengths alone (no
coupling constant needed). Optical lengths are physical lengths × ng.

Pipeline (each section is guarded by ``if True`` — flip to disable):
    1. Graph topology plots
    2. Irreducible periodic orbits (no vertex visited twice), exported to CSV
    3. All periodic orbits up to a physical length L_max, exported to CSV
    4. Plots of the shortest irreducible orbits
"""

# ============================================================================
#  Paths & imports
# ============================================================================

import os
import sys
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

from quantum_graphs.periodic_orbits import PeriodicOrbits


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

ng = 3.97   # group index: optical length = ng × physical length


# ============================================================================
#  Build the graph
# ============================================================================

graph = PeriodicOrbits(
    vertex_positions       = vertex_positions,
    bonds                  = bonds,
    ng                     = ng,
    name                   = f"BowTieGraph_periodic_orbits_ng{ng:.2f}",
    base_dir               = _here,
)

print("\nGraph summary:")
print(f"  Vertices       : {graph.n_vertices}")
print(f"  Bonds          : {graph.n_bonds} ({graph.topology.n_directed} directed)")
print(f"  Save path      : {graph.save_path}\n")


# ============================================================================
#  1. Topology plots
# ============================================================================

if True:
    graph.plot_graph(show_lengths=True,  save_fig=True)
    graph.plot_graph(show_lengths=False, save_fig=True)


# ============================================================================
#  2. Irreducible periodic orbits
# ============================================================================

if True:
    irreducible = graph.find_irreducible_orbits()
    graph.export_orbits_csv(which="irreducible")


# ============================================================================
#  3. All periodic orbits (primitive or not) up to L_max
# ============================================================================

if True:
    L_max_um = 2000.0
    all_orbits = graph.find_orbits_up_to_length(L_max_um=L_max_um, max_orbits=20_000)
    graph.export_orbits_csv(which="all")


# ============================================================================
#  4. Plot the shortest irreducible orbits
# ============================================================================

if True:
    shortest = sorted(irreducible.values(), key=lambda r: r["physical_length_um"])[:6]
    for record in shortest:
        print(f"  {record['name']:>20s}   L_phys = {record['physical_length_um']:8.2f} um"
              f"   L_opt = {record['optical_length_um']:8.2f} um")
        graph.plot_orbit(record["name"], color="#7a1f1f", show_port_markers=True,
                         save_fig=True)
        plt.close("all")

print("\nDone.")
