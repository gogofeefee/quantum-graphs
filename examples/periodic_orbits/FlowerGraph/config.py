#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowerGraph — periodic-orbit enumeration
========================================

Same closed graph as examples/closed_graphs/FlowerGraph (five 4-port
directional couplers, each neighbouring pair joined by two bonds).

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
    (  0.000,  73.539),   # vertex 0
    (-64.244,  20.874),   # vertex 1
    (-33.058, -45.500),   # vertex 2
    ( 35.379, -48.695),   # vertex 3
    ( 78.195,  25.407),   # vertex 4
]

# port_map[i][p] = (j, q): port p of vertex i <-> port q of vertex j.
# Ports (0, 1) are one arm of the coupler, ports (2, 3) the other arm.
port_map = [
    [(4, 1), (1, 0), (4, 3), (1, 2)],   # vertex 0
    [(0, 1), (2, 0), (0, 3), (2, 2)],   # vertex 1
    [(1, 1), (3, 0), (1, 3), (3, 2)],   # vertex 2
    [(2, 1), (4, 0), (2, 3), (4, 2)],   # vertex 3
    [(3, 1), (0, 0), (3, 3), (0, 2)],   # vertex 4
]

# Bond lengths in um (short / long bond of each petal)
lengths = {
    ((0, 0), (4, 1)): 123.830,   # petal 0-4
    ((0, 2), (4, 3)): 224.798,
    ((0, 1), (1, 0)): 112.068,   # petal 0-1
    ((0, 3), (1, 2)): 215.516,
    ((1, 1), (2, 0)): 100.260,   # petal 1-2
    ((1, 3), (2, 2)): 207.531,
    ((2, 1), (3, 0)):  92.901,   # petal 2-3
    ((2, 3), (3, 2)): 183.485,
    ((3, 1), (4, 0)): 116.961,   # petal 3-4
    ((3, 3), (4, 2)): 222.271,
}

ng = 3.97   # group index: optical length = ng × physical length


# ============================================================================
#  Build the graph
# ============================================================================

graph = PeriodicOrbits(
    vertex_positions       = vertex_positions,
    port_map               = port_map,
    lengths                = lengths,
    ng                     = ng,
    virtual_vertex_spacing = [40, -40, -40, -40, -40],   # drawing only
    name                   = f"FlowerGraph_periodic_orbits_ng{ng:.2f}",
    base_dir               = _here,
)

print("\nGraph summary:")
print(f"  Vertices       : {graph.n_vertices} real + {graph.n_virtual_vertices} virtual")
print(f"  Directed bonds : {graph.n_directed_bonds}")
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
