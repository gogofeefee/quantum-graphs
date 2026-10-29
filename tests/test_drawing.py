"""Bond drawing paths (quantum_graphs.plotting.drawing)."""

import numpy as np

import quantum_graphs as qg
from quantum_graphs.core.topology import build_topology
from quantum_graphs.plotting.drawing import BondPaths


def test_curves_join_their_end_vertices():
    t = build_topology(**qg.layouts.bow_tie_graph())
    paths = BondPaths(t)
    for b, (i, p, j, q, L) in enumerate(t.bonds):
        c = paths.curve(b)
        assert np.allclose(c[0], t.positions[i]) and np.allclose(c[-1], t.positions[j])
        assert np.allclose(paths.directed_point(b + t.n_bonds, 0.0), t.positions[j])


def test_parallel_bonds_are_symmetric_arcs_through_the_old_virtual_vertices():
    t = build_topology(**qg.layouts.flower_graph())
    paths = BondPaths(t)
    for (i, j), group in t.parallel_groups().items():
        assert len(group) == 2
        mid = 0.5 * (t.positions[i] + t.positions[j])
        dist = np.hypot(*(t.positions[j] - t.positions[i]))
        offsets = [np.hypot(*(paths.point(b, 0.5) - mid)) for b in group]
        # middle of each arc 0.15 x the vertex distance away from the chord (default spacing)
        assert np.allclose(offsets, 0.15 * dist, rtol=1e-3)
        a, b = (paths.point(b, 0.5) - mid for b in group)
        assert np.allclose(a, -b, atol=1e-9 * dist)


def test_self_loops_are_drawn_with_their_length_and_point_apart():
    t = build_topology(**qg.layouts.double_ring())
    paths = BondPaths(t)
    for b in range(t.n_bonds):
        assert np.isclose(paths.drawn_length(b), t.length[b], rtol=1e-3)
    tips = [paths.point(b, 0.5) - t.positions[0] for b in range(2)]
    assert np.dot(tips[0], tips[1]) < 0                       # opposite directions


def test_arc_length_parametrisation_and_tangents():
    t = build_topology(**qg.layouts.bow_tie_graph())
    paths = BondPaths(t)
    b = next(g[0] for g in t.parallel_groups().values() if len(g) == 2)
    s = np.linspace(0, 1, 11)
    pts = paths.point(b, s)
    steps = np.hypot(*np.diff(pts, axis=0).T)
    assert np.allclose(steps, steps.mean(), rtol=2e-2)       # equal arc-length steps
    d = paths.directed_tangent(b, 0.5)[0]
    assert np.isclose(np.hypot(*d), 1.0)
    assert np.allclose(paths.directed_tangent(b + t.n_bonds, 0.5)[0], -d, atol=1e-6)
