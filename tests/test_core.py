"""Topology construction, validation and vertex matrices."""

import numpy as np
import pytest

import quantum_graphs as qg
from quantum_graphs.core.topology import build_topology, canonical_edge
from quantum_graphs.core.vertices import directional_coupler_matrix, neumann_matrix


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_double_ring_virtualisation():
    t = build_topology(**qg.layouts.double_ring())
    # two self-loops, each split into 4 segments by 3 virtual vertices
    assert t.n_virtual_vertices == 6
    assert t.n_internal_bonds == 8 and t.n_directed_bonds == 16
    assert np.isclose(sum(t.virtual_lengths.values()), t.total_length)
    assert np.isclose(t.total_length, 2 * np.pi * (100 + 100 / np.sqrt(2)))


def test_flower_graph_virtualisation_and_indexing():
    t = build_topology(**qg.layouts.flower_graph())
    assert (t.n_bonds, t.n_virtual_vertices, t.n_directed_bonds) == (10, 10, 40)
    B = t.n_internal_bonds
    for (a, b), i in t.directed_bond_index.items():
        j = t.directed_bond_index[(b, a)]
        assert abs(i - j) == B and min(i, j) < B
        assert t.reverse_directed_bond[(a, b)] == (b, a)
    assert np.isclose(sum(t.virtual_lengths.values()), 1599.62, atol=1e-2)
    # every port of every vertex is connected exactly once in the virtual graph
    for v, ports in enumerate(t.virtual_port_map):
        for p, conn in enumerate(ports):
            assert t.virtual_port_map[conn[0]][conn[1]] == (v, p)


def test_length_keys_in_either_orientation_are_equivalent():
    layout = qg.layouts.double_ring()
    flipped = {(b, a): L for (a, b), L in layout["lengths"].items()}
    t1 = build_topology(layout["vertex_positions"], layout["port_map"], layout["lengths"])
    t2 = build_topology(layout["vertex_positions"], layout["port_map"], flipped)
    assert t1.virtual_lengths == t2.virtual_lengths


@pytest.mark.parametrize("lengths, message", [
    ({((0, 0), (0, 2)): 10.0}, "Missing bond lengths"),
    ({((0, 0), (0, 2)): 10.0, ((0, 1), (0, 3)): 5.0, ((0, 0), (0, 1)): 1.0}, "not in port_map"),
    ({((0, 0), (0, 2)): 10.0, ((0, 2), (0, 0)): 10.0, ((0, 1), (0, 3)): 5.0}, "given twice"),
    ({((0, 0), (0, 2)): -1.0, ((0, 1), (0, 3)): 5.0}, "> 0"),
])
def test_invalid_lengths_raise(lengths, message):
    with pytest.raises(ValueError, match=message):
        build_topology([(0, 0)], [[(0, 2), (0, 3), (0, 0), (0, 1)]], lengths)


@pytest.mark.parametrize("port_map, message", [
    ([[(0, 2), (0, 3), (0, 0), (0, 0)]], "not symmetric"),
    ([[(0, 0), (0, 2), (0, 1), (0, 3)]], "connected to itself"),
    ([[(0, 2), (1, 0), (0, 0), (0, 1)]], "non-existent"),
    ([[(0, 2), None, (0, 0), None]], "closed graphs"),
])
def test_invalid_port_maps_raise(port_map, message):
    with pytest.raises(ValueError, match=message):
        build_topology([(0, 0)], port_map, {((0, 0), (0, 2)): 1.0})


def test_zero_euclidean_length_requires_explicit_length():
    with pytest.raises(ValueError, match="zero Euclidean length"):
        build_topology([(0, 0)], [[(0, 1), (0, 0)]], None)


def test_leads_allowed_only_on_request():
    layout = qg.layouts.ring_with_access(n_access=2)
    t = build_topology(layout["vertex_positions"], layout["port_map"], layout["lengths"],
                       allow_leads=True)
    assert t.leads == [(0, 0), (0, 2), (1, 0), (1, 2)]


def test_remove_access_coupler_matches_the_closed_bow_tie():
    closed = qg.layouts.remove_access_coupler(qg.layouts.bow_tie_graph(open=True))
    ref = qg.layouts.bow_tie_graph()
    assert closed["port_map"] == ref["port_map"]
    canon = lambda d: {canonical_edge(*k): round(v, 6) for k, v in d.items()}
    assert canon(closed["lengths"]) == canon(ref["lengths"])


def test_vertex_labels_start_at_1_and_mark_the_access_couplers():
    def labels(layout):
        return build_topology(layout["vertex_positions"], layout["port_map"],
                              layout["lengths"], allow_leads=True).vertex_labels
    assert labels(qg.layouts.bow_tie_graph()) == {v: str(v + 1) for v in range(5)}
    # open graph: access coupler "A", the others keep their closed-graph labels
    assert labels(qg.layouts.bow_tie_graph(open=True)) == {0: "A", 1: "1", 2: "2", 3: "3",
                                                           4: "4", 5: "5"}
    assert labels(qg.layouts.ring_with_access(n_access=2)) == {0: "A1", 1: "A2"}


# ---------------------------------------------------------------------------
# Vertex matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("C", [0.0, 0.3, 0.5, 1.0])
def test_directional_coupler_is_unitary_symmetric_without_reflection(C):
    s = directional_coupler_matrix(C)
    assert np.allclose(s.conj().T @ s, np.eye(4))
    assert np.allclose(s, s.T)                       # reciprocity
    assert np.allclose(s[:2, :2], 0) and np.allclose(s[2:, 2:], 0)
    assert np.isclose(abs(s[3, 0]) ** 2, C)          # power cross-coupling


def test_coupling_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        directional_coupler_matrix(1.2)
    with pytest.raises(ValueError):
        directional_coupler_matrix(lambda k: 1.5)(4.0)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_neumann_matrix_is_unitary(n):
    s = neumann_matrix(n)
    assert np.allclose(s.conj().T @ s, np.eye(n))


def test_wrong_vertex_matrix_shape_raises(save):
    with pytest.raises(ValueError, match="shape"):
        qg.ClosedGraphs(**qg.layouts.tetrahedron(), vertex_matrices=np.eye(4), **save)


def test_non_unitary_vertex_matrix_warns(save):
    with pytest.warns(RuntimeWarning, match="not unitary"):
        qg.ClosedGraphs(**qg.layouts.tetrahedron(), vertex_matrices=0.9 * neumann_matrix(3), **save)


def test_coupling_list_and_dict(save):
    layout = qg.layouts.flower_graph()
    g1 = qg.ClosedDirectionalGraphs(**layout, vertex_couplings=[0.5, 0.3, 0.5, 0.7, 0.5], **save)
    g2 = qg.ClosedDirectionalGraphs(**layout, vertex_couplings={0: 0.5, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.5},
                                    **save)
    assert np.allclose(g1.U(4.0), g2.U(4.0))
    with pytest.raises(ValueError, match="Missing coupling"):
        qg.ClosedDirectionalGraphs(**layout, vertex_couplings={0: 0.5}, **save)
