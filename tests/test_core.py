"""Topology construction, validation and vertex matrices."""

import numpy as np
import pytest

import quantum_graphs as qg
from quantum_graphs.core.topology import build_topology
from quantum_graphs.core.vertices import directional_coupler_matrix, neumann_matrix


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_double_ring_has_two_self_loops_and_no_extra_vertex():
    t = build_topology(**qg.layouts.double_ring())
    assert (t.n_vertices, t.n_bonds, t.n_directed) == (1, 2, 4)
    assert np.isclose(t.total_length, 2 * np.pi * (100 + 100 / np.sqrt(2)))
    assert list(t.valency) == [4]


def test_flower_graph_directed_bond_indexing():
    t = build_topology(**qg.layouts.flower_graph())
    assert (t.n_bonds, t.n_directed) == (10, 20)
    B = t.n_bonds
    for d in range(t.n_directed):
        r = t.reverse(d)
        assert abs(r - d) == B and t.reverse(r) == d
        assert (t.src_vertex[r], t.src_port[r]) == (t.dst_vertex[d], t.dst_port[d])
        assert t.bond_of(d) == t.bond_of(r) == d % B and t.is_forward(d) == (d < B)
        assert t.directed_length[d] == t.directed_length[r] == t.length[t.bond_of(d)]
        assert t.directed(t.src_vertex[d], t.src_port[d], t.dst_vertex[d], t.dst_port[d]) == d
        assert t.leaving(t.src_vertex[d], t.src_port[d]) == d
        assert t.arriving(t.dst_vertex[d], t.dst_port[d]) == d
    assert np.isclose(t.total_length, 1599.62, atol=1e-2)
    # five pairs of parallel bonds
    assert sorted(len(g) for g in t.parallel_groups().values()) == [2] * 5


def test_bond_without_length_uses_the_euclidean_distance():
    t = build_topology([(0, 0), (3, 4)], [(0, 0, 1, 0), (0, 1, 1, 1, 7.0)])
    assert np.allclose(t.length, [5.0, 7.0])
    assert t.bonds[0] == (0, 0, 1, 0, 5.0)


@pytest.mark.parametrize("bonds, leads, message", [
    ([(0, 0, 0, 2, 1.0), (0, 1, 0, 2, 1.0)], (), "used twice"),
    ([(0, 0, 0, 0, 1.0)], (), "to itself"),
    ([(0, 0, 0, 2, 1.0), (0, 1, 0, 3, 1.0)], [(0, 1)], "used twice"),
    ([(0, 0, 0, 2, 1.0), (0, 1, 1, 3, 1.0)], (), "no such vertex"),
    ([(0, 0, 0, 3, 1.0)], (), r"ports \[1, 2\] are not connected"),
    ([(0, 0, 0, 2, -1.0), (0, 1, 0, 3, 1.0)], (), "> 0"),
    ([(0, 0, 0, 2)], (), "zero Euclidean length"),
    ([(0, 0, 0)], (), "must be"),
])
def test_invalid_bonds_raise(bonds, leads, message):
    with pytest.raises(ValueError, match=message):
        build_topology([(0, 0)], bonds, leads)


def test_leads_allowed_only_on_request():
    layout = qg.layouts.ring_with_access(n_access=2)
    t = build_topology(layout["vertex_positions"], layout["bonds"], layout["leads"])
    assert t.leads == ((0, 0), (0, 2), (1, 0), (1, 2))
    assert t.lead_index(1, 0) == 2 and t.leaving(1, 0) is None
    with pytest.raises(ValueError, match="closed graphs"):
        build_topology(layout["vertex_positions"], layout["bonds"], layout["leads"],
                       allow_leads=False)


def test_remove_access_coupler_matches_the_closed_bow_tie():
    closed = qg.layouts.remove_access_coupler(qg.layouts.bow_tie_graph(open=True))
    ref = qg.layouts.bow_tie_graph()
    assert closed["vertex_positions"] == ref["vertex_positions"]
    assert [b[:4] for b in closed["bonds"]] == [b[:4] for b in ref["bonds"]]
    assert np.allclose([b[4] for b in closed["bonds"]], [b[4] for b in ref["bonds"]])
    assert "leads" not in closed


def test_as_dict_rebuilds_the_same_topology():
    t = build_topology(**qg.layouts.closed(qg.layouts.flower_graph(open=True)),
                       leads=[(0, 0), (0, 2)])
    t2 = build_topology(**t.as_dict())
    assert t2.bonds == t.bonds and t2.leads == t.leads
    assert np.array_equal(t2.positions, t.positions)


def test_vertex_labels_start_at_1_and_mark_the_access_couplers():
    def labels(layout):
        return build_topology(layout["vertex_positions"], layout["bonds"],
                              layout.get("leads", ())).vertex_labels
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
