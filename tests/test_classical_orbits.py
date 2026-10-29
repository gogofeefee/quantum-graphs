"""Classical Perron-Frobenius dynamics and periodic orbits."""

import numpy as np
import pytest

import quantum_graphs as qg


def classical(layout, save):
    return qg.DirectionalClassicalGraphs(**layout, couplings=0.5, **save)


def test_perron_frobenius_is_doubly_stochastic(save):
    g = classical(qg.layouts.flower_graph(), save)
    P = g.perron_frobenius()
    assert np.allclose(P.sum(axis=0), 1) and np.allclose(P.sum(axis=1), 1)
    rho = g.evolve_rho(g.basis_state(index=0), n_steps=11)
    assert np.isclose(rho.sum(), 1.0)
    assert np.allclose(g.invariant_measure(), 1.0 / g.topology.n_directed)


def test_ergodicity_and_mixing(save):
    fg, btg = classical(qg.layouts.flower_graph(), save), classical(qg.layouts.bow_tie_graph(), save)
    # one step = one bond traversal: FG has eigenvalue -1 (period two), BTG a gap
    # 1 - 1/sqrt(2) ≈ 0.29 (the values of the paper's supplementary material)
    assert fg.is_ergodic() and not fg.is_mixing()
    assert np.any(np.isclose(fg.perron_frobenius_spectrum()[0], -1.0))
    assert btg.is_ergodic() and btg.is_mixing()
    assert np.isclose(btg.spectral_gap(), 1 - 1 / np.sqrt(2), atol=1e-12)


def _brute_force_orbits(po, L_max):
    t = po.topology
    T, Ld, found = po.bond_transitions, t.directed_length, set()

    def dfs(start, cur, path, length):
        for nxt in T[cur]:
            if nxt == start:
                found.add(po._bracelet_min(tuple(path), t.reverse))
            if length + Ld[nxt] <= L_max:
                path.append(nxt)
                dfs(start, nxt, path, length + Ld[nxt])
                path.pop()
    for s in range(t.n_directed):
        if Ld[s] <= L_max:
            dfs(s, s, [s], Ld[s])
    return found


@pytest.mark.parametrize("layout", [qg.layouts.flower_graph(), qg.layouts.bow_tie_graph()])
def test_orbit_enumeration_matches_brute_force(layout, save):
    po = qg.PeriodicOrbits(**layout, ng=3.97, **save)
    orbits = po.find_orbits_up_to_length(1300.0, max_orbits=10**6, max_search_nodes=10**8)
    keys = {po._bracelet_min(r["path"], po.topology.reverse) for r in orbits.values()}
    assert keys == _brute_force_orbits(po, 1300.0) and po.orbits_complete
    for r in orbits.values():
        assert np.isclose(r["optical_length_um"], 3.97 * r["physical_length_um"])


def test_irreducible_orbits(save):
    po = qg.PeriodicOrbits(**qg.layouts.bow_tie_graph(), ng=1.0, **save)
    irr = po.find_irreducible_orbits()
    assert sorted(irr) == ["1-2-3", "1-4-5", "2-3-4-5", "2-5", "3-4"]


def test_truncated_orbit_search_warns(save):
    po = qg.PeriodicOrbits(**qg.layouts.flower_graph(), ng=1.0, **save)
    with pytest.warns(RuntimeWarning, match="INCOMPLETE"):
        po.find_orbits_up_to_length(3000.0, max_orbits=50)
    assert not po.orbits_complete
