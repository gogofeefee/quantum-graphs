"""Entropy / IPR: per physical bond, basis-independent for degenerate levels."""

import numpy as np
import pytest

import quantum_graphs as qg
from quantum_graphs.stats import eigenspace_intensity, localization_measures


def test_measures_of_uniform_and_single_bond_distributions():
    S, S_norm, ipr = localization_measures(np.ones(7))
    assert np.isclose(S, np.log(7)) and np.isclose(S_norm, 1.0) and np.isclose(ipr, 1.0)
    S, S_norm, ipr = localization_measures([0, 0, 3.0, 0])
    assert S == 0.0 and S_norm == 0.0 and np.isclose(ipr, 4.0)
    assert np.isnan(localization_measures([2.0])[1])                 # S_norm of one bond
    assert np.all(np.isnan(localization_measures(np.zeros(3))))


def test_eigenspace_intensity_does_not_depend_on_the_basis():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3)))
    vecs = q.T                                                       # 3 orthonormal rows
    u, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    assert np.allclose(eigenspace_intensity(vecs), eigenspace_intensity(u @ vecs))
    assert np.allclose(eigenspace_intensity(vecs[0]), np.abs(vecs[0]) ** 2)


@pytest.mark.parametrize("layout", ["single_ring", "double_ring", "flower_graph", "bow_tie_graph"])
def test_physical_bond_arrivals_end_at_the_real_ports(layout, save):
    g = qg.ClosedDirectionalGraphs(**getattr(qg.layouts, layout)(), vertex_couplings=0.5,
                                   neff=3.97, wavelength_range=(1.50, 1.51), **save) \
        if layout != "single_ring" else \
        qg.ClosedGraphs(**qg.layouts.single_ring(), vertex_matrices=qg.PASS_THROUGH, neff=2.41,
                        wavelength_range=(1.50, 1.51), **save)
    bond_of = {idx: db for db, idx in g.directed_bond_index.items()}
    assert g.physical_bond_arrivals.shape == (g.n_bonds, 2)
    for (i, pi, j, pj), (d_fwd, d_bwd) in zip(g.bonds, g.physical_bond_arrivals):
        assert bond_of[d_fwd][1] == (j, pj)
        assert bond_of[d_bwd][1] == (i, pi)
        if (min((i, pi), (j, pj)), max((i, pi), (j, pj))) in g.updated_bonds:   # not split
            assert g.reverse_directed_bond[bond_of[d_fwd]] == bond_of[d_bwd]


def segment_to_physical_bond(g):
    """Physical bond (index in ``g.bonds``) of each virtualised bond, by walking
    each bond from its first real end through the virtual vertices."""
    out = np.full(g.n_internal_bonds, -1)
    for n, (i, pi, _, _) in enumerate(g.bonds):
        end = (i, pi)
        while True:
            seg = next(idx for idx, (x, y) in enumerate(g.updated_bonds) if end in (x, y))
            out[seg] = n
            x, y = g.updated_bonds[seg]
            other = y if end == x else x
            if other[0] not in g.virtual_vertices:
                break
            end = (other[0], 1 - other[1])                # through the pass-through vertex
    return out


def test_closed_eigenmode_has_one_intensity_per_physical_bond(save):
    """Real k: all segments of a split bond carry the same intensity, which the
    physical-bond intensity reproduces; S_norm uses ln(physical N_b)."""
    g = qg.ClosedDirectionalGraphs(**qg.layouts.bow_tie_graph(), vertex_couplings=0.5,
                                   neff=3.97, wavelength_range=(1.550, 1.552), **save)
    assert g.n_internal_bonds > g.n_bonds                             # some bonds are split
    phys = segment_to_physical_bond(g)
    assert np.all(phys >= 0)
    spectrum, _ = g.compute_spectrum(N_scan=20_000, save_spectrum=False)
    B = g.n_internal_bonds
    for k in spectrum[:4]:
        a = g.compute_wavefunction(k)
        seg = np.abs(a[0, :B]) ** 2 + np.abs(a[0, B:]) ** 2
        I_b = g.bond_intensities(a)
        assert np.allclose(seg, I_b[phys], rtol=1e-6, atol=0)
        p = I_b / I_b.sum()
        S, S_norm, ipr = g.localization_measures(a)
        assert np.isclose(S_norm, -np.sum(p * np.log(p)) / np.log(g.n_bonds))
        assert np.isclose(ipr, g.n_bonds * np.sum(p ** 2))


def test_flower_graph_entropy_is_the_half_bond_entropy_minus_ln2(save):
    """Every bond of the closed flower graph is split in two equal halves."""
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), vertex_couplings=0.5,
                                   neff=3.97, wavelength_range=(1.550, 1.552), **save)
    assert g.n_internal_bonds == 2 * g.n_bonds
    spectrum, _ = g.compute_spectrum(N_scan=20_000, save_spectrum=False)
    B = g.n_internal_bonds
    for k in spectrum[:3]:
        a = g.compute_wavefunction(k)[0]
        S_half = localization_measures(np.abs(a[:B]) ** 2 + np.abs(a[B:]) ** 2)[0]
        assert np.isclose(g.localization_measures(a)[0], S_half - np.log(2))


def test_degenerate_level_measures_do_not_depend_on_the_basis(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.double_ring(), vertex_couplings=0.5, neff=1.0,
                                   wavelength_range=(1.48, 1.52), **save)
    spectrum, deg = g.compute_spectrum(N_scan=20_000, save_spectrum=False)
    rng = np.random.default_rng(1)
    for k in spectrum[:3]:
        a = g.compute_wavefunction(k)
        assert a.shape[0] == 2
        u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        assert np.allclose(g.localization_measures(a), g.localization_measures(u @ a))
    g.spectrum = spectrum
    _, _, S_norm, ipr = g._compute_mode_quantities()
    assert S_norm.size == spectrum.size                               # one value per level


def test_split_bond_intensity_is_the_same_for_two_virtualisations(save):
    """Complex k (lossy, driven): the amplitudes arriving at the real ends of a
    bond do not depend on where it is split. Ring with one access coupler: the
    loop is split in four by the virtual vertices; the same ring with an
    explicit pass-through vertex half way is split differently."""
    C = 0.3
    t, c = np.sqrt(1 - C), 1j * np.sqrt(C)
    coupler = np.array([[0, 0, t, c], [0, 0, c, t], [t, c, 0, 0], [c, t, 0, 0]])
    L = 2 * np.pi * 100.0
    kw = dict(neff=2.41, wavelength_range=(1.50, 1.51), **save)
    a = qg.OpenGraphs(**{k: v for k, v in qg.layouts.ring_with_access().items()
                         if k != "lead_orientations"},
                      vertex_matrices={0: coupler}, **kw)
    b = qg.OpenGraphs(vertex_positions=[(0.0, 0.0), (0.0, 200.0)],
                      port_map=[[None, (1, 0), None, (1, 1)], [(0, 1), (0, 3)]],
                      lengths={((0, 1), (1, 0)): 0.5 * L, ((0, 3), (1, 1)): 0.5 * L},
                      vertex_matrices={0: coupler, 1: qg.PASS_THROUGH}, **kw)
    k = 4.1 - 2e-4j
    amp_a = a.model.internal_amplitudes(k, [1.0, 0.0])
    amp_b = b.model.internal_amplitudes(k, [1.0, 0.0])
    # bond (0,1)-(0,3) of `a` is the path (0,1)-(1,0)-(1,1)-(0,3) of `b`
    arriving_b = {db[1]: idx for db, idx in b.directed_bond_index.items()}
    expected = abs(amp_b[arriving_b[(0, 3)]]) ** 2 + abs(amp_b[arriving_b[(0, 1)]]) ** 2
    assert np.isclose(a.bond_intensities(amp_a)[0], expected, rtol=1e-10)
    # and the four quarter-segments of `a` do differ at complex k
    seg = np.abs(amp_a[:a.n_internal_bonds]) ** 2 + np.abs(amp_a[a.n_internal_bonds:]) ** 2
    assert np.ptp(seg) > 1e-3 * seg.max()
