"""Entropy / IPR: per bond, basis-independent for degenerate levels."""

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


def test_closed_eigenmode_has_one_intensity_per_bond(save):
    """I_b = |a_b|² + |a_{b+B}|² (the two waves arriving at the ends of bond b),
    S_norm and IPR over the N_b bonds."""
    g = qg.ClosedDirectionalGraphs(**qg.layouts.bow_tie_graph(), vertex_couplings=0.5,
                                   neff=3.97, wavelength_range=(1.550, 1.552), **save)
    spectrum, _ = g.compute_spectrum(N_scan=20_000, save_spectrum=False)
    B = g.n_bonds
    for k in spectrum[:4]:
        a = g.compute_wavefunction(k)
        I_b = g.bond_intensities(a)
        assert np.allclose(I_b, np.abs(a[0, :B]) ** 2 + np.abs(a[0, B:]) ** 2)
        p = I_b / I_b.sum()
        S, S_norm, ipr = g.localization_measures(a)
        assert np.isclose(S_norm, -np.sum(p * np.log(p)) / np.log(B))
        assert np.isclose(ipr, B * np.sum(p ** 2))


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


def test_bond_intensity_does_not_depend_on_an_extra_pass_through_vertex(save):
    """Complex k (lossy, driven): a ring drawn as one self-loop, or as two bonds
    joined by a pass-through vertex half way, carries the same intensity at the
    ports of the access coupler."""
    C = 0.3
    t, c = np.sqrt(1 - C), 1j * np.sqrt(C)
    coupler = np.array([[0, 0, t, c], [0, 0, c, t], [t, c, 0, 0], [c, t, 0, 0]])
    L = 2 * np.pi * 100.0
    kw = dict(neff=2.41, wavelength_range=(1.50, 1.51), **save)
    a = qg.OpenGraphs(**qg.layouts.ring_with_access(), vertex_matrices={0: coupler}, **kw)
    b = qg.OpenGraphs(vertex_positions=[(0.0, 0.0), (0.0, 200.0)],
                      bonds=[(0, 1, 1, 0, 0.5 * L), (0, 3, 1, 1, 0.5 * L)], leads=[(0, 0), (0, 2)],
                      vertex_matrices={0: coupler, 1: qg.PASS_THROUGH}, **kw)
    k = 4.1 - 2e-4j
    amp_a = a.model.internal_amplitudes(k, [1.0, 0.0])
    amp_b = b.model.internal_amplitudes(k, [1.0, 0.0])
    # bond (0,1)-(0,3) of `a` is the path (0,1)-(1,0)-(1,1)-(0,3) of `b`
    tb = b.topology
    expected = abs(amp_b[tb.arriving(0, 3)]) ** 2 + abs(amp_b[tb.arriving(0, 1)]) ** 2
    assert np.isclose(a.bond_intensities(amp_a)[0], expected, rtol=1e-10)
    assert np.allclose(a.compute_S_matrix(k), b.compute_S_matrix(k))
