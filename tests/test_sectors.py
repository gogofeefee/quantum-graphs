"""Sectors of the bond dynamics and the spectrum used by the statistics."""

import numpy as np
import pytest

import quantum_graphs as qg
from quantum_graphs.core.vertices import neumann_matrix


def test_double_ring_has_two_partner_sectors(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.double_ring(), couplings=0.5, neff=1.0,
                                   wavelength_range=(1.48, 1.64), **save)
    s0, s1 = g.sectors
    assert (s0.reverse, s1.reverse) == (1, 0)
    assert [s.index for s in g.sectors.independent] == [0]
    assert g.sectors.multiplicity(1) == 2
    assert np.isclose(s0.length, g.total_length) and np.isclose(s1.length, g.total_length)
    assert g.spectral_ensemble() == "GUE"
    # the partner has the same spectrum: same eigenvalues of U restricted to it
    k = 4.1
    e0, e1 = (np.sort_complex(np.linalg.eigvals(g.model.sector(s.directed).U(k))) for s in (s0, s1))
    assert np.allclose(e0, e1, atol=1e-12)


@pytest.mark.parametrize("layout", ["flower_graph", "bow_tie_graph"])
def test_flower_and_bow_tie_are_one_time_reversal_sector(layout, save):
    g = qg.ClosedDirectionalGraphs(**getattr(qg.layouts, layout)(), couplings=0.5, neff=3.97,
                                   wavelength_range=(1.50, 1.53), **save)
    (s,) = g.sectors
    assert s.reverse == 0 and s.time_reversal and s.ensemble == "GOE"
    assert s.directed.size == 2 * g.n_bonds and np.isclose(s.length, 2 * g.total_length)


def test_desymmetrised_spectrum_and_weyl_law(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.double_ring(), couplings=0.5, neff=1.0,
                                   wavelength_range=(1.48, 1.64), **save)
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    assert np.all(deg == 2) and np.all(g.level_sectors == 0)
    assert np.array_equal(g.levels(), spectrum)               # one level per pair
    assert np.array_equal(g.levels("all"), np.repeat(spectrum, 2))
    assert np.array_equal(g.levels(1), spectrum)               # the partner's class
    assert np.isclose(g.weyl_length(), g.total_length)
    assert np.isclose(g.weyl_length("all"), 2 * g.total_length)
    assert abs(spectrum.size - g.weyl_count()) < 3
    x = g.unfolded_spectrum()
    assert abs(np.mean(np.diff(x)) - 1) < 0.02
    # an explicit spectrum (e.g. measured peaks, one per pair) unfolds the same way
    assert np.allclose(g.unfolded_spectrum(spectrum=spectrum), x)
    L, s2 = g.sigma2(3.0, 0.5)
    assert L.size == s2.size == 7


def test_non_reciprocal_partners_are_independent(save):
    # a ring through a non-reciprocal 2-port vertex: the two directions acquire
    # different phases, so the reversed sectors do not share their spectrum
    sigma = np.array([[0, 1], [np.exp(0.7j), 0]])
    g = qg.ClosedGraphs([(0.0, 0.0)], [(0, 0, 0, 1, 500.0)], vertex_matrices={0: sigma},
                        neff=2.0, wavelength_range=(1.50, 1.60), **save)
    assert len(g.sectors.independent) == 2
    assert all(s.ensemble == "GUE" for s in g.sectors)
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    assert np.all(deg == 1) and set(g.level_sectors) == {0, 1}
    with pytest.warns(RuntimeWarning, match="superpose 2 sector spectra"):
        g.unfolded_spectrum()
    assert np.all(np.isin(g.levels(0), spectrum)) and g.levels(0).size < spectrum.size


def test_degeneracy_inside_a_sector_is_reported(save):
    # two identical loops on a Neumann vertex: geometric symmetry, one sector
    g = qg.ClosedGraphs([(0.0, 0.0)], [(0, 0, 0, 1, 300.0), (0, 2, 0, 3, 300.0)],
                        vertex_matrices={0: neumann_matrix(4)}, neff=2.0,
                        wavelength_range=(1.50, 1.60), **save)
    assert len(g.sectors) == 1 and g.sectors[0].time_reversal
    with pytest.warns(RuntimeWarning, match="degenerate inside their sector"):
        g.compute_spectrum(save_spectrum=False)


def test_resonances_are_assigned_to_sectors(save):
    g = qg.OpenDirectionalGraphs(**qg.layouts.ring_with_access(), couplings=0.5, access_couplings=0.3,
                                 neff=2.41, wavelength_range=(1.54, 1.56), **save)
    assert len(g.sectors) == 2 and len(g.sectors.independent) == 1
    spectrum, deg = g.find_resonances(im_min=-3e-4, im_max=0.0)
    assert spectrum.size and np.all(deg == 2)
    vecs = g.resonance_vectors(0)
    assert vecs.shape[0] == 2
    assert np.allclose(vecs @ g.U(spectrum[0]).T, vecs, atol=1e-4)
