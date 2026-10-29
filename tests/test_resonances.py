"""Resonance search: rectangle, argument-principle count, completeness and Weyl checks."""

import warnings

import numpy as np
import pytest

import quantum_graphs as qg
from quantum_graphs.core import PASS_THROUGH, directional_coupler_matrix
from quantum_graphs.core.argument_principle import count_zeros

C_EXT, NEFF, R = 0.3, 2.41, 100.0


def ring1(save):
    return qg.OpenDirectionalGraphs(**qg.layouts.ring_with_access(R, 1), access_couplings=C_EXT,
                                    couplings=0.5, neff=NEFF, wavelength_range=(1.54, 1.56), **save)


def flower(save, neff=3.97, wavelength_range=(1.550, 1.552)):
    return qg.OpenDirectionalGraphs(**qg.layouts.flower_graph(open=True), couplings=0.5,
                                    access_couplings=C_EXT, neff=neff,
                                    wavelength_range=wavelength_range, **save)


def no_warning(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return func(*args, **kwargs)


def test_count_matches_the_analytic_ring_poles(save):
    """Poles k_m = (2π m + i ln t) / (n L), each doubly degenerate (two partner sectors)."""
    g = ring1(save)
    k_min, k_max = g._k_range()
    t, L = np.sqrt(1 - C_EXT), 2 * np.pi * R
    m = np.arange(np.ceil(k_min * NEFF * L / (2 * np.pi)), np.floor(k_max * NEFF * L / (2 * np.pi)) + 1)
    assert np.log(t) / (NEFF * L) > -3e-4
    count = g.count_resonances(-3e-4, 0.0)
    assert count.total == 2 * m.size and count.defect < 1e-6
    assert g.count_resonances(-1e-4, 0.0).total == 0
    # double zeros 2e-6 above the lower edge: the phase turns by 2π between two
    # coarse samples, only the log-derivative criterion resolves it
    assert g.count_resonances(-1.2e-4, 0.0).total == 2 * m.size
    assert g.count_resonances(-3e-4, 0.0, sector=0).total == m.size
    cuts = 2 * np.pi * (m[:-1] + 0.5) / (NEFF * L)           # one pole per sub-window
    assert np.all(g.count_resonances(-3e-4, 0.0, sector=0, cuts=cuts).counts == 1)


def test_count_raises_when_a_zero_is_on_the_contour(save):
    g = ring1(save)
    t, L = np.sqrt(1 - C_EXT), 2 * np.pi * R
    im = np.log(t) / (NEFF * L)
    with pytest.raises(RuntimeError, match="contour"):
        count_zeros(g.model.sector(g.sectors[0].directed), 4.0, 4.1, im, 0.0)


def test_refinement_stays_in_the_rectangle(save):
    """The ring resonances lie at Im k = −1.18e-4: none may be returned with im_min = −1e-4
    (the unbounded Nelder-Mead stage used to return them, and others far outside)."""
    g = ring1(save)
    with pytest.warns(RuntimeWarning, match="just below the search rectangle"):
        spectrum, _ = g.find_resonances(N_re=40_000, N_im=8, im_min=-1e-4, im_max=0.0)
    assert spectrum.size == 0


def test_complete_search_passes_every_check(save):
    g = flower(save)
    spectrum, deg = no_warning(g.find_resonances, N_re=4_000, N_im=8, im_min=-2e-4, im_max=0.0)
    count = g.resonance_counts[0]
    assert count.total == deg.sum() == spectrum.size >= 10
    k_min, k_max = g._k_range()
    assert np.all((spectrum.real >= k_min) & (spectrum.real <= k_max))
    assert np.all((spectrum.imag >= -2e-4) & (spectrum.imag <= 0))


def test_missed_resonances_are_reported_by_sub_window(save):
    wr = (1.55, 1.56)
    g = flower(save, wavelength_range=wr)
    with pytest.warns(RuntimeWarning, match=r"counts 51 in the rectangle; 3 sub-window\(s\) differ"):
        spectrum, _ = g.find_resonances(N_re=60, N_im=3, im_min=-1e-4, im_max=0.0)
    assert spectrum.size < 51 and g.resonance_counts[0].edges.size == 4
    # the warning comes back when the cached spectrum is loaded
    with pytest.warns(RuntimeWarning, match="argument principle counts 51"):
        flower(save, wavelength_range=wr).find_resonances(N_re=60, N_im=3, im_min=-1e-4, im_max=0.0)


def test_shallow_rectangle_is_reported_by_the_weyl_law(save):
    g = flower(save, wavelength_range=(1.550, 1.555))
    assert g.weyl_count() > 20
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.find_resonances(N_re=2_000, N_im=4, im_min=-1e-5, im_max=0.0)
    messages = [str(w.message) for w in caught]
    assert any("Weyl law predicts" in m for m in messages)
    assert any("just below the search rectangle" in m for m in messages)


def test_real_roots_on_the_upper_edge_are_kept_and_counted(save):
    """An isolated loop carries real resonances (bound states in the continuum) at
    k = 2π m / (n L): they lie on Im k = im_max = 0."""
    L2 = 500.0
    g = qg.OpenGraphs([(0.0, 0.0), (300.0, 0.0)], [(0, 1, 0, 3, 2 * np.pi * R), (1, 0, 1, 1, L2)],
                      [(0, 0), (0, 2)],
                      vertex_matrices={0: directional_coupler_matrix(C_EXT), 1: PASS_THROUGH},
                      neff=NEFF, wavelength_range=(1.54, 1.56), **save)
    spectrum, deg = no_warning(g.find_resonances, N_re=40_000, N_im=8, im_min=-3e-4, im_max=0.0)
    k_min, k_max = g._k_range()
    m = np.arange(np.ceil(k_min * NEFF * L2 / (2 * np.pi)), np.floor(k_max * NEFF * L2 / (2 * np.pi)) + 1)
    real = spectrum[np.abs(spectrum.imag) < 1e-12]
    assert np.allclose(np.sort(real.real), 2 * np.pi * m / (NEFF * L2), atol=1e-10)
    assert g.count_resonances(-3e-4, 0.0).total == deg.sum()


def test_dispersive_index_is_counted_like_the_search(save):
    g = flower(save, neff=lambda k: 3.9 + 0.05 * (k - 4.0))
    spectrum, deg = no_warning(g.find_resonances, N_re=4_000, N_im=6, im_min=-2e-4, im_max=0.0)
    assert g.resonance_counts[0].total == deg.sum() >= 10
