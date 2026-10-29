"""Open graphs: analytic ring, S-matrix properties, resonances, S cache."""

import numpy as np
import pytest

import quantum_graphs as qg

C_EXT, NEFF, R = 0.3, 2.41, 100.0


def ring1(save, wr=(1.54, 1.56)):
    return qg.OpenDirectionalGraphs(**qg.layouts.ring_with_access(R, 1), access_couplings=C_EXT,
                                    couplings=0.5, neff=NEFF, wavelength_range=wr, **save)


def test_single_access_ring_is_the_analytic_all_pass_filter(save):
    """S_10 = (t − φ) / (1 − t φ), t = √(1−C), φ = exp(i k n L)."""
    g = ring1(save)
    t, L = np.sqrt(1 - C_EXT), 2 * np.pi * R
    for k in np.linspace(4.03, 4.08, 50) - 2e-6j:
        phi = np.exp(1j * k * NEFF * L)
        assert np.isclose(g.S(k)[1, 0], (t - phi) / (1 - t * phi), atol=1e-12)


def test_single_access_ring_resonances(save):
    """Poles: exp(i k n L) = 1/t, i.e. k_m = (2π m + i ln t) / (n L)."""
    g = ring1(save)
    spectrum, _ = g.find_resonances(N_re=20_000, N_im=12, im_min=-3e-4, im_max=0.0, plot=False)
    t, L = np.sqrt(1 - C_EXT), 2 * np.pi * R
    m = np.round(spectrum.real * NEFF * L / (2 * np.pi))
    expected = (2 * np.pi * m + 1j * np.log(t)) / (NEFF * L)
    assert spectrum.size >= 9
    assert np.max(np.abs(spectrum - expected)) < 1e-8


@pytest.mark.parametrize("layout", [qg.layouts.flower_graph(open=True),
                                    qg.layouts.bow_tie_graph(open=True),
                                    qg.layouts.ring_with_access(n_access=2)])
def test_S_is_unitary_and_symmetric_on_the_real_axis(layout, save):
    g = qg.OpenDirectionalGraphs(**layout, couplings=0.5, access_couplings=0.3,
                                 neff=3.97, wavelength_range=(1.55, 1.56), **save)
    S = g.S(np.linspace(4.03, 4.06, 40))
    eye = np.eye(g.n_leads)
    assert np.allclose(np.conj(np.swapaxes(S, 1, 2)) @ S, eye, atol=1e-12)
    assert np.allclose(S, np.swapaxes(S, 1, 2), atol=1e-12)       # reciprocity


def test_batched_and_single_S_agree(save):
    g = qg.OpenDirectionalGraphs(**qg.layouts.flower_graph(open=True), couplings=0.5,
                                 access_couplings=0.3, neff=lambda k: 3.9 + 0.01 * k,
                                 wavelength_range=(1.55, 1.56), **save)
    ks = np.linspace(4.03, 4.06, 7) - 1e-5j
    assert np.allclose(g.S(ks), [g.S(k) for k in ks], atol=1e-13)


def test_coarse_S_cache_is_never_interpolated(save):
    from quantum_graphs.open_graphs.cache import get_S_full
    g = ring1(save)
    get_S_full(g, 1.54, 1.56, 200, 0.0)                     # coarse, wide
    lam, k, dk, S_fine = get_S_full(g, 1.545, 1.555, 4000, 0.0)
    exact = g.S(k)
    assert np.allclose(S_fine, exact, atol=1e-12)


def test_resonance_cache_detects_changed_detection_parameters(save):
    g = ring1(save)
    kw = dict(N_re=4_000, N_im=6, im_min=-3e-4, im_max=0.0, plot=False)
    g.find_resonances(**kw)
    with pytest.warns(RuntimeWarning, match="different parameters"):
        ring1(save).find_resonances(**kw, peak_prominence=0.5)


def test_results_do_not_depend_on_the_number_of_threads(save):
    g = qg.OpenDirectionalGraphs(**qg.layouts.flower_graph(open=True), couplings=0.5,
                                 access_couplings=0.3, neff=3.97, wavelength_range=(1.55, 1.56), **save)
    ks = np.linspace(4.03, 4.06, 9000) - 1e-5j
    out = {}
    for w in (1, 4):
        with qg.options(n_workers=w):
            out[w] = (g.model.sigma_min_many(ks), g.S(ks))
    assert np.array_equal(out[1][0], out[4][0]) and np.array_equal(out[1][1], out[4][1])
