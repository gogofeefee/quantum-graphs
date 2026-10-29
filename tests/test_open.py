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
    spectrum, _ = g.find_resonances(im_min=-3e-4, im_max=0.0, plot=False)
    t, L = np.sqrt(1 - C_EXT), 2 * np.pi * R
    m = np.round(spectrum.real * NEFF * L / (2 * np.pi))
    expected = (2 * np.pi * m + 1j * np.log(t)) / (NEFF * L)
    assert spectrum.size >= 9
    assert np.max(np.abs(spectrum - expected)) < 1e-13


def test_dispersive_ring_resonance_widths_follow_the_group_index(save):
    """With n_eff(k) = a + b (k − k0), exactly continued off the real axis, the poles
    solve k n_eff(k) L = 2π m + i ln t: Im k ≈ ln t / (n_g L), not ln t / (n_eff L)."""
    a, b, k0 = 2.41, 0.3, 4.06
    g = qg.OpenDirectionalGraphs(**qg.layouts.ring_with_access(R, 1), access_couplings=C_EXT,
                                 couplings=0.5, neff=lambda k: a + b * (k - k0),
                                 wavelength_range=(1.54, 1.56), **save)
    spectrum, deg = g.find_resonances(im_min=-2e-4, im_max=0.0)
    t, L = np.sqrt(1 - C_EXT), 2 * np.pi * R
    m = np.round(spectrum.real * (a + b * (spectrum.real - k0)) * L / (2 * np.pi))
    c = -(2 * np.pi * m + 1j * np.log(t)) / L
    expected = (-(a - b * k0) + np.sqrt((a - b * k0) ** 2 - 4 * b * c)) / (2 * b)
    assert spectrum.size >= 9 and np.all(deg == 2)
    assert np.max(np.abs(spectrum - expected)) < 1e-12
    n_g = a + b * (2 * spectrum.real - k0)
    assert np.allclose(spectrum.imag, np.log(t) / (n_g * L), rtol=1e-3)


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


def test_resonance_cache_detects_a_changed_graph(save):
    kw = dict(im_min=-3e-4, im_max=0.0, plot=False)
    ring1(save).find_resonances(**kw)
    other = qg.OpenDirectionalGraphs(**qg.layouts.ring_with_access(R, 1), access_couplings=0.2,
                                     couplings=0.5, neff=NEFF, wavelength_range=(1.54, 1.56), **save)
    with pytest.warns(RuntimeWarning, match="different parameters"):
        other.find_resonances(**kw)


def test_results_do_not_depend_on_the_number_of_threads(save):
    g = qg.OpenDirectionalGraphs(**qg.layouts.flower_graph(open=True), couplings=0.5,
                                 access_couplings=0.3, neff=3.97, wavelength_range=(1.55, 1.56), **save)
    ks = np.linspace(4.03, 4.06, 9000) - 1e-5j
    out = {}
    for w in (1, 4):
        with qg.options(n_workers=w):
            out[w] = (g.model.sigma_min_many(ks), g.S(ks))
    assert np.array_equal(out[1][0], out[4][0]) and np.array_equal(out[1][1], out[4][1])


def test_finer_S_cache_of_a_wider_window_is_interpolated(save, monkeypatch):
    from quantum_graphs.open_graphs.cache import get_S_full
    g = ring1(save)
    get_S_full(g, 1.54, 1.56, 40_000, 0.0)                   # fine, wide
    _, _, _, S_exact = get_S_full(g, 1.545, 1.555, 4000, 0.0, reuse_if_included=False)
    k_sub = np.linspace(2 * np.pi / 1.554, 2 * np.pi / 1.546, 3000)
    S_sub = g.S(k_sub)

    def no_computation(*args, **kwargs):
        raise AssertionError("S recomputed instead of read from the cache")
    monkeypatch.setattr(type(g.model), "S_many", no_computation)
    _, _, _, S_again = get_S_full(g, 1.545, 1.555, 4000, 0.0)        # exact cache
    assert np.array_equal(S_again, S_exact)
    _, k, _, S_interp = get_S_full(g, 1.546, 1.554, 3000, 0.0)       # interpolated
    assert np.allclose(k, k_sub) and np.allclose(S_interp, S_sub, atol=1e-2)


def test_curvature_derivatives_are_exact_on_polynomials():
    from quantum_graphs.open_graphs.experimental import derivatives_k_to_lambda, derivatives_uniform
    x = np.linspace(1.0, 2.0, 101)
    _, d1, d2 = derivatives_uniform(x, x ** 3)
    assert np.allclose(d1[2:-2], 3 * x[2:-2] ** 2) and np.allclose(d2[2:-2], 6 * x[2:-2])
    assert np.allclose(d2[[0, 1, -2, -1]], 6 * x[[0, 1, -2, -1]], atol=1e-6)
    with pytest.raises(ValueError, match="uniformly"):
        derivatives_uniform(x ** 2, x)
    # T(k) = k² as a function of λ = 2π/k: T = 4π²/λ², dT/dλ = −8π²/λ³, d²T/dλ² = 24π²/λ⁴
    k = np.linspace(4.0, 4.1, 5)
    lam = 2 * np.pi / k
    dT, d2T = derivatives_k_to_lambda(k, 2 * k, 2 * np.ones_like(k))
    assert np.allclose(dT, -8 * np.pi ** 2 / lam ** 3) and np.allclose(d2T, 24 * np.pi ** 2 / lam ** 4)


def test_experimental_resonances_match_the_poles_and_are_cached(save, monkeypatch):
    g = qg.OpenDirectionalGraphs(**qg.layouts.ring_with_access(R, 2), access_couplings=C_EXT,
                                 couplings=0.5, neff=NEFF, wavelength_range=(1.545, 1.555), **save)
    kw = dict(n_points=40_000, k_imag=0.0, component=(1, 0))
    k_exp = g.find_experimental_resonances(**kw)
    poles, _ = g.find_resonances(im_min=-5e-4, im_max=0.0)
    assert k_exp.size == poles.size == 7
    assert np.max(np.abs(k_exp - poles.real)) < 2e-6          # ~ 3 grid steps

    lam = 2 * np.pi / k_exp
    manual, excluded = [float(lam[0] + 2e-4)], [float(lam[-1] * 1e3)]
    k_edit = g.find_experimental_resonances(**kw, manual_lambda_peaks=manual,
                                            exclude_lambda_peaks=excluded)
    assert k_edit.size == k_exp.size
    assert np.isclose(2 * np.pi / k_edit, manual[0]).any() and not np.isclose(k_edit, k_exp[-1]).any()
    assert g.find_experimental_resonances(**kw, amp_threshold=np.inf).size == 0

    def no_computation(*args, **kwargs):
        raise AssertionError("S recomputed instead of read from the cache")
    monkeypatch.setattr(type(g.model), "S_many", no_computation)
    assert np.array_equal(g.find_experimental_resonances(**kw), k_exp)              # peaks cached
    assert np.array_equal(g.find_experimental_resonances(**kw, distance=6), k_exp)  # S cached
