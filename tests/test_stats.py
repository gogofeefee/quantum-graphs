"""Spectral statistics and random-matrix reference curves."""

import numpy as np
import pytest
from scipy.integrate import quad

from quantum_graphs import stats
from quantum_graphs.stats.spectral import _delta3_window


@pytest.mark.parametrize("ens", ["poisson", "GOE", "GUE", "GSE"])
def test_nnsd_is_normalised_with_unit_mean(ens):
    norm = quad(lambda s: stats.nnsd(s, ens), 0, 50)[0]
    mean = quad(lambda s: s * stats.nnsd(s, ens), 0, 50)[0]
    assert np.isclose(norm, 1.0, atol=1e-6) and np.isclose(mean, 1.0, atol=1e-3)


def test_unknown_ensemble_raises():
    with pytest.raises(ValueError):
        stats.sigma2_rmt(1.0, "GOEE")


def test_delta3_closed_form_equals_least_squares():
    rng = np.random.default_rng(1)
    E = np.sort(np.cumsum(rng.exponential(size=200)))
    x, L = E[40], 6.0
    sub = E[(E > x) & (E <= x + L)]
    e = np.linspace(x, x + L, 200_001)
    N = np.searchsorted(sub, e, side="right").astype(float)
    M = np.vstack([e, np.ones_like(e)]).T
    resid = N - M @ np.linalg.lstsq(M, N, rcond=None)[0]
    assert np.isclose(_delta3_window(sub, x, L), np.trapezoid(resid**2, e) / L, rtol=1e-4)


def test_poisson_statistics():
    rng = np.random.default_rng(2)
    E = np.cumsum(rng.exponential(size=40_000))
    L, s2 = stats.number_variance(E, 1.0, 5.0)
    assert np.allclose(s2[1:], L[1:], rtol=0.06)
    L, d3 = stats.spectral_rigidity(E[:4000], 1.0, 5.0)
    assert np.allclose(d3[1:], L[1:] / 15, rtol=0.15)


def test_sigma2_tables_match_exact_dyson_mehta():
    L = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    for ens in ("GOE", "GUE"):
        assert np.allclose(stats.sigma2_rmt(L, ens), stats.sigma2_dyson_mehta(L, ens), rtol=0.025)


def test_delta3_conventions_agree_at_large_L():
    L = np.array([5.0, 10.0])
    for ens in ("GOE", "GUE"):
        assert np.allclose(stats.delta3_rmt(L, ens), stats.delta3_dyson_mehta(L, ens), rtol=0.03)
    # and differ at small L (documented convention difference)
    assert stats.delta3_dyson_mehta(1.0, "GOE") > 1.4 * stats.delta3_rmt(1.0, "GOE")


def test_weyl_staircase_constant_index():
    # N = (L_s / 2π) n_g k: L_s = 2 L for the whole spectrum of a graph of length L
    assert np.isclose(stats.weyl_staircase(4.0, 2000.0, 2.0), 1000 * 2 * 4 / np.pi)
    k = np.linspace(3, 4, 5)
    assert np.allclose(stats.weyl_staircase(k, 2000.0, lambda x: 2.0 + 0 * x, k0=3.0),
                       1000 * 2 * (k - 3) / np.pi)


@pytest.mark.slow
def test_goe_estimators_against_theory():
    """Level-anchored Δ₃ reproduces the tables, uniform windows the exact theory."""
    rng = np.random.default_rng(0)

    def goe_unfolded(N):
        A = rng.normal(size=(N, N))
        ev = np.linalg.eigvalsh((A + A.T) / 2) / np.sqrt(N)
        R = np.sqrt(2.0)
        u = N * (0.5 + (ev * np.sqrt(np.clip(R**2 - ev**2, 0, None))
                        + R**2 * np.arcsin(np.clip(ev / R, -1, 1))) / (np.pi * R**2))
        return u[N // 4: 3 * N // 4]
    Ls = np.array([1.0, 2.0, 5.0])
    lev, uni = [], []
    for _ in range(30):
        u = goe_unfolded(600)
        L, d = stats.spectral_rigidity(u, 1.0, 5.0, windows="level")
        lev.append(d[[1, 2, 5]])
        L, d = stats.spectral_rigidity(u, 1.0, 5.0, windows="uniform")
        uni.append(d[[1, 2, 5]])
    assert np.allclose(np.mean(lev, 0), stats.delta3_rmt(Ls, "GOE"), rtol=0.04)
    assert np.allclose(np.mean(uni, 0), stats.delta3_dyson_mehta(Ls, "GOE"), rtol=0.04)
