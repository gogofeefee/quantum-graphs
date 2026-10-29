"""Closed graphs: analytic spectra, eigenvectors, Weyl law, caches."""

import os
import warnings

import numpy as np
import pytest

import quantum_graphs as qg


def ring(save, neff=2.41, wr=(1.50, 1.60)):
    return qg.ClosedGraphs(**qg.layouts.single_ring(), vertex_matrices=qg.PASS_THROUGH,
                           neff=neff, wavelength_range=wr, **save)


def test_single_ring_matches_analytic_spectrum(save):
    g = ring(save)
    spectrum, deg = g.compute_spectrum(N_scan=20_000, save_spectrum=False)
    L = 2 * np.pi * 100.0
    m = np.round(spectrum * 2.41 * L / (2 * np.pi))
    assert np.max(np.abs(spectrum - 2 * np.pi * m / (2.41 * L))) < 1e-7
    assert np.all(np.diff(m) == 1)                     # no level missed
    assert np.all(deg == 2)                            # clockwise / anticlockwise


def test_eigenvectors_solve_a_equals_Ua(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), vertex_couplings=0.5,
                                   neff=3.97, wavelength_range=(1.550, 1.552), **save)
    spectrum, deg = g.compute_spectrum(N_scan=20_000, save_spectrum=False)
    assert spectrum.size > 5 and np.all(deg == 1)
    for k in spectrum[:3]:
        a = g.compute_wavefunction(k)
        assert a.shape == (1, g.topology.n_directed)
        assert np.isclose(np.linalg.norm(a[0]), 1.0)
        # residual ~ sigma_min(I - U(k)) <= eps_accept; |dk| ~ sigma_min / (n L) ~ 1e-9
        assert np.linalg.norm(a[0] - g.U(k) @ a[0]) < 1e-4
        assert g.sigma_min(k) < 1e-4
        assert abs(g.secular_det(k)) < 1e-2 * abs(g.secular_det(k + 2e-4))


def test_double_ring_levels_are_doubly_degenerate(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.double_ring(), vertex_couplings=0.5, neff=1.0,
                                   wavelength_range=(1.48, 1.64), **save)
    spectrum, deg = g.compute_spectrum(N_scan=60_000, save_spectrum=False)
    assert spectrum.size == 71 and np.all(deg == 2)
    assert g.compute_wavefunction(spectrum[0]).shape[0] == 2


def test_level_count_agrees_with_weyl_law(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.bow_tie_graph(), vertex_couplings=0.5, neff=3.97,
                                   wavelength_range=(1.50, 1.53), **save)
    spectrum, deg = g.compute_spectrum(N_scan=150_000, save_spectrum=False)
    assert abs(deg.sum() - g.weyl_count()) < 3


def test_too_coarse_scan_warns_about_missing_levels(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), vertex_couplings=0.5, neff=3.97,
                                   wavelength_range=(1.50, 1.53), **save)
    with pytest.warns(RuntimeWarning, match="Weyl law"):
        g.compute_spectrum(N_scan=300, save_spectrum=False)


def test_dispersive_group_index():
    neff = lambda k: 2.4 + 0.08 * (k - 4.0)
    ng = qg.core.group_index_function(neff)
    assert np.isclose(ng(4.5), 2.4 + 0.08 * 0.5 + 4.5 * 0.08, rtol=1e-8)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_files(path):
    return sorted(f for f in os.listdir(path) if f.startswith("spectrum_") and f.endswith(".npz"))


def test_spectrum_cache_is_reused_and_verified(save):
    g = ring(save)
    s1, _ = g.compute_spectrum(N_scan=5_000)
    files = _cache_files(save["save_path"])
    assert len(files) == 1 and os.path.exists(os.path.join(save["save_path"], files[0] + ".meta.json"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # a valid cache loads silently
        s2, _ = ring(save).compute_spectrum(N_scan=5_000)
    assert np.array_equal(s1, s2)


def test_changed_index_function_is_not_served_from_cache(save):
    """Two different neff functions with the same __name__ must not share a cache."""
    def neff(k):
        return 2.41 + 0 * k
    s1, _ = ring(save, neff=neff).compute_spectrum(N_scan=5_000)

    def neff(k):                                       # same name, different values
        return 2.45 + 0 * k
    with pytest.warns(RuntimeWarning, match="different parameters"):
        s2, _ = ring(save, neff=neff).compute_spectrum(N_scan=5_000)
    assert not np.allclose(s1[:5], s2[:5])


def test_legacy_cache_without_metadata_loads_with_warning(save):
    g = ring(save)
    s1, _ = g.compute_spectrum(N_scan=5_000)
    meta = os.path.join(save["save_path"], _cache_files(save["save_path"])[0] + ".meta.json")
    os.remove(meta)
    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        s2, _ = ring(save).compute_spectrum(N_scan=5_000)
    assert np.array_equal(s1, s2)
