"""Closed graphs: analytic spectra, eigenvectors, Weyl law, caches."""

import os
import warnings

import numpy as np
import pytest

import quantum_graphs as qg
from quantum_graphs.core.vertices import neumann_matrix


def ring(save, neff=2.41, wr=(1.50, 1.60)):
    return qg.ClosedGraphs(**qg.layouts.single_ring(), vertex_matrices=qg.PASS_THROUGH,
                           neff=neff, wavelength_range=wr, **save)


def test_single_ring_matches_analytic_spectrum(save):
    g = ring(save)
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    L = 2 * np.pi * 100.0
    m = np.round(spectrum * 2.41 * L / (2 * np.pi))
    assert np.max(np.abs(spectrum - 2 * np.pi * m / (2.41 * L))) < 1e-13
    assert np.all(np.diff(m) == 1)                     # no level missed
    assert np.all(deg == 2)                            # clockwise / anticlockwise


def test_dispersive_ring_keeps_every_level(save):
    """k n(k) L = 2π m with consecutive m: the σ_min search (up to reference_v7)
    rejected levels where n_g L made the refined σ_min exceed 1e-4."""
    neff = lambda k: 2.4 + 0.35 * (k - 4.0)
    g = ring(save, neff=neff, wr=(1.48, 1.64))
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    m = spectrum * neff(spectrum) * 2 * np.pi * 100.0 / (2 * np.pi)
    assert np.max(np.abs(m - np.round(m))) < 1e-10
    assert np.all(np.diff(np.round(m)) == 1) and np.all(deg == 2)


def test_eigenvectors_solve_a_equals_Ua(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), couplings=0.5,
                                   neff=3.97, wavelength_range=(1.550, 1.552), **save)
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    assert spectrum.size > 5 and np.all(deg == 1)
    for k in spectrum[:3]:
        a = g.compute_wavefunction(k)
        assert a.shape == (1, g.topology.n_directed)
        assert np.isclose(np.linalg.norm(a[0]), 1.0)
        # levels to ~1e-15: residual ~ sigma_min(I - U(k)) ~ |dk| n L ~ 1e-12
        assert np.linalg.norm(a[0] - g.U(k) @ a[0]) < 1e-10
        assert g.sigma_min(k) < 1e-10
        assert abs(g.secular_det(k)) < 1e-2 * abs(g.secular_det(k + 2e-4))


def test_double_ring_levels_are_doubly_degenerate(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.double_ring(), couplings=0.5, neff=1.0,
                                   wavelength_range=(1.48, 1.64), **save)
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    assert spectrum.size == 71 and np.all(deg == 2)
    assert g.compute_wavefunction(spectrum[0]).shape[0] == 2


def test_level_count_agrees_with_weyl_law(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.bow_tie_graph(), couplings=0.5, neff=3.97,
                                   wavelength_range=(1.50, 1.53), **save)
    spectrum, deg = g.compute_spectrum(save_spectrum=False)
    assert abs(deg.sum() - g.weyl_count(sector="all")) < 3


def test_inconsistent_group_index_warns(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), couplings=0.5, neff=3.97,
                                   ng=3.0, wavelength_range=(1.50, 1.53), **save)
    with pytest.warns(RuntimeWarning, match="Weyl law"):
        g.compute_spectrum(save_spectrum=False)


def test_close_levels_are_resolved_and_degeneracies_counted(save):
    """Two loops on a Neumann vertex: equal lengths give levels of multiplicity
    3 (simultaneous eigenphase crossings); lengths differing by 1e-7 µm split
    them into levels ~1e-9 µm⁻¹ apart, all found. The count is exact."""
    def loops(dL):
        return qg.ClosedGraphs([(0.0, 0.0)], [(0, 0, 0, 1, 300.0), (0, 2, 0, 3, 300.0 + dL)],
                               vertex_matrices={0: neumann_matrix(4)}, neff=2.0,
                               wavelength_range=(1.50, 1.60), **save)
    with pytest.warns(RuntimeWarning, match="degenerate inside their sector"):
        spectrum, deg = loops(0.0).compute_spectrum(save_spectrum=False)
    assert set(deg) == {1, 3}
    split, split_deg = loops(1e-7).compute_spectrum(save_spectrum=False)
    assert np.all(split_deg == 1) and split.size == deg.sum()
    assert np.min(np.diff(split)) < 1e-8
    # every level found solves det(I - U) = 0
    assert np.max(loops(1e-7).sigma_min(split)) < 1e-10


def test_count_is_exact_on_any_interval():
    from quantum_graphs.core.level_search import find_levels
    g = qg.ClosedDirectionalGraphs(**qg.layouts.bow_tie_graph(), couplings=0.5, neff=3.97,
                                   wavelength_range=(1.50, 1.53))
    k_min, k_max = g._k_range()
    whole = find_levels(g.model, k_min, k_max)
    parts = [find_levels(g.model, a, b) for a, b in
             zip(np.linspace(k_min, k_max, 4)[:-1], np.linspace(k_min, k_max, 4)[1:])]
    assert whole.count == whole.multiplicities.sum() == sum(p.count for p in parts)
    assert np.allclose(whole.levels, np.concatenate([p.levels for p in parts]), atol=1e-14)
    assert whole.defect < 1e-6


def test_lossy_closed_graph_is_rejected(save):
    with pytest.warns(RuntimeWarning, match="not unitary"):
        g = qg.ClosedGraphs(**qg.layouts.single_ring(), vertex_matrices=0.9 * qg.PASS_THROUGH,
                            neff=2.41, wavelength_range=(1.50, 1.60), **save)
    with pytest.raises(ValueError, match="not unitary"):
        g.compute_spectrum(save_spectrum=False)


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
    s1, _ = g.compute_spectrum()
    files = _cache_files(save["save_path"])
    assert len(files) == 1 and os.path.exists(os.path.join(save["save_path"], files[0] + ".meta.json"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # a valid cache loads silently
        s2, _ = ring(save).compute_spectrum()
    assert np.array_equal(s1, s2)


def test_changed_index_function_is_not_served_from_cache(save):
    """Two different neff functions with the same __name__ must not share a cache."""
    def neff(k):
        return 2.41 + 0 * k
    s1, _ = ring(save, neff=neff).compute_spectrum()

    def neff(k):                                       # same name, different values
        return 2.45 + 0 * k
    with pytest.warns(RuntimeWarning, match="different parameters"):
        s2, _ = ring(save, neff=neff).compute_spectrum()
    assert not np.allclose(s1[:5], s2[:5])


def test_legacy_cache_without_metadata_loads_with_warning(save):
    g = ring(save)
    s1, _ = g.compute_spectrum()
    meta = os.path.join(save["save_path"], _cache_files(save["save_path"])[0] + ".meta.json")
    os.remove(meta)
    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        s2, _ = ring(save).compute_spectrum()
    assert np.array_equal(s1, s2)
