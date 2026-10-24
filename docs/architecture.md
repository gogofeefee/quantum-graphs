# Architecture

For maintainers, collaborators and AI agents: how the code is organised, where
to change what, and how to check a change.

## Package map

```
quantum_graphs/
  __init__.py          public API (classes with unambiguous names, options, layouts, stats)
  core/
    config.py          options verbose / show / n_workers, log(), show_figure()
    topology.py        build_topology(): validation, canonical lengths, virtual vertices,
                       directed-bond indexing  -> Topology (attributes copied onto graphs)
    vertices.py        directional_coupler_matrix, neumann_matrix, PASS_THROUGH,
                       couplings_for_vertices, shape / unitarity validation
    scattering.py      ScatteringModel: D(k), Σ blocks (constant ones cached), U(k),
                       batched sigma_min_many / S_many (threaded), internal_amplitudes
    optics.py          k <-> λ, group_index_function (n_g from n_eff)
    persistence.py     save paths, physics fingerprint, verified caches (.meta.json)
    base.py            GraphBase mixin: _setup_topology / _setup_vertex_matrices /
                       _setup_save_path, .model, .fingerprint, physics_spec()
  closed_graphs/       CoreMixin(GraphBase) + SpectrumMixin + SpectralStatsMixin + plotting mixins
  open_graphs/         CoreMixin(GraphBase) + ScatteringMixin + ResonancesMixin + ExperimentalMixin
                       + SpectralStatsMixin + plotting mixins; cache.py (S(λ) cache)
  classical_graphs/    CoreMixin(GraphBase) + DynamicsMixin + plotting / animation mixins
  periodic_orbits/     PeriodicOrbits(GraphBase): transition rule, DFS enumeration, plots
  stats/               spectral.py (pure functions: weyl_staircase, unfold, number_variance,
                       spectral_rigidity), rmt.py (nnsd, tabulated and exact RMT curves, data/)
  plotting/style.py    shared figure style and axis helpers
  layouts.py           reference geometries; remove_access_coupler
  dispersion.py        load_mpb_indices / plot_mpb_indices_vs_k / apply_index_scaling
```

Each graph class is assembled from mixins (e.g. `closed_graphs.DirectionalGraphs =
CoreMixin + SpectrumMixin + SpectralStatsMixin + plotting mixins`). Shared
physics lives in `core` and `stats`; the subpackages only add what is
specific to their point of view.

## Data flow

1. **Construction** (`CoreMixin.__init__` of each subpackage):
   `build_topology` validates `port_map` / `lengths` and virtualises loops and
   multiple bonds; vertex matrices are resolved and validated; the save path
   is resolved (historical directory names are kept so old results are
   found) and `graph_parameters.txt` is written.
2. **Numerics** go through `graph.model` (a `ScatteringModel` built lazily
   from the topology, the vertex matrices and `neff`). The historical methods
   (`_build_sigma_alpha_alpha`, `_build_propagation_matrix`, `compute_S_matrix`,
   `_U`, `_sigma_min`) delegate to it.
3. **Caches**: `persistence.use_cache(path, graph.fingerprint, params,
   overwrite)` before loading, `write_cache_meta(...)` after writing. The
   fingerprint samples every function of k at 9 wavenumbers of the window.
4. **Plots** end with `show_figure()` (respects the `show` option); messages
   go through `log()` or `print` guarded by a resolved `verbose`.

## Extending

* **A new graph**: no code needed — describe it with `vertex_positions`,
  `port_map`, `lengths` (see `examples/new_study_template.py`); add it to
  `layouts.py` if it is reused.
* **A new vertex model**: write a function returning σ (or σ(k)) in
  `core/vertices.py`; pass it through `vertex_matrices`. Add a test of
  unitarity / symmetry.
* **A new analysis of a spectrum**: a pure function in `stats/spectral.py`
  (arrays in, arrays out), then a thin mixin method that selects the spectrum
  and plots. Test the function on an analytic case.
* **A new cached computation**: build the file name from the parameters,
  call `use_cache` / `write_cache_meta` with the full parameter dict.
* **A new graph class**: inherit `GraphBase`, call `_setup_topology`,
  `_setup_vertex_matrices`, `_setup_save_path`; use `self.model`.

## Validation

* `pytest` — fast suite (~30 s): analytic cases, physical invariants,
  validation errors, cache behaviour, plotting smoke tests.
* `pytest -m slow` — GOE ensemble validation of the Δ₃ estimators and
  non-regression: `tests/reference/generate_reference.py` recomputes 196
  quantities on nine reference graphs and `compare_reference.py` compares them
  with `reference_v1.npz` using per-quantity tolerances.
* `reference_v0.npz` is the state published before the 2026-10 refactor;
  `test_v0_to_v1_differences_are_the_documented_ones` pins the only
  intentional change (orbits recovered by the orbit-enumeration fix). When a
  change of results is intended, regenerate a new reference, document the
  differences in `CHANGELOG.md` and update that test.

## Legacy import names

Scripts written before 1.0 import `closed_graphs`, `open_graphs`,
`classical_graphs`, `periodic_orbits` (top level) or `sigma2_RMT` /
`delta3_RMT`. On the author's machine, small shim packages at the historical
locations (`ClosedGraphs/closed_graphs/`, ..., `wigner/`, not versioned) alias
them to `quantum_graphs.*`. Elsewhere, replace `from closed_graphs import X`
by `from quantum_graphs.closed_graphs import X`.
