# Architecture

For maintainers, collaborators and AI agents: how the code is organised, where
to change what, and how to check a change.

## Package map

```
quantum_graphs/
  __init__.py          public API (classes with unambiguous names, options, layouts, stats)
  core/
    config.py          options verbose / show / n_workers, log(), show_figure()
    topology.py        Topology (graph.topology): bonds (i, p, j, q, L) and leads as
                       validated read-only arrays, directed bonds d / d + B, lookups
    vertices.py        directional_coupler_matrix, neumann_matrix, PASS_THROUGH,
                       couplings_for_vertices, shape / unitarity validation
    scattering.py      ScatteringModel: D(k), Σ blocks (constant ones cached), U(k),
                       n_eff continued to first order off the real axis (index,
                       phase_rate), batched sigma_min_many / singular_values_many /
                       eigenvalues_many / log_det_many / S_many (threaded),
                       internal_amplitudes, model.sector(directed) (U restricted to a
                       sector), NULLITY_TOL
    level_search.py    find_levels -> LevelSearch: real levels of a unitary U(k) (closed
                       graphs) = eigenphase crossings, exact count per interval,
                       Brent on Z(k) = Π 2 sin(θ_j/2), multiplicity = simultaneous crossings
    argument_principle.py  count_zeros -> ZeroCount: zeros of det(I − U) in a rectangle
                       of the complex k-plane and its sub-rectangles (adaptive contour)
    zero_search.py     find_zeros -> ZeroSearch: the zeros themselves (batched Newton on
                       log det(I − U) from a lattice of seeds, deflation until the
                       count of each sub-rectangle is reached)
    sectors.py         find_sectors: strongly connected components of the pattern of Σ,
                       reversed partners, expected ensemble (graph.sectors)
    spectral_stats.py  SpectralStatsMixin (closed and open): selected levels, Weyl law
                       of the selected sectors, unfolded_spectrum, sigma2, delta3
    optics.py          k <-> λ, group_index_function (n_g from n_eff)
    persistence.py     save paths, physics fingerprint, verified caches (.meta.json)
    base.py            GraphBase mixin: _setup_topology / _setup_vertex_matrices /
                       _setup_save_path, .model, .fingerprint, physics_spec()
  closed_graphs/       CoreMixin(GraphBase) + SpectrumMixin + SpectralStatsMixin + plotting mixins
  open_graphs/         CoreMixin(GraphBase) + ScatteringMixin + ResonancesMixin + ExperimentalMixin
                       + SpectralStatsMixin + plotting mixins; cache.py (S(λ) cache)
                       (SpectralStatsMixin and SpectralPlotsMixin are shared, from core/
                       and plotting/)
  classical_graphs/    CoreMixin(GraphBase) + DynamicsMixin + plotting / animation mixins
  periodic_orbits/     PeriodicOrbits(GraphBase): transition rule, DFS enumeration, plots
  stats/               spectral.py (pure functions: weyl_staircase, unfold, number_variance,
                       spectral_rigidity), rmt.py (nnsd, tabulated and exact RMT curves, data/),
                       localization.py (bond intensities, entropy, IPR)
  plotting/            style.py (figure style, axis helpers, save_figure, box_ticks);
                       localization_plots.py (plot_entropy / plot_ipr of closed and open
                       graphs from their mode_localization); drawing.py (BondPaths: one
                       curve per bond -- segment, Bézier arcs for parallel bonds, teardrops
                       for loops -- and the draw_* primitives); topology_plot.py (the
                       single plot_graph shared by every class); spectral_plots.py
                       (staircase, NNSD, Σ² / Δ₃, length spectrum of closed and open graphs)
  layouts.py           reference geometries; remove_access_coupler
  dispersion.py        load_mpb_indices / plot_mpb_indices_vs_k / apply_index_scaling
```

Each graph class is assembled from mixins (e.g. `closed_graphs.DirectionalGraphs =
CoreMixin + SpectrumMixin + SpectralStatsMixin + plotting mixins`). Shared
physics lives in `core` and `stats`; the subpackages only add what is
specific to their point of view.

## Data flow

1. **Construction** (`CoreMixin.__init__` of each subpackage):
   `build_topology` validates `vertex_positions` / `bonds` / `leads` into
   `graph.topology` (loops and multiple bonds are ordinary bonds); vertex
   matrices are resolved and validated; the save path is resolved and
   nothing is written to disk before the first save (`graph.save_path`
   creates the directory and `graph_parameters.txt` on first use). The graph exposes only the shortcuts
   `n_vertices`, `n_bonds`, `n_leads`, `total_length`; everything else is
   read from `graph.topology`.
2. **Numerics** go through `graph.model` (a `ScatteringModel` built lazily
   from the topology, the vertex matrices and `neff`); `graph.U(k)`,
   `graph.sigma_min(k)`, `graph.secular_det(k)` (every graph) and `graph.S(k)`
   (open graphs; one k or an array) are thin wrappers around it.
3. **Caches**: `persistence.use_cache(path, graph.fingerprint, params,
   overwrite)` before loading, `write_cache_meta(...)` after writing. The
   fingerprint samples every function of k at 9 wavenumbers of the window.
4. **Plots** draw every bond along `graph.bond_paths` (geometry only, for
   figures; the physics uses the bond lengths). Every plotting method takes
   `show=None`, saves with `plotting.style.save_figure` and, when it created
   the figure, ends with `show_figure(fig, show)`: shown, or that figure
   closed (never the caller's); given an `ax`, it neither saves nor shows.
   It returns its figure(s). Messages go through `log()` or `print`
   guarded by a resolved `verbose`.

## Extending

* **A new graph**: no code needed — describe it with `vertex_positions` and
  `bonds` (+ `leads`) (see `examples/new_study_template.py`); add it to
  `layouts.py` if it is reused.
* **A new vertex model**: write a function returning σ (or σ(k)) in
  `core/vertices.py`; pass it through `vertex_matrices`. Add a test of
  unitarity / symmetry.
* **A new analysis of a spectrum**: a pure function in `stats/spectral.py`
  (arrays in, arrays out), then a thin method of `core/spectral_stats.py`
  that takes the selected levels (`_stats_levels(spectrum, sector,
  wavelength_range)`, Weyl length `weyl_length(sector)`) and a plot in
  `plotting/spectral_plots.py`. Test the function on an analytic case.
* **A new cached computation**: build the file name from the parameters,
  call `use_cache` / `write_cache_meta` with the full parameter dict.
* **A new graph class**: inherit `GraphBase`, call `_setup_topology`,
  `_setup_vertex_matrices`, `_setup_save_path`; use `self.model`.

## Validation

* `pytest` — fast suite (~30 s): analytic cases, physical invariants,
  validation errors, cache behaviour, plotting smoke tests.
* `pytest -m slow` — GOE ensemble validation of the Δ₃ estimators and
  non-regression: `tests/reference/generate_reference.py` recomputes 271
  quantities on nine reference graphs and `compare_reference.py` compares them
  with `reference_v8.npz` using per-quantity tolerances.
* Only basis- and platform-independent quantities are compared: eigenvectors
  of degenerate levels through the diagonal of the eigenspace projector,
  eigenvalue lists through their power sums Σλⁿ (a defective eigenvalue, such
  as the zero eigenvalue of the classical operators, is split by LAPACK into a
  cloud of radius ~ε^(1/m) whose shape depends on the BLAS library).
* `reference_v0.npz` is the state published before the 2026-10 refactor;
  `test_v0_to_v1_differences_are_the_documented_ones` pins the only
  intentional change (orbits recovered by the orbit-enumeration fix) and
  `test_v1_to_v2_differences_are_the_documented_ones` the change of stored
  quantity for degenerate eigenvectors; reference_v3 only adds the entropy
  and IPR (`test_v2_to_v3_differences_are_the_documented_ones`), reference_v4
  the bond lists without virtual vertices, reference_v5 the sector structure
  (`test_v4_to_v5_...`, nothing else changed), reference_v6 the bounded
  resonance refinement (the two ring resonance sets, which v5 got wrong) and
  the argument-principle counts (`test_v5_to_v6_...`), reference_v7 the
  Newton resonance search and the continuation of a dispersive index off the
  real axis (`test_v6_to_v7_...`: only the dispersive S at complex k changes,
  and the dispersive resonances are added), reference_v8 the closed spectra
from the eigenphase crossings (`test_v7_to_v8_...`: the levels move by
≤ 6e-8, the accuracy of the old σ_min minimisation, with everything computed
from them; same levels and multiplicities). When a change of results is intended,
  regenerate a new reference, document the differences in `CHANGELOG.md` and
  add the corresponding test.

## Documentation

* `README.md` (user guide and gallery), `docs/api.md` (one table per public
  class and module), `docs/scientific_notes.md` (conventions, algorithms),
  this file, `CHANGELOG.md`.
* The figures of the README are made by `python docs/make_figures.py` (about
  ten seconds) from the public API and written to `docs/images/`. Run it
  again after a change of results or of the figure style, and update
  `docs/api.md` with any change of the public API.

## Legacy import names

Version 2.0 changed the API (bond lists instead of port maps, no virtual
vertices, new method names: see `CHANGELOG.md`). Scripts written for 1.x run
on the tag `v1.0.0`
(`pip install git+https://github.com/gogofeefee/quantum-graphs@v1.0.0`), or
can be ported to the 2.0 API. Older scripts that import the pre-1.0
top-level names `closed_graphs`, `open_graphs`, `classical_graphs`,
`periodic_orbits`, `sigma2_RMT` or `delta3_RMT` run on that tag once these
imports point to `quantum_graphs.closed_graphs`, ...,
`quantum_graphs.stats.rmt`.
