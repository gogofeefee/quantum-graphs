# Changelog

## Unreleased (2.0.0)

Breaking change of the API (scripts written for 1.0.0 keep working on the tag
`v1.0.0`). Physics unchanged: spectra, degeneracies, S-matrices, resonances,
secular determinants, spectral statistics, entropy / IPR and periodic-orbit
catalogues are those of 1.0.0 (reference_v4 vs reference_v3, see
`tests/test_regression.py`), with matrices up to 4× smaller.

### Changed (API)
- **Graphs are described by a bond list**: `bonds=[(i, p, j, q, L), ...]`
  (L optional: Euclidean distance) and, for open graphs, `leads=[(v, p), ...]`
  (S-matrix order). `port_map` and `lengths` are gone; every layout of
  `quantum_graphs.layouts` and every example uses the new form.
- **No virtual vertices.** Self-loops and multiple bonds are ordinary bonds:
  U has 2B rows for B bonds (flower graph 40 → 20, double ring 16 → 4).
- **`graph.topology`** (`core.topology.Topology`) replaces the ~25 attributes
  copied onto every graph (`port_map`, `virtual_port_map`, `updated_bonds`,
  `virtual_lengths`, `directed_bond_index`, `directed_lengths`,
  `reverse_directed_bond`, `physical_bond_arrivals`, `G`, ...): read-only
  arrays (`bonds`, `leads`, `length`, `valency`, `positions`, `src_vertex`,
  `src_port`, `dst_vertex`, `dst_port`, `directed_length`) and lookups
  (`reverse`, `bond_of`, `is_forward`, `leaving`, `arriving`, `directed`, `bond_id`). Directed bond
  `d = b` goes from (i, p) to (j, q), `d = b + B` back. Shortcuts on the
  graph: `n_vertices`, `n_bonds`, `n_leads`, `total_length`.
- `virtual_vertex_spacing` → `bond_spacing` (drawing only).
- `stats.physical_bond_intensity` → `stats.bond_intensity`.
- Periodic-orbit records: `path` holds directed-bond indices; `path_ports`
  the same as `(i, p, j, q)` tuples. `basis_state(dbond=(i, p, j, q))`.
- Removed: `copy_topology_kwargs` (use `graph.topology.as_dict()`), the
  unused `open_graphs.DirectionalGraphs.get_ring` / `path_length`, and the
  duplicated `_directional_coupler_matrix` methods.
- Save directories are named from the topology hash (`Nv=…_Nb=…_<hash>`):
  results cached by 1.0.0 are recomputed.
- **networkx is no longer a dependency** (it only drew the figures).

### Changed (API, continued)
- **Constructors**: everything after `bonds` (and `leads` for open graphs) is
  keyword-only; `wavelength_range` is required (no silent (1.48, 1.64)
  window) and checked. **Nothing is written to disk at construction**: the
  output directory and `graph_parameters.txt` are created by the first save
  (`graph.save_path` is resolved lazily; assigning it redirects the next
  outputs).
- **Coupling names**: `couplings` (float, C(k), list or dict) for every
  directional class (was `vertex_couplings`, `internal_coupling`); open
  graphs: `access_couplings` for the couplers carrying leads (was
  `external_coupling`); `OpenGraphs`: `vertex_matrices` + `access_matrices`
  (were `internal_matrix` / `external_matrix`). `DirectionalClassicalGraphs`
  takes `couplings` only (use `ClassicalGraphs` for arbitrary matrices).
- **Matrices**: `graph.U(k)`, `graph.sigma_min(k)` (one k or an array) and
  `graph.secular_det(k)` for every graph; `graph.S(k)` (one k or an array)
  replaces `compute_S_matrix` / `compute_S_matrices`. Removed the private
  wrappers `_build_propagation_matrix`, `_build_sigma_*`, `_U`, `_sigma_min`.
- **Classical dynamics**: `perron_frobenius(k)`, `perron_frobenius_spectrum(k)`,
  `invariant_measure(k)`, `spectral_gap`, `is_ergodic`, `is_mixing`,
  `classical_summary`, `basis_state`, `evolve_rho(rho0, n, trajectory=False)`,
  cached per k; removed `build_` / `get_` / `diagonalize_` variants,
  `evolve_rho_trajectory`, `evolve_and_store`, `perron_power`, the `rebuild`
  arguments and the results stored on the object (`P`, `P_evals`, ...).
- **Saving and plotting**: `save_fig` everywhere (the open-graph `save`
  argument is gone). Computations no longer plot by default:
  `find_resonances(plot=False)`, `find_experimental_resonances(plot=False,
  cache=True, save_fig=False)` (was `plot_figures=True, save=True`);
  `plot_length_spectrum` / `plot_resonances_sigma` no longer save by default.
  Figures are closed after drawing when display is disabled (`show=False`).
- Experimental-search helpers are private (`_k_to_lambda_um`, ...).
- `layouts.from_port_map(vertex_positions, port_map, lengths)` converts the
  1.0 description (for tools that still produce port maps, e.g. GDS builders).

### Changed (spectral statistics: sectors)
- **`graph.sectors`** (`core.sectors`): the strongly connected components of
  the transition pattern of Σ_αα split U(k) into independent blocks; with
  reciprocal vertices the reversed bonds of a sector are the sector itself
  (time-reversal symmetry, GOE expected) or a partner sector with the same
  spectrum (every level doubly degenerate, GUE expected). Double ring = two
  partners; flower and bow-tie graphs = one self-reverse sector.
- **`compute_spectrum` scans one sector per pair of partners** (double ring:
  U of size 2 instead of 4); the multiplicity of a level is known by
  construction. `find_resonances` assigns each resonance to a sector. Both set
  `graph.level_sectors`. A level degenerate *inside* a sector (geometric
  symmetry) raises a warning.
- **Statistics on the desymmetrised spectrum** with the Weyl law of the
  selected sectors, N(k) = (L_s/2π) ∫ n_g dk: the rule "all levels doubly
  degenerate → one per pair and Weyl × 1/2", which a single mis-detected
  level silently switched off, is gone with `mode=` (`raw` / `expanded` /
  `directional`), `stats_mode`, `degenerate_spectrum`, `N_weyl`,
  `N_weyl_scaled`, `_stats_selection`, `_is_pure_directional`,
  `use_experimental` and `NNSD_func` (use `qg.stats.nnsd`).
- New public methods on closed and open graphs: `levels(sector=None)`,
  `weyl_length`, `weyl_staircase(k)`, `weyl_count(k_min, k_max)`,
  `unfolded_spectrum()`, `sigma2()`, `delta3()`, `spectral_ensemble()`;
  `sector=None` (desymmetrised), an index, or `"all"`; an explicit
  `spectrum=` (e.g. `graph.exp_spectrum`) replaces `use_experimental=True`.
  `weyl_count()` counts the desymmetrised levels (`sector="all"`: with
  multiplicity).
- The statistics figures take `spectrum=`, `sector=` and `wavelength_range=`
  (was `wavelength_range_stats=`), default to Poisson + the expected ensemble,
  return their data, and are shared by closed and open graphs
  (`plotting.spectral_plots`); file names no longer carry `__directional` /
  `__expanded`. `plot_length_spectrum(spectrum=None, sector="all")` uses every
  level with its multiplicity (open graphs: was each resonance once).
- `stats.weyl_staircase(k, directed_length, ng, k0)` / `stats.unfold` take the
  directed length L_s (2 L for a whole graph) instead of `total_length` and
  `length_scale`.
- One degeneracy threshold (`core.scattering.NULLITY_TOL` = 1e-4): open-graph
  wavefunctions, entropy and IPR use the multiplicity found by
  `find_resonances` (`resonance_vectors(i)`; was a relative 1e-3 threshold);
  closed ones the multiplicity of `compute_spectrum`
  (`graph.null_vectors(k, multiplicity)`). `nullity_tol` is gone from the
  open entropy / IPR / wavefunction plots.
- Values unchanged on every reference graph (reference_v5 adds only the
  sector structure; spectra bit-identical, spacings ≤ 2e-12). Cached spectra
  and resonances of 2.0.0.dev0 are recomputed (new `level_sectors`).

### Changed (figures)
- One `plot_graph` for every class (`plotting.topology_plot`), drawn with
  matplotlib along `graph.bond_paths` (`plotting.drawing.BondPaths`):
  straight bonds, symmetric Bézier arcs for parallel bonds, teardrops of the
  true length for self-loops, lengths written on the curves, port markers
  tangent to the vertex disks, leads dashed. Wavefunctions, energy
  distributions, classical ρ arrows and orbit arrows follow the curves.
- **Figure life cycle, the same everywhere**: every plotting method takes
  `show=None` (default: the `show` option), saves through
  `plotting.style.save_figure`, and, when it created the figure, shows it or
  closes **only that figure** (`show_figure(fig, show)`; was
  `plt.close("all")`, which also closed the caller's figures). Methods given
  an `ax` (e.g. `plot_graph(ax=...)` as a background) neither save nor show.
  Plotting methods return their figure: `(fig, ax)` (`plot_graph`,
  `plot_orbit`, `plot_rho`, S-matrix plots, `plot_zeta_B`,
  `plot_energy_distribution`, `plot_resonances_sigma`,
  `plot_staircase_function`), a list of `(fig, ax)` (`plot_wavefunction`,
  `plot_resonance_wavefunction`, `plot_entropy`, `plot_ipr`);
  `find_experimental_resonances(plot=True)` still returns the resonances.
  `dispersion.plot_mpb_indices_vs_k(show=None)` (was `show=True`, and never
  closed its figure).
- `plot_orbit` builds its own figure: with `show=True` it used to show the
  bare graph before drawing the orbit.
- **Entropy / IPR shared by closed and open graphs**
  (`plotting.localization_plots.LocalizationPlotsMixin`): `plot_entropy` and
  `plot_ipr` take `wavelength_range` for both (open graphs had none) and draw
  the mean as a red dashed line (`show_mean_hline=True`, closed graphs had
  none); files `entropy_vs_kL[__lam<a>-<b>][__smoothW<n>]` (closed graphs
  wrote `__full` / `__range_[a-b]`). The values come from the new public
  `graph.mode_localization(wavelength_range=None)` → `(k, S_norm, ipr)`
  sorted by Re k (replaces `_compute_mode_quantities` /
  `_compute_all_resonance_quantities`).
- `find_experimental_resonances`: split into S cache / peak detection /
  figures; the three-panel figures use the constrained layout (no more
  tight_layout warnings); `amp_threshold` defaults to 0 (was 2e8, a value
  tuned for one graph) and the log reports the range of peak heights (µm²) to
  choose it. The curvature helpers are module functions
  (`derivatives_uniform`, `derivatives_k_to_lambda`).
- One `graph_parameters.txt` writer (`GraphBase._save_parameters`) for the
  four classes (were four copies); complex vertex matrices are written as
  complex numbers.
- Removed the re-export modules `closed_graphs.plotting.style` and
  `open_graphs.plotting.style` (use `quantum_graphs.plotting.style`) and dead
  helpers of `open_graphs.cache` (`comp_tag`, `progress_print`,
  `finish_progress`) and `classical_graphs._utils`.
- CI: actions/checkout@v7, actions/setup-python@v7, Python 3.14 added.

### Changed (results)
- **Classical dynamics**: one step of P = |Σ_αα|² is one bond traversal.
  With the virtual vertices a multiple bond took 2 steps and a self-loop 4,
  which changed the Perron-Frobenius spectrum: bow tie spectral gap
  0.1215 → 1 − 1/√2 ≈ 0.2929; flower graph eigenvalues ±i → −1 (period two).
  The new values are those of the paper's supplementary material and of the
  thesis (computed there without virtual vertices).
- Entropy / IPR per level of the closed bow tie differ by ≤ 4e-6 (eigenvectors
  at roots refined to ~1e-9, now from a smaller matrix).

### Fixed (resonance search, changes results)
- **`find_resonances` returns only the resonances inside its rectangle.** The
  Nelder-Mead stage of the refinement was unbounded: roots below im_min were
  returned (flower graph, 1.48–1.64 µm, im_min = −1e-4: 9 resonances at
  Im k down to −1.05e-4), and on the rings of the reference set, whose
  resonances all lie at Im k = −1.18e-4 / −2.36e-4 < im_min, it ran away to
  Re k = 3.456 and −0.610 (outside the window) and missed the resonance at
  Re k = 4.06639. Both stages are now bounded to a box around the candidate
  (which may cross the edges of the rectangle: a root just outside is found
  where it is), roots outside the rectangle are left out, and a warning says
  how many lie just below im_min. Grid minima in the first and last columns
  are candidates too. reference_v6: the two ring resonance sets (now the
  analytic poles to 1e-15, searched down to im_min = −3e-4) and the added
  counts below; nothing else changes (flower and bow tie over 1.48–1.64 µm:
  every resonance inside the rectangle identical to ≤ 2e-15).

### Changed (resonance search)
- **`find_resonances` uses Newton's method instead of the σ_min grid**
  (`quantum_graphs.core.zero_search`, `find_zeros`), sector by sector (one
  per pair of reciprocal partners, multiplicity by construction, as
  `compute_spectrum`): the zeros of det(I − U_s) in the rectangle are counted
  by the argument principle in sub-windows of ~20 resonances, found by
  batched Newton iterations on log det(I − U_s) from a lattice of seeds, then
  by deflation (zeros found divided out) in the sub-windows where fewer are
  found than counted. Flower graph, 1.48–1.64 µm, im_min = −1e-4: the same
  828 resonances as the grid search (≤ 1.8e-15) in ~1 s instead of 90 s.
  Resonances of reference_v6 unchanged (≤ 9e-16).
- New signature: `find_resonances(im_min, im_max, seeds_per_spacing=4,
  tol=1e-14, maxiter=100, max_rounds=6, eps_accept=1e-4, nullity_tol,
  overwrite=False, plot=False, verbose)`. Gone: `N_re`, `N_im`,
  `peak_prominence`, `neighbor_rows`, `box_re`, `box_im`, `tol_step`,
  `grid_dtype`, `certify` (the count is part of the search),
  `overwrite_grid` / `overwrite_refined` / `overwrite_spectrum` (→
  `overwrite`); no refined-candidate cache; the spectrum cache is
  `resonances_*.npz`. `plot_resonances_sigma(N_re=None)` sizes its heatmap
  grid from the mean spacing (20 points per spacing).

### Changed (closed spectrum, changes results)
- **`compute_spectrum` finds the eigenphase crossings of U(k)**
  (`quantum_graphs.core.level_search`, `find_levels` → `LevelSearch`)
  instead of the minima of σ_min(I − U(k)) on a grid. The number of levels in
  any interval is computed exactly from the eigenphases at its ends (counting
  function N(k) = Σ_j ⌊θ_j/2π⌋, no tracking), so the search is complete by
  construction and close levels are not merged; each level is located by
  Brent's method on the real secular function Z(k) = Π_j 2 sin(θ_j/2), and
  its multiplicity is the number of eigenphases crossing there. Levels are
  now accurate to ~1e-15 µm⁻¹: the bounded Brent minimisation of σ_min
  stopped at ~sqrt(ε)|k| (scipy's `fminbound`, whatever `tol`), i.e. ~4e-8
  µm⁻¹ (σ_min ~ 1e-5 at the level). Flower graph over 1.48–1.64 µm: same 837
  levels in 1.4 s instead of 18 s (N_scan = 3e6).
- **Fixed: levels silently lost with a large group index × length.** The
  refined σ_min (slope ~ n_g L, times the ~sqrt(ε)|k| accuracy of
  `fminbound`) could exceed `eps_accept` = 1e-4 and the level was rejected:
  the dispersive examples (MPB index) lost 14 of 160 levels (single ring) and
  11 of 272 (double ring), only flagged by the Weyl warning. Every paper
  spectrum (constant index, σ_min ≤ 2.4e-5) was complete: same levels and
  multiplicities, moved by ≤ 6.2e-8 µm⁻¹.
- New signature `compute_spectrum(tol=1e-15, show_progress=False,
  save_spectrum=True, overwrite=False)`; gone: `N_scan`, `maxiter`,
  `nullity_tol`, `eps_accept`. Cache `spectrum_lamMin_*_lamMax_*_tol*.npz`
  (older `spectrum_sigmaMin_*` caches are not read). A closed graph whose U is
  not unitary (lossy vertices) raises ValueError. The Weyl warning now means
  a group index inconsistent with the effective index (the count is exact).
  `ScatteringModel.eigenvalues_many`. Removed `_classify_levels`.
- reference_v8: closed spectra move by ≤ 6e-8 µm⁻¹ (same levels and
  multiplicities in every case), and with them spacings (≤ 2e-4 unfolded),
  Σ², Δ₃, entropy / IPR (≤ 5e-5), wavefunction residuals (1e-5 → 1e-12) and
  intensities; Σ² of the single ring (equal spacings) by up to 1e-2. Open
  graphs, classical dynamics and orbits unchanged.

### Changed (dispersive index off the real axis, changes results)
- **A dispersive n_eff is continued to first order at complex k**:
  n_eff(x + iy) = n_eff(x) + iy n_eff'(x) (`ScatteringModel.index`,
  `phase_rate`), instead of n_eff(Re k). Resonance widths now follow the
  group index, Im k ≈ ln t/(n_g L) for a ring, as the real-axis line shape
  of S does; with n_eff(Re k) they were too large by n_g/n_eff (×1.133 for
  the dispersive flower graph of the reference set). Re k is unchanged to
  ~1e-9; nothing changes on the real axis or for a constant index. S(k),
  σ_min heatmaps and transmission caches at complex k (k_imag ≠ 0) of
  dispersive graphs change accordingly; their caches carry the parameter
  `index_continuation` and older ones are recomputed. reference_v7: S of the
  dispersive flower graph at the complex probe wavenumbers (3e-4) and its
  resonances (added).

### Added
- **Completeness certificate for resonances**
  (`quantum_graphs.core.argument_principle`): `count_zeros(model, ...)` counts
  the zeros of det(I − U) in a rectangle of the complex k-plane, with
  multiplicity, by the argument principle on an adaptively sampled contour,
  in sub-rectangles if `cuts` are given (`ZeroCount`). `find_resonances`
  counts every independent sector in sub-windows of ~20 resonances and warns
  where the count differs from the roots found (missed roots, wrong
  multiplicity); the counts are stored in
  `graph.resonance_counts` and in the spectrum cache, and the warnings come
  back when it is loaded. `graph.count_resonances(im_min, im_max, sector=None,
  cuts=())` gives the count alone (flower graph over 1.48–1.64 µm: 0.4 s).
  `ScatteringModel.log_det_many`.
- **Weyl check for resonances**: a warning when the count in the rectangle
  falls short of the Weyl law of the sector (all Im k) by more than
  max(5 %, 3), i.e. when broader resonances lie below im_min (skipped for a
  non-Weyl sector, det Σ_s = 0).
- **Example `examples/closed_graphs/Pentagon/config_nconst.py`**: complete
  graph K5 of 2×2 couplers (C = 0.5, bond lengths drawn in [80, 120] µm).
  Although every coupler matrix is symmetric, the directed bonds split into
  two partner sectors (`graph.sectors`): every level is doubly degenerate and
  the desymmetrised spectrum (1909 levels in 0.8–2.0 µm) follows the GUE.

## 1.0.0 — 2026-10-29

Last version with the v1 API (`port_map` input, virtual vertices, historical
method names). The next version breaks the API; scripts written for v1 keep
running on this tag (`pip install git+https://github.com/gogofeefee/quantum-graphs@v1.0.0`).

### Changed (changes results)
- **Entropy and IPR** (`plot_entropy`, `plot_ipr`, wavefunction titles,
  `plot_energy_distribution`, closed and open graphs) are computed per
  **physical** bond (the bonds of `port_map`) instead of per virtualised
  half-bond: a self-loop or multiple bond counts once, with the amplitudes
  arriving at its real end vertices, and N_b is the number of physical bonds.
  For a degenerate level (or a resonance whose null space has dimension > 1)
  the diagonal of the eigenspace projector is used instead of one arbitrary,
  LAPACK-dependent eigenvector, and each level counts once. Paper BTG
  (tl = 85 µm, 1520–1540 nm): ⟨S_norm⟩ = 0.89 ± 0.04 (the change is well
  within the error bar); graphs without self-loops or multiple bonds are
  unchanged. See
  docs/scientific_notes.md §5. S_norm is NaN (was 0) for a single bond.

### Added
- `stats.localization` (`eigenspace_intensity`, `physical_bond_intensity`,
  `localization_measures`), `graph.bond_intensities(vecs)`,
  `graph.localization_measures(vecs)`, topology attribute
  `physical_bond_arrivals`.
- Topology attribute `vertex_labels`, used by every graph figure.

### Changed
- Vertex labels in `open_graphs` figures start at 1, like everywhere else;
  a vertex carrying leads (access coupler) is labelled `A` (`A1`, `A2`, …
  if there are several). The other vertices keep the labels of the closed
  graph obtained by `layouts.remove_access_coupler`. Closed-graph, classical
  and periodic-orbit figures are unchanged.

### Tests
- The non-regression data set failed on GitHub Actions (OpenBLAS) although
  it passed locally: it compared quantities that are not uniquely defined.
  `reference_v2.npz` stores the diagonal of the eigenspace projector for
  degenerate levels (single and double ring) instead of one arbitrary basis
  vector, and eigenvalue lists are compared through their power sums instead
  of after sorting (the defective zero eigenvalue of the classical
  Perron-Frobenius operators splits differently with each LAPACK). No change
  of computed results.
- `reference_v3.npz` adds the entropy and IPR of the reference levels and
  resonances (49 arrays); no stored quantity changes.

## 1.0.0.dev0 (2026-10-24)

Restructuring into one installable package, `quantum_graphs`, with a shared
core, a test suite and verified caches. Numerical results are unchanged
except where a bug was fixed (non-regression data set in `tests/reference/`).

### Fixed (could change results)
- **Bond lengths**: a self-loop or multiple bond whose `lengths` key was
  written in the other orientation (`((j, q), (i, p))`) silently used a default
  length of 40 µm. Keys are now accepted in either orientation, and missing,
  unknown or duplicated lengths raise an error.
- **Periodic orbits**: `find_orbits_up_to_length(L_max)` missed every orbit
  longer than L_max minus its shortest bond (flower graph, 1500 µm: 90 → 99
  orbits; bow-tie: 62 → 70). Truncation by `max_orbits` / `max_search_nodes`
  now warns and sets `orbits_complete = False`.
- **Eigenvectors**: `compute_wavefunction` (closed) and the resonance
  wavefunctions (open) returned the complex conjugate of the eigenvector;
  |a|², hence all figures, entropies and IPRs, were unaffected.
- **Caches**: cached spectra, σ_min grids, resonances and S(k) were reused
  even when the graph had changed in ways not encoded in the directory name
  (body of a `neff(k)` / `C(k)` function, dict couplings, C beyond two
  decimals, vertex matrices of the base classes) or when resonance-detection
  parameters (`peak_prominence`, `neighbor_rows`, `box_re/im`, manual or
  excluded peak lists) changed. Every cache now carries a `.meta.json` with a
  physics fingerprint and the parameters, and is recomputed on mismatch.
- **S-matrix cache**: a cache of a wider window could be linearly
  interpolated onto a finer grid (inventing line shapes narrower than the
  cached step) and the result was saved as if computed. Interpolation is now
  only allowed from a grid at least as fine, and never written back.
- Coupling lists (`vertex_couplings=[...]`) failed in closed graphs; C is
  checked to lie in [0, 1]; vertex-matrix sizes are checked against the
  valencies; non-unitary matrices warn.
- RMT curves: an unknown ensemble name silently meant Poisson; it now raises.

### Added
- `pip install -e .` (pyproject), public API `quantum_graphs` with
  unambiguous class names, `layouts` (reference geometries,
  `remove_access_coupler`), `dispersion`, `stats` (pure functions and exact
  Dyson-Mehta curves), options `set_options(verbose, show, n_workers)`.
- `spectral_rigidity(..., windows="uniform")` (textbook Δ₃) next to the
  historical level-anchored estimator; see docs/scientific_notes.md.
- Weyl-law check in `compute_spectrum`; `weyl_count()`; `U(k)`, `sigma_min(k)`,
  `compute_S_matrices(ks)`.
- `DirectionalClassicalGraphs(vertex_couplings=...)`.
- `PeriodicOrbits(base_dir=...)` (same save-path rule as the other classes).
- Test suite (unit, analytic, invariants, cache, plotting smoke, slow
  scientific validation and non-regression), GitHub Actions workflow,
  `examples/new_study_template.py`, docs/.

### Changed
- Batched, threaded linear algebra: closed flower-graph spectrum (300k scan
  points) 52.5 s → 7.8 s; σ_min on the resonance grid 157 → 19 µs/point;
  S(k) 131 → 5.7 µs/point (identical results).
- Code moved from `ClosedGraphs/closed_graphs`, ... to
  `quantum_graphs/closed_graphs`, ...; examples to `examples/`; RMT tables to
  `quantum_graphs/stats/data`.
- Open graphs: self-loops are virtualised with the same convention as the
  other modules (drawing of the loop and internal bond order change; S-matrix
  and resonances unchanged).
- Removed unused `N_weyl_scaled2`, `rho_weyl_scaled`,
  `find_experimental_resonances_old`.
