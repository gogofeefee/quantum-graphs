# Changelog

## Unreleased

### Tests
- The non-regression data set failed on GitHub Actions (OpenBLAS) although
  it passed locally: it compared quantities that are not uniquely defined.
  `reference_v2.npz` stores the diagonal of the eigenspace projector for
  degenerate levels (single and double ring) instead of one arbitrary basis
  vector, and eigenvalue lists are compared through their power sums instead
  of after sorting (the defective zero eigenvalue of the classical
  Perron-Frobenius operators splits differently with each LAPACK). No change
  of computed results.

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
