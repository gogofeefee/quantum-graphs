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

### Changed (figures)
- One `plot_graph` for every class (`plotting.topology_plot`), drawn with
  matplotlib along `graph.bond_paths` (`plotting.drawing.BondPaths`):
  straight bonds, symmetric Bézier arcs for parallel bonds, teardrops of the
  true length for self-loops, lengths written on the curves, port markers
  tangent to the vertex disks, leads dashed. Wavefunctions, energy
  distributions, classical ρ arrows and orbit arrows follow the curves.

### Changed (results)
- **Classical dynamics**: one step of P = |Σ_αα|² is one bond traversal.
  With the virtual vertices a multiple bond took 2 steps and a self-loop 4,
  which changed the Perron-Frobenius spectrum: bow tie spectral gap
  0.1215 → 1 − 1/√2 ≈ 0.2929; flower graph eigenvalues ±i → −1 (period two).
  The new values are those of the paper's supplementary material and of the
  thesis (computed there without virtual vertices).
- Entropy / IPR per level of the closed bow tie differ by ≤ 4e-6 (eigenvectors
  at roots refined to ~1e-9, now from a smaller matrix).

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
  (tl = 85 µm, 1520–1540 nm): ⟨S_norm⟩ 0.892 → 0.888, ⟨IPR⟩ 1.55 → 1.52;
  graphs without self-loops or multiple bonds are unchanged. See
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
