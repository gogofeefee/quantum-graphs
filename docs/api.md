# API reference

Everything below is reachable from `import quantum_graphs as qg`. This page
lists every public class, method and function, what it does and what it
returns; the docstrings give the full list of parameters
(`help(qg.ClosedDirectionalGraphs.compute_spectrum)`), and the
[README](../README.md) shows them at work.

**Conventions**

* Lengths and wavelengths in **µm** (vacuum wavelengths), wavenumbers
  k = 2π/λ in **µm⁻¹**. A wavenumber may be complex wherever it makes sense
  (`U`, `S`, `secular_det`).
* A graph has B bonds and 2B **directed bonds**: bond `b = (i, p, j, q, L)`
  gives `d = b` from port (i, p) to port (j, q) and `d = b + B` back.
  Vectors on directed bonds (eigenvectors, classical densities) have 2B
  components in this order.
* **Figures**: every `plot_*` method takes `save_fig=False`, `file_format`
  ("pdf", "png", "svg"...) and `show=None` (default: the `show` option); it
  returns its figure(s), usually `(fig, ax)`. Methods that accept an `ax`
  draw into it and neither save nor show.
* **Files**: results go to `graph.save_path`
  (`<base_dir>/<name>/<parameters>_<hash>/`, created on first use). Cached
  computations are reloaded only if their parameters and the physics
  fingerprint of the graph match.

**Contents** —
[Graph classes](#graph-classes) ·
[Common to every graph](#common-to-every-graph) ·
[Closed graphs](#closed-graphs) ·
[Spectral statistics](#spectral-statistics-closed-and-open-graphs) ·
[Open graphs](#open-graphs) ·
[Classical graphs](#classical-graphs) ·
[Periodic orbits](#periodic-orbits) ·
[Result objects](#result-objects) ·
[stats](#stats) ·
[layouts](#layouts) ·
[dispersion](#dispersion) ·
[Vertex matrices and options](#vertex-matrices-and-options)

---

## Graph classes

| Class | Point of view | Vertex model |
|-------|---------------|--------------|
| `qg.ClosedGraphs` | spectrum of a closed graph | any `vertex_matrices` |
| `qg.ClosedDirectionalGraphs` | same | 4-port directional couplers, `couplings` |
| `qg.OpenGraphs` | scattering by a graph with leads | any `vertex_matrices` (+ `access_matrices`) |
| `qg.OpenDirectionalGraphs` | same | couplers, `couplings` (+ `access_couplings`) |
| `qg.ClassicalGraphs` | ray (Perron-Frobenius) dynamics | any `vertex_matrices` |
| `qg.DirectionalClassicalGraphs` | same | couplers, `couplings` |
| `qg.PeriodicOrbits` | periodic orbits | couplers (the ray rule needs no C) |

`ClosedDirectionalGraphs` is `quantum_graphs.closed_graphs.DirectionalGraphs`
and `OpenDirectionalGraphs` is `quantum_graphs.open_graphs.DirectionalGraphs`.

### Constructor arguments

`vertex_positions` and `bonds` (and `leads` for open graphs) are positional;
everything else is keyword-only.

| Argument | Classes | Meaning |
|----------|---------|---------|
| `vertex_positions` | all | `[(x, y), ...]` in µm, one per vertex; used for drawing and for the length of a bond given without one |
| `bonds` | all | `[(i, p, j, q, L), ...]`: port p of vertex i joined to port q of vertex j by a bond of length L (µm); `(i, p, j, q)`: Euclidean length |
| `leads` | open | `[(v, p), ...]`: ports carrying external leads, in S-matrix order |
| `couplings` | directional | power cross-coupling C of the couplers: a float, a list or dict per vertex, or a function `C(k)` |
| `access_couplings` | open directional | C of the couplers carrying leads (default: `couplings`) |
| `vertex_matrices` | generic | σ of the vertices: one array for all, a list or dict per vertex, or a function `σ(k)` |
| `access_matrices` | `OpenGraphs` | σ of the vertices carrying leads |
| `wavelength_range` | closed, open (required) | `(λ_min, λ_max)` in µm: the window of spectra, resonances and statistics |
| `neff` | closed, open | effective index: a float (default 1.0) or a function `neff(k)` |
| `ng` | closed, open, orbits | group index (float or function); default: derived from `neff`; orbits: converts physical to optical lengths (default 1.0) |
| `bond_spacing` | all | drawing only: distance (µm) between parallel bonds at their middle (float, or a sequence used cyclically, one value per group of parallel bonds) |
| `lead_orientations` | open | drawing only: `{(v, p): angle in degrees}` of the drawn leads |
| `name` | all | name of the output folder |
| `save_path`, `base_dir` | all | output folder, or the folder in which `<name>/...` is created (default: the folder of the running script) |

---

## Common to every graph

Every class above has these attributes and methods.

| Attribute / method | What it does | Returns |
|--------------------|--------------|---------|
| `topology` | the validated graph (see [Topology](#topology)) | `Topology` |
| `n_vertices`, `n_bonds`, `n_leads`, `total_length` | shortcuts to the topology | int, float (µm) |
| `U(k)` | bond evolution map U(k) = D(k) Σ_αα(k) | array (2B, 2B) |
| `sigma_min(k)` | smallest singular value of I − U(k) | float, or array for an array of k |
| `secular_det(k)` | secular determinant det(I − U(k)) | complex |
| `null_vectors(k, multiplicity=None, tol=1e-4)` | orthonormal basis of ker(I − U(k)) | array (n, 2B), one vector per row |
| `bond_intensities(vecs)` | intensity per bond, \|a_{u→v}\|² + \|a_{v→u}\|² (eigenspace average for several rows) | array (B,) |
| `localization_measures(vecs)` | Shannon entropy and inverse participation ratio of the bond intensities | `(S, S_norm, IPR)` |
| `sectors` | independent blocks of U(k) and their reversed partners (see [Sectors](#sectors)) | `Sectors` |
| `model` | the numerical model: D(k), Σ(k), U(k), batched evaluations | `ScatteringModel` |
| `fingerprint`, `physics_spec()` | digest / description of everything that determines the results (stored with every cache) | str, dict |
| `save_path` | output folder, created with `graph_parameters.txt` on first use | `Path` |
| `bond_paths` | drawing curve of every bond (segments, Bézier arcs, teardrop loops) | `BondPaths` |
| `plot_graph(show_lengths=False, show_port_markers=False, ...)` | the graph: bonds, labelled vertices, optionally lengths, port markers (blue: ports 0, 1; orange: 2, 3) and leads; sizes, fonts and `ax` adjustable | `(fig, ax)` |

---

## Closed graphs

`ClosedGraphs`, `ClosedDirectionalGraphs`.

### Spectrum and eigenmodes

| Method | What it does | Returns |
|--------|--------------|---------|
| `compute_spectrum(tol=1e-15, show_progress=False, save_spectrum=True, overwrite=False)` | every level in the window: eigenphase crossings of U_s(k), sector by sector, located to `tol`; cached. Also sets `graph.spectrum`, `graph.degeneracies` | `spectrum` (sorted real k, one per distinct level), `degeneracies` (int, 2 for partner sectors) |
| `compute_wavefunction(k, tol=1e-4)` | basis of ker(I − U(k)) at a level: component d is the amplitude arriving at the end of directed bond d | array (n_modes, 2B) |
| `mode_localization(wavelength_range=None)` | S_norm and IPR of every computed level (a degenerate level once, through its eigenspace projector) | `k, S_norm, ipr` arrays |

### Figures

| Method | What it shows | Returns |
|--------|---------------|---------|
| `plot_wavefunction(a_in, k)` | intensity on every bond, one figure per vector of `a_in` | list of `(fig, ax)` |
| `plot_zeta_B(wavelength_range=None, N_plot=100000, spectrum=None, xlim=None)` | real part, imaginary part and modulus of the secular determinant, with the levels | `(fig, axs)` |
| `plot_entropy(wavelength_range=None, show_smooth=False)` | S_norm of every mode vs k·L_tot and vs index | list of `(fig, ax)` |
| `plot_ipr(wavelength_range=None, show_smooth=False)` | IPR of every mode (dashed: 2, a random real vector) | list of `(fig, ax)` |

Closed graphs also have the [spectral statistics](#spectral-statistics-closed-and-open-graphs)
and the length spectrum below.

---

## Spectral statistics (closed and open graphs)

Shared by closed graphs (levels of `compute_spectrum`) and open graphs (real
parts of the resonances of `find_resonances`).

**Selection of the levels.** Every method takes

* `sector=None`: the desymmetrised spectrum — one sector of each pair of
  partners; an integer selects one sector, `"all"` every level with its
  multiplicity;
* `spectrum=None`: explicit levels instead of the computed ones (e.g. measured
  peaks), unfolded with the Weyl law of the selected sectors;
* `wavelength_range=None`: keep only the levels in this window (µm).

| Method | What it does | Returns |
|--------|--------------|---------|
| `levels(sector=None)` | real wavenumbers of the selected levels, repeated by multiplicity within the selection | array |
| `weyl_length(sector=None)` | directed length L_s of the selection: N(k) = (L_s/2π)∫ n_g dk | float (µm) |
| `weyl_staircase(k, sector=None, k0=0.0)` | smooth counting function N(k) = (L_s/2π)∫_{k0}^{k} n_g dκ | float or array |
| `weyl_count(k_min=None, k_max=None, sector=None)` | Weyl estimate of the number of levels in [k_min, k_max] (default: the window) | float |
| `spectral_ensemble(sector=None)` | random-matrix ensemble expected if the graph is chaotic: "GOE" (time-reversal symmetric sector) or "GUE" | str or None |
| `unfolded_spectrum(spectrum=None, sector=None, wavelength_range=None)` | unfolded levels x_i = N(k_i) − N(k_1), unit mean spacing | array |
| `sigma2(L_max=10, L_step=0.2, ...)` | number variance Σ²(L) | `(L, Σ²)` |
| `delta3(L_max=10, L_step=0.2, ..., windows="level")` | spectral rigidity Δ₃(L) (windows anchored on the levels, or uniform) | `(L, Δ₃)` |

| Figure | What it shows | Returns |
|--------|---------------|---------|
| `plot_staircase_function(plot_weyl=True, weyl_diff=False, ...)` | staircase N(k) with the Weyl law; `weyl_diff`: N − N_Weyl vs index and vs λ | `(fig, ax)` of the staircase |
| `short_range_correlations(width=0.1, xlim=None, plot_wigner=None, cumulative_spacing=False, yscale="linear", ...)` | histogram of the nearest-neighbour spacings with the Wigner surmises (default: Poisson and the expected ensemble), optionally the cumulative distribution | spacings array |
| `long_range_correlations(L_max=10, L_step=0.2, plot_wigner=None, correlation_type="both", ...)` | Σ²(L) and / or Δ₃(L) with the random-matrix curves | `{"sigma2": (L, Σ²), "delta3": (L, Δ₃)}` |
| `plot_length_spectrum(L_max=5000, n_points=100000, spectrum=None, sector="all")` | \|Σ_j exp(−i k_j n_eff ℓ)\|: peaks at the lengths ℓ of the periodic orbits | `(ell, rho_hat)` |

---

## Open graphs

`OpenGraphs`, `OpenDirectionalGraphs`.

### Scattering matrix

| Method | What it does | Returns |
|--------|--------------|---------|
| `S(k)` | S(k) = Σ_ββ + Σ_βα (I − D Σ_αα)⁻¹ D Σ_αβ; `S[j, i]`: amplitude leaving lead j for a unit amplitude injected in lead i | array (n_leads, n_leads), or (m, n_leads, n_leads) for m values of k |

### Resonances

| Method | What it does | Returns |
|--------|--------------|---------|
| `find_resonances(im_min=-1e-4, im_max=0.0, seeds_per_spacing=4, tol=1e-14, ..., overwrite=False, plot=False)` | every pole of S in [k_min, k_max] × [im_min, im_max]: Newton's method with deflation, sector by sector, until the argument-principle count is reached (warns otherwise); cached. Sets `graph.spectrum`, `graph.degeneracies`, `graph.level_sectors`, `graph.resonance_counts` | `spectrum` (complex, sorted by Re k), `degeneracies` |
| `count_resonances(im_min=-1e-4, im_max=0.0, sector=None, cuts=())` | number of poles in the rectangle (and in its sub-rectangles between the `cuts`), by the argument principle | `ZeroCount` |
| `resonance_vectors(index)` | basis of ker(I − U(k)) at resonance `graph.spectrum[index]` | array (multiplicity, 2B) |
| `mode_localization(wavelength_range=None)` | S_norm and IPR of every resonance | `k` (complex), `S_norm`, `ipr` |

### Measured-like spectra

| Method | What it does | Returns |
|--------|--------------|---------|
| `find_experimental_resonances(n_points=5_000_000, component=(1, 0), k_imag=1e-7, prominence=None, ...)` | peaks of d²T/dk², T = \|S_ij\|² on a real grid, as on a measured transmission; manual additions / exclusions; sets `graph.exp_spectrum` | sorted real k |
| `best_spectrum` | the resonances if computed, else the peaks above, else None | array or None |

### Figures

The S-matrix figures take `(lambda_min, lambda_max, n_points, components)`
(µm, number of k points, `(i, j)` or a list of them) and cache S(λ).

| Method | What it shows | Returns |
|--------|---------------|---------|
| `plot_S_amplitude(..., spectrum=None, k_imag=0.0, x_axis="wavelength")` | \|S_ij\|² vs wavelength or frequency, resonances marked | `(fig, ax)` |
| `plot_S_phase(...)` | arg S_ij | `(fig, ax)` |
| `plot_S_amplitude_phase(...)` | \|S_ij\|² and arg S_ij on a shared axis | `(fig, axs)` |
| `plot_S_complex_plane(...)` | S_ij(λ) as a curve in the complex plane, crossings of the negative real axis | `(fig, axs)` |
| `plot_S_fourier(..., xlim_um=(0, 8000))` | Fourier transform of \|S_ij\|² vs optical length | `(fig, axs)` |
| `plot_resonances_sigma(lam_min=None, lam_max=None, ..., plot_heatmap=True)` | resonances in the complex k-plane over a σ_min(I − U) map | `(fig, ax)` |
| `plot_energy_distribution(k, injection_port, show_leads=True)` | intensity on the bonds for light injected in one lead, with \|S_ji\|² on the leads | `(fig, ax)` |
| `plot_resonance_wavefunction(index)` | intensity on the bonds of a resonance, one figure per vector | list of `(fig, ax)` |
| `plot_entropy`, `plot_ipr` | localisation of every resonance (as for closed graphs) | list of `(fig, ax)` |

Open graphs also have every [spectral statistic](#spectral-statistics-closed-and-open-graphs).

---

## Classical graphs

`ClassicalGraphs`, `DirectionalClassicalGraphs`. `k` is needed only for
vertex matrices that depend on k.

| Method | What it does | Returns |
|--------|--------------|---------|
| `perron_frobenius(k=None)` | P = \|Σ_αα\|² (element-wise), column-stochastic for unitary vertices | array (2B, 2B) |
| `perron_frobenius_spectrum(k=None)` | eigenvalues of P by decreasing modulus and right eigenvectors | `(eigenvalues, eigenvectors)` |
| `spectral_gap(k=None)` | 1 − \|μ₂\| (0 if \|μ₂\| = 1) | float |
| `is_ergodic(k=None)`, `is_mixing(k=None)` | eigenvalue 1 simple; and every other \|μ\| < 1 | bool |
| `invariant_measure(k=None)` | ρ* with P ρ* = ρ* (uniform for unitary vertices) | array (2B,) |
| `classical_summary(k=None)` | all of the above | dict |
| `basis_state(index=None, dbond=None)` | probability 1 on one directed bond (index or `(i, p, j, q)`) | array (2B,) |
| `evolve_rho(rho0, n_steps=1, trajectory=False)` | ρ_n = Pⁿ ρ₀ (or every step) | array (2B,) or (n_steps + 1, 2B) |
| `validate_rho(rho, normalize=False)` | checks (and normalises) a probability vector | array |
| `plot_rho(rho, ax=None, rho_max=1.0, ...)` | arrows on the directed bonds, width ∝ ρ | `(fig, ax)` |
| `plot_rho_after_steps(rho0, n_steps, ...)` | evolves, then `plot_rho` | `(rho_n, fig, ax)` |
| `animate_rho(rho0, n_steps, style="flow", fps=15, save_animation=False, writer="mp4")` | animation of the probability flow (`writer="gif"` without ffmpeg) | `(anim, trajectory)` |

---

## Periodic orbits

`PeriodicOrbits(vertex_positions, bonds, *, ng=None, ...)` for graphs of
directional couplers: a ray entering on one arm (ports 0, 1) leaves on the
other (ports 2, 3).

| Method | What it does | Returns |
|--------|--------------|---------|
| `find_irreducible_orbits(max_orbits=200000)` | every orbit visiting no vertex twice (finitely many; every orbit is a concatenation of them); sets `graph.irreducible_orbits` | `{name: record}` |
| `find_orbits_up_to_length(L_max_um, max_orbits=20000, max_search_nodes=5000000)` | every orbit of physical length ≤ L_max_um (their number grows exponentially); sets `graph.orbits` | `{name: record}` |
| `export_orbits_csv(path=None, which="all")` | CSV of `graph.orbits` (`"all"`) or `graph.irreducible_orbits` | `Path` |
| `directed_bond_ports(d)` | ports of directed bond d | `(i, p, j, q)` |
| `plot_graph(ax=None, show_lengths=False, show_port_markers=True)` | the graph in light grey, so that orbits stand out | `(fig, ax)` |
| `plot_orbit(name, color=None, ax=None, ...)` | one orbit as arrows along its bonds | `(fig, ax)` |

An orbit **record** is a dict with `name` (vertex labels, unique up to
rotation and time reversal, e.g. `"1-4-5"`), `vertex_sequence`, `path`
(directed bonds), `path_ports`, `n_bonds`, `physical_length_um`,
`optical_length_um` (× `ng`), `is_irreducible`, `is_primitive`,
`multiplicity` and `primitive_name` (for repetitions of a shorter orbit).

---

## Result objects

### Topology

`graph.topology` (`quantum_graphs.core.topology.Topology`), read-only.

| Attribute / method | Meaning |
|--------------------|---------|
| `bonds`, `leads`, `length`, `valency`, `positions` | the input, validated (arrays) |
| `n_vertices`, `n_bonds`, `n_directed`, `n_leads`, `total_length` | sizes |
| `src_vertex`, `src_port`, `dst_vertex`, `dst_port`, `directed_length` | arrays indexed by directed bond d |
| `reverse(d)`, `bond_of(d)`, `is_forward(d)` | directed-bond numbering (int or array) |
| `leaving(v, p)`, `arriving(v, p)`, `directed(i, p, j, q)` | directed bond leaving / arriving at a port, or joining two ports |
| `bond_id(i, p, j, q)`, `lead_index(v, p)`, `vertex_bonds(v)`, `parallel_groups` | lookups |
| `vertex_labels` | labels of the figures: 1, 2, … in index order; `A` (`A1`, `A2`…) for the vertices with leads |
| `as_dict()` | `dict(vertex_positions, bonds, leads)`, to build another graph on the same topology |

### Sectors

`graph.sectors` (`quantum_graphs.core.sectors.Sectors`): a tuple of `Sector`,
the strongly connected components of the transition pattern of Σ.

| Attribute / method | Meaning |
|--------------------|---------|
| `Sector.directed` | its directed bonds |
| `Sector.length` | L_s, the summed length of its directed bonds (µm) |
| `Sector.reverse` | index of the sector made of the reversed bonds (itself if `time_reversal`) |
| `Sector.time_reversal` | the sector is its own reverse |
| `Sector.ensemble` | "GOE" or "GUE" |
| `Sectors.independent` | one representative per distinct spectrum |
| `Sectors.multiplicity(i)`, `Sectors.class_of(i)` | 2 for partner sectors; representative of sector i |

### ZeroCount

Returned by `count_resonances` and stored in `graph.resonance_counts`.

| Attribute | Meaning |
|-----------|---------|
| `total` | number of zeros in the rectangle, with multiplicity |
| `counts`, `edges` | zeros in each sub-rectangle between consecutive `edges` (Re k) |
| `winding`, `defect` | winding numbers before rounding, and their largest distance to an integer (≪ 1) |
| `im_min`, `im_max`, `n_samples` | the rectangle and the number of determinant evaluations |

---

## stats

`quantum_graphs.stats`: pure functions on arrays (`qg.nnsd`, `qg.sigma2_rmt`,
`qg.delta3_rmt` are also top-level).

| Function | What it does | Returns |
|----------|--------------|---------|
| `weyl_staircase(k, directed_length, ng, k0=0.0)` | N(k) = (L_s/2π)∫_{k0}^{k} n_g dκ | float or array |
| `unfold(spectrum, directed_length, ng, k0=0.0)` | sorted unfolded levels N(k_i) | array |
| `spacings(unfolded)`, `mean_spacing(unfolded)` | nearest-neighbour spacings, their mean (≈ 1) | array, float |
| `number_variance(unfolded, dE, L_max)` | Σ²(L) = ⟨(n(E, L) − L)²⟩ | `(L, Σ²)` |
| `spectral_rigidity(unfolded, dE, L_max, windows="level")` | Δ₃(L), least-squares deviation from a straight line | `(L, Δ₃)` |
| `nnsd(s, ensemble="GOE")` | Wigner surmise / Poisson P(s) | array |
| `sigma2_rmt(L, ensemble="GOE")`, `delta3_rmt(L, ensemble="GOE")` | random-matrix curves: Poisson analytic, GOE / GUE / GSE tabulated (B. Dietz), NaN outside the table | array |
| `sigma2_dyson_mehta(L, ensemble)`, `delta3_dyson_mehta(L, ensemble)` | exact Dyson-Mehta formulas (validation of the tables) | array |
| `table_range(quantity, ensemble)` | L range of a table | `(L_min, L_max)` |
| `normalize_ensemble(name)` | canonical name: 'poisson', 'goe', 'gue', 'gse' | str |
| `eigenspace_intensity(vecs)`, `bond_intensity(I)`, `localization_measures(I)` | intensity per directed bond (eigenspace average), per bond, and (S, S_norm, IPR) | arrays, tuple |

---

## layouts

`quantum_graphs.layouts`: ready-made geometries, as dicts to unpack into a
constructor (`qg.ClosedDirectionalGraphs(**qg.layouts.double_ring(), ...)`).

| Function | Graph |
|----------|-------|
| `single_ring(radius=100.0)` | one 2-port vertex closed on itself (use `qg.PASS_THROUGH` or a Neumann matrix) |
| `double_ring(r1=100.0, r2=r1/√2)` | one coupler whose two arms are closed into rings |
| `ring_with_access(radius=100.0, n_access=1)` | open ring with one access coupler, or two (add-drop) |
| `tetrahedron()` | complete graph K₄ of 3-port vertices |
| `flower_graph(open=False)` | five couplers, every neighbouring pair joined by two bonds (fabricated device, arXiv:2605.12538); `open=True` with its access coupler |
| `bow_tie_graph(open=False)` | five couplers, 10 bonds of which two are doubled (fabricated device, arXiv:2605.12538) |
| `closed(layout)` | copy of a layout without its leads |
| `remove_access_coupler(layout, vertex=0)` | deletes a 2-bond access coupler and merges its two bonds |
| `from_port_map(vertex_positions, port_map, lengths=None)` | converts the 1.x description (`port_map[i][p] = (j, q)`) to a bond list |

---

## dispersion

`quantum_graphs.dispersion`: tabulated indices (e.g. from MPB), to pass as
`neff` and `ng`. The repository ships the dispersion of a silicon waveguide
in `data/mpb/`.

| Function | What it does | Returns |
|----------|--------------|---------|
| `load_mpb_indices(neff_npz_path, ng_npz_path)` | reads `k_vac`, `neff` / `ng` arrays and interpolates them | `k_neff, neff_raw, neff_func, k_ng, ng_raw, ng_func` |
| `apply_index_scaling(neff, ng, scale=1.0)` | multiplies both indices (numbers or functions) | `(neff, ng)` |
| `plot_mpb_indices_vs_k(...)` | the tables and their interpolants | `(fig, (ax1, ax2))` |

---

## Vertex matrices and options

| Name | What it does | Returns |
|------|--------------|---------|
| `qg.directional_coupler_matrix(C)` | lossless 4-port coupler with power cross-coupling C (README, *Vertex matrices*) | array (4, 4) |
| `qg.neumann_matrix(n)` | Neumann-Kirchhoff vertex, σ = (2/n) J − I | array (n, n) |
| `qg.PASS_THROUGH` | 2-port vertex that transmits everything | array (2, 2) |
| `qg.set_options(verbose=..., show=..., n_workers=...)` | global options (also `QUANTUM_GRAPHS_VERBOSE`, `QUANTUM_GRAPHS_SHOW`, `QUANTUM_GRAPHS_WORKERS`) | — |
| `qg.options(**kwargs)` | the same as a context manager | context manager |
| `qg.get_option(name)` | current value of an option | value |
