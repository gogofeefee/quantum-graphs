# quantum-graphs

Python tools to simulate **quantum graphs** — networks of one-dimensional
waveguides joined at vertices — in the vertex scattering-matrix formalism
(Kottos & Smilansky; Gnutzmann & Smilansky). The code was written to model
integrated photonic circuits made of waveguides and directional couplers, but
every module accepts arbitrary vertex scattering matrices.

The same graph can be studied from four complementary points of view:

| Subpackage | What it computes |
|------------|------------------|
| `quantum_graphs.closed_graphs`    | Spectrum of a closed graph, eigenvectors, spectral statistics (Weyl law, NNSD, Σ², Δ₃), length spectrum |
| `quantum_graphs.open_graphs`      | Scattering matrix S(k) of a graph with external leads, complex resonances, transmission spectra, resonance statistics |
| `quantum_graphs.classical_graphs` | Classical (ray) dynamics: Perron-Frobenius operator, ergodicity, mixing, animated probability flow |
| `quantum_graphs.periodic_orbits`  | Enumeration of the periodic orbits of a directional-coupler graph, with physical and optical lengths |

plus `quantum_graphs.stats` (unfolding, NNSD, Σ², Δ₃ and random-matrix
reference curves), `quantum_graphs.layouts` (ready-made geometries) and
`quantum_graphs.dispersion` (tabulated n_eff(k), n_g(k)).

---

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/gogofeefee/quantum-graphs.git
cd quantum-graphs
pip install -e .                 # or: pip install -e ".[test]" to run the tests
```

[ffmpeg](https://ffmpeg.org) is only needed to save the ClassicalGraphs
animations as `.mp4` (use `writer="gif"` otherwise).

## Quick start

```python
import quantum_graphs as qg

layout = qg.layouts.double_ring()              # vertex_positions, bonds (lengths in µm)
graph = qg.ClosedDirectionalGraphs(
    **layout,
    couplings=0.5,                      # power cross-coupling C of every coupler
    neff=1.0,                                  # float, or a function neff(k)
    wavelength_range=(1.48, 1.64),             # vacuum wavelength window, µm
    name="MyDoubleRing",
)
spectrum, degeneracies = graph.compute_spectrum(N_scan=200_000)
graph.plot_graph(show_lengths=True, save_fig=True)
graph.short_range_correlations(save_fig=True)  # NNSD vs Poisson / GOE
```

Results (figures, cached computations, a `graph_parameters.txt` summary) are
written to `<script folder>/<name>/<parameters>/`. To start a study on your
own graph, copy [`examples/new_study_template.py`](examples/new_study_template.py).

**Options.** Every plotting method calls `plt.show()` and computations print
progress messages. For batch runs:

```python
qg.set_options(show=False, verbose=False)      # or: with qg.options(show=False): ...
```

or the environment variables `QUANTUM_GRAPHS_SHOW=0`,
`QUANTUM_GRAPHS_VERBOSE=0`. Scans run in `n_workers` threads (default: half
the cores, at most 8; `QUANTUM_GRAPHS_WORKERS`); results do not depend on it.

---

## Repository layout

```
quantum-graphs/
├── quantum_graphs/
│   ├── core/              topology & validation, vertex matrices, scattering model, caches, options
│   ├── closed_graphs/     ClosedGraphs, DirectionalGraphs
│   ├── open_graphs/       OpenGraphs, DirectionalGraphs
│   ├── classical_graphs/  ClassicalGraphs, DirectionalClassicalGraphs
│   ├── periodic_orbits/   PeriodicOrbits
│   ├── stats/             spectral statistics + RMT tables
│   ├── plotting/          shared figure style
│   ├── layouts.py         reference geometries, remove_access_coupler
│   └── dispersion.py      MPB / tabulated dispersion loaders
├── examples/              one folder per subpackage, one sub-folder per graph
├── data/                  MPB dispersion of a silicon waveguide (used by examples)
├── tests/                 pytest suite + numerical reference data set
└── docs/                  architecture and scientific notes
```

The top-level namespace gives unambiguous names to the main classes:
`qg.ClosedGraphs`, `qg.ClosedDirectionalGraphs`, `qg.OpenGraphs`,
`qg.OpenDirectionalGraphs`, `qg.ClassicalGraphs`,
`qg.DirectionalClassicalGraphs`, `qg.PeriodicOrbits`.

## Examples

Every example is a self-contained script; the scripts are split into numbered
blocks guarded by `if True:` (set one to `False` to skip it). Expensive
results are cached, so a second run only redraws the figures.

| Script (under `examples/`) | Graph | Shows |
|--------|-------|-------|
| `closed_graphs/SingleRing/config_nconst.py`   | one ring (pass-through vertex)        | equally spaced (picket-fence) spectrum; generic `ClosedGraphs` class |
| `closed_graphs/SingleRing/config_nfunc.py`    | same                                   | dispersive `neff(k)`, `ng(k)` from MPB data |
| `closed_graphs/DoubleRing/config_nconst.py`   | one coupler, two rings                 | full closed-graph pipeline, exact 2-fold degeneracies |
| `closed_graphs/DoubleRing/config_nfunc.py`    | same                                   | dispersive index |
| `closed_graphs/Tetrahedron/config_nconst.py`  | complete graph K₄, 3-port vertices     | arbitrary vertex matrices (Neumann-Kirchhoff) |
| `closed_graphs/FlowerGraph/config_nconst.py`  | 5 couplers, 10 bonds (all doubled)     | ~1600 levels, statistics on a sub-window |
| `closed_graphs/BowTieGraph/config_nconst.py`  | 5 couplers, 10 bonds (2 doubled)       | same pipeline on a second fabricated device |
| `open_graphs/ring/config_n*_1access.py`       | ring + 1 access coupler                | S-matrix, resonances, transmission-based detection |
| `open_graphs/ring/config_n*_2access.py`       | ring + 2 access couplers (add-drop)    | through / drop ports, 4×4 S-matrix |
| `open_graphs/ring/ring_theory.py`             | analytic ring resonator                | closed-form transmission to compare with |
| `open_graphs/FlowerGraph/config_nconst.py`    | flower graph + access coupler          | resonances, transmission, S-matrix plots, resonance wavefunctions, statistics |
| `open_graphs/BowTieGraph/config_nconst.py`    | bow-tie graph + access coupler         | same |
| `classical_graphs/FlowerGraph/config.py`      | flower graph                           | Perron-Frobenius spectrum (ergodic, not mixing), ρ snapshots, animation |
| `classical_graphs/BowTieGraph/config.py`      | bow-tie graph                          | ergodic and mixing (spectral gap ≈ 0.12) |
| `periodic_orbits/FlowerGraph/config.py`       | flower graph                           | irreducible orbits, all orbits up to L_max, CSV export, orbit plots |
| `periodic_orbits/BowTieGraph/config.py`       | bow-tie graph                          | same |
| `new_study_template.py`                       | your graph                             | the four points of view on one graph |

First runs take from seconds (orbits, classical) to a few minutes (open-graph
resonance scans); see the header of each script. The FlowerGraph and
BowTieGraph geometries are those of two fabricated silicon devices (also
available as `qg.layouts.flower_graph()` and `qg.layouts.bow_tie_graph()`,
`open=True` for the version with its access coupler).

---

## Physics and numerics

A graph has B bonds of length L_b, hence 2B **directed bonds**. At vacuum
wavenumber k = 2π/λ:

* `D(k)` is the 2B×2B diagonal propagation matrix, `D_bb = exp(i k n_eff L_b)`;
* `Σ(k)` is the 2B×2B bond-to-bond scattering matrix built from the vertex
  matrices σ⁽ᵛ⁾: entry (b' ← b) is σ⁽ᵛ⁾ between the port where b arrives and
  the port where b' leaves, at their common vertex v;
* `U(k) = D(k) Σ(k)` is the quantum evolution map.

**Closed graphs.** The spectrum is the set of real roots of
`det(I − U(k)) = 0`. `compute_spectrum()` evaluates `σ_min(I − U(k))` on
`N_scan` points, refines every local minimum (bounded Brent minimisation) and
takes the degeneracy of each root as the number of singular values of
`I − U(k*)` below `nullity_tol`; it warns if the number of levels departs from
the Weyl law. The scan runs sector by sector (see *Spectral statistics*): the
levels of two partner sectors are computed once and are exactly doubly
degenerate by construction. `compute_wavefunction(k*)` returns an orthonormal basis of the
null space of `I − U(k*)`: component b is the amplitude of the wave on
directed bond b = (u → v) when it **arrives** at v. With x the distance to u,
the field on bond (u, v) is

    ψ(x) = a_{u→v} exp(i k n_eff (x − L)) + a_{v→u} exp(−i k n_eff x)

`plot_entropy` / `plot_ipr` (and `graph.localization_measures(a)`) measure the
localisation of a mode on the bonds, with
p_b ∝ |a_{u→v}|² + |a_{v→u}|²: S_norm = −Σ p_b ln p_b / ln N_b and
IPR = N_b Σ p_b² (1 when uniform). A degenerate level counts once, through the
projector onto its eigenspace.

**Open graphs.** The ports listed in `leads` carry external leads, and

    S(k) = Σ_ββ + Σ_βα (I − D Σ_αα)⁻¹ D Σ_αβ

Leads are numbered in the order of `leads`; `S[j, i]` is the amplitude
leaving lead j for a unit amplitude
injected in lead i. Resonances (poles of S) are the zeros of
det(I − D Σ_αα) in a rectangle of the complex k-plane (`find_resonances`):
their number is counted by the argument principle (`count_resonances`) and
they are found by Newton's method, with deflation until the count is
reached, sector by sector (about one second for the flower graph over
1.48–1.64 µm); a dispersive n_eff is continued off the real axis to first
order, so that the widths follow the group index;
`find_experimental_resonances` locates peaks of d²|S_ij|²/dk² on the real
axis, as one would on a measured spectrum.

**Sectors.** A wave on a directed bond only scatters into the directed bonds
reached through the nonzero entries of Σ. The strongly connected components
of this pattern (`graph.sectors`) split U(k) into independent blocks U_s, and
the spectrum is the union of their spectra. With reciprocal vertices
(σ symmetric), the reversed bonds of a sector form a sector with the same
spectrum: either the sector itself (time-reversal symmetry, GOE expected —
flower and bow-tie graphs) or a partner (every level doubly degenerate, GUE
expected for each sector — double ring, rings of couplers).

**Spectral statistics** use the desymmetrised spectrum (one sector of each
pair of partners; `sector=` selects another one, `sector="all"` every level)
unfolded with the Weyl law of the selected sectors,
N(k) = (L_s/2π)∫ n_g dk, L_s the summed length of their directed bonds (2L
for a graph made of one sector, L for one of two partners). They are compared
with Poisson and the expected ensemble (GOE / GUE; GSE also available):
nearest-neighbour spacings (Wigner surmises), number variance Σ²(L)
(`sigma2()`) and spectral rigidity Δ₃(L) (`delta3()`). Measured peaks are
analysed with `spectrum=graph.exp_spectrum`. `plot_length_spectrum`
Fourier-transforms the spectrum, with peaks at the lengths of periodic
orbits.

**Classical dynamics.** `P = |Σ_αα|²` (element-wise) is doubly stochastic and
evolves probability vectors on the directed bonds, ρ_{n+1} = P ρ_n.

**Periodic orbits.** For directional-coupler graphs (a ray entering a coupler
on one arm leaves on the other), `PeriodicOrbits` enumerates the irreducible
orbits and all orbits up to a given length (validated against a brute-force
enumeration); orbits are named by their sequence of vertices, unique up
to rotation and time reversal (e.g. `1-4-5`).

The conventions, the accuracy of each algorithm and the points where a
choice of definition matters (Δ₃ windows, entropy weights, degenerate
levels...) are detailed in [docs/scientific_notes.md](docs/scientific_notes.md).

## Describing a graph

All lengths are in µm and all wavelengths are vacuum wavelengths in µm.

```python
vertex_positions = [(x0, y0), (x1, y1), ...]   # um, used for drawing (and default lengths)
bonds = [(i, p, j, q, L), ...]   # port p of vertex i <-> port q of vertex j, length L (um);
                                 # (i, p, j, q) without L: Euclidean distance
leads = [(v, p), ...]            # open graphs only: external leads, in S-matrix order
```

For example, two couplers joined by two bonds, with leads on vertex 0:

```python
graph = qg.OpenDirectionalGraphs(
    vertex_positions=[(0, 0), (100, 0)],
    bonds=[(0, 1, 1, 0, 120.0), (0, 3, 1, 2, 180.0),   # two bonds between 0 and 1
           (1, 1, 1, 3, 400.0)],                       # a self-loop on vertex 1
    leads=[(0, 0), (0, 2)],
    couplings=0.5, access_couplings=0.3, neff=2.4,
    wavelength_range=(1.50, 1.60))
```

The inputs are validated: a port used twice or left unconnected (the ports of
a vertex are numbered 0 … valency−1), a non-positive length, or a vertex matrix
whose size differs from the vertex valency raise a `ValueError`.

* **Self-loops and multiple bonds are ordinary bonds.** Figures draw parallel
  bonds as symmetric arcs (`bond_spacing` sets their distance) and self-loops
  as teardrops of their true length; this geometry is for drawing only.
* `graph.topology` holds the validated graph as read-only arrays: `bonds`,
  `leads`, `length`, `valency`, `positions`, and the directed bonds — bond
  `b = (i, p, j, q)` gives `d = b` from (i, p) to (j, q) and `d = b + B` back —
  with `src_vertex`, `src_port`, `dst_vertex`, `dst_port`, `directed_length`
  and the lookups `reverse(d)`, `bond_of(d)`, `is_forward(d)`, `leaving(v, p)`, `arriving(v, p)`,
  `directed(i, p, j, q)`. `graph.n_vertices`, `n_bonds`, `n_leads` and
  `total_length` are shortcuts.
* `neff` can be a constant or a function `neff(k)`; the group index `ng` is
  derived numerically from it when not given.
* Vertex labels on the figures (`graph.vertex_labels`) start at 1: the
  vertices are numbered 1, 2, … in index order, except those carrying
  external leads (access couplers), labelled `A` (`A1`, `A2`, … if there are
  several). With the access coupler as vertex 0, an open graph and the closed
  graph obtained by `layouts.remove_access_coupler` have the same labels.

### Vertex matrices

`DirectionalGraphs` / `DirectionalClassicalGraphs` model every real vertex as
a lossless 4-port directional coupler with power cross-coupling C (a float, a
list or dict per vertex, or a function `C(k)`); ports (0, 1) are one arm,
(2, 3) the other:

```
σ = [[ 0,      0,      √(1−C), i√C    ],
     [ 0,      0,      i√C,    √(1−C) ],
     [ √(1−C), i√C,    0,      0      ],
     [ i√C,    √(1−C), 0,      0      ]]
```

In `open_graphs`, `access_couplings` (default: `couplings`) sets the couplers
that carry leads. For any other junction, use the base classes with
`vertex_matrices` (one array per vertex, one array for all, or a function
σ(k); `access_matrices` for the vertices with leads of an open graph);
`qg.neumann_matrix(n)` gives the Neumann-Kirchhoff matrix.

### Caching and reproducibility

Each cached file (spectrum, σ_min grid, resonances, S(k) samples) is stored
with a `.meta.json` record of the computation parameters and of a *physics
fingerprint* of the graph (topology, lengths, vertex matrices and indices,
functions being sampled numerically). A cache is reused only if both match;
otherwise it is recomputed with a warning. Files written by versions before
1.0 have no record: they are loaded with a warning (`overwrite=True` to
recompute). `graph_parameters.txt` describes the graph and its fingerprint.

## Tests

```bash
pip install -e ".[test]"
pytest                 # fast suite (~30 s)
pytest -m slow         # scientific validation and non-regression (~2 min)
```

The tests check analytic cases (ring spectrum, all-pass ring S-matrix and
poles), physical properties (unitarity and reciprocity of S, stochasticity of
P, Weyl law), the RMT reference curves against the exact Dyson-Mehta
formulas, the orbit enumeration against brute force, the cache invalidation,
and reproduce a stored numerical reference data set (`tests/reference/`).
[docs/architecture.md](docs/architecture.md) explains how the code is
organised and how to extend it.

## Acknowledgements

The GOE and GUE reference tables for Σ²(L) and Δ₃(L) were computed by
Barbara Dietz.

## References

* T. Kottos and U. Smilansky, *Quantum chaos on graphs*, Phys. Rev. Lett. **79**, 4794 (1997).
* T. Kottos and U. Smilansky, *Periodic orbit theory and spectral statistics for quantum graphs*, Ann. Phys. **274**, 76 (1999).
* S. Gnutzmann and U. Smilansky, *Quantum graphs: applications to quantum chaos and universal spectral statistics*, Adv. Phys. **55**, 527 (2006).
* G. Berkolaiko and P. Kuchment, *Introduction to Quantum Graphs*, AMS (2013).
* M. L. Mehta, *Random Matrices*, 3rd ed., Elsevier (2004).

## Author

**Hugo Girin** — Centre de Nanosciences et de Nanotechnologies (C2N),
Université Paris-Saclay. PhD research on quantum graphs and integrated
photonics.

## License

[MIT](LICENSE) © 2026 Hugo Girin
