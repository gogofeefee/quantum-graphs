# quantum-graphs

Python tools to simulate **quantum graphs** — networks of one-dimensional
waveguides joined at vertices — in the vertex scattering-matrix formalism
(Kottos & Smilansky; Gnutzmann & Smilansky). The code was written to model
integrated photonic circuits made of waveguides and directional couplers, but
every module accepts arbitrary unitary vertex scattering matrices.

The same graph can be studied from four complementary points of view:

| Module | Package | What it computes |
|--------|---------|------------------|
| [`ClosedGraphs/`](ClosedGraphs)       | `closed_graphs`    | Spectrum of a closed graph (real eigenwavenumbers), eigenvectors, spectral statistics (Weyl law, NNSD, Σ², Δ₃), length spectrum |
| [`OpenGraphs/`](OpenGraphs)           | `open_graphs`      | Scattering matrix S(k) of a graph with external leads, complex resonances, transmission spectra, resonance statistics |
| [`ClassicalGraphs/`](ClassicalGraphs) | `classical_graphs` | Classical (ray) dynamics: Perron-Frobenius operator, ergodicity, mixing, animated probability flow |
| [`PeriodicOrbits/`](PeriodicOrbits)   | `periodic_orbits`  | Enumeration of the periodic orbits of a directional-coupler graph, with their physical and optical lengths |

`wigner/` holds the random-matrix-theory reference curves (GOE, GUE, GSE,
Poisson) used by the spectral statistics, and `data/` a waveguide dispersion
used by some examples.

---

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/gogofeefee/quantum-graphs.git
cd quantum-graphs
pip install -r requirements.txt      # numpy>=2.0, scipy, matplotlib>=3.7, networkx>=3.2, numdifftools
```

There is no install step: each example script adds its package directory to
`sys.path` itself. Keep the folder structure as it is (the packages find
`wigner/` and `data/` relative to their own location).

[ffmpeg](https://ffmpeg.org) is only needed to save the ClassicalGraphs
animations as `.mp4` (use `writer="gif"` otherwise).

## Quick start

```bash
python ClosedGraphs/examples/DoubleRing/config_nconst.py
```

```python
import sys
sys.path.insert(0, "ClosedGraphs")
from closed_graphs import DirectionalGraphs

# one directional coupler whose two arms are closed on themselves: two rings
R1, R2 = 100.0, 100.0 / 2**0.5                       # um
graph = DirectionalGraphs(
    vertex_positions = [(0.0, 0.0)],                 # um, for drawing
    port_map         = [[(0, 2), (0, 3), (0, 0), (0, 1)]],
    lengths          = {((0, 0), (0, 2)): 2 * 3.14159265 * R1,
                        ((0, 1), (0, 3)): 2 * 3.14159265 * R2},
    vertex_couplings = 0.5,                          # power cross-coupling C
    neff             = 1.0,                          # float or function neff(k)
    wavelength_range = (1.48, 1.64),                 # vacuum wavelength, um
    name             = "MyDoubleRing",
)
spectrum, degeneracies = graph.compute_spectrum(N_scan=200_000)
graph.plot_graph(show_lengths=True, save_fig=True)
graph.short_range_correlations(save_fig=True)        # NNSD vs Poisson / GOE / GUE
```

Results (figures, cached spectra, a `graph_parameters.txt` summary) are
written to a folder next to the calling script whose name encodes the
parameters. Every plotting method calls `plt.show()`; to run without opening
windows (ssh, cluster, background job):

```bash
MPLBACKEND=Agg python ClosedGraphs/examples/DoubleRing/config_nconst.py
```

---

## Repository layout

```
quantum-graphs/
├── ClosedGraphs/
│   ├── closed_graphs/        ClosedGraphs, DirectionalGraphs, NNSD_func
│   └── examples/             SingleRing, DoubleRing, Tetrahedron, FlowerGraph, BowTieGraph
├── OpenGraphs/
│   ├── open_graphs/          OpenGraphs, DirectionalGraphs, NNSD_func
│   └── examples/             ring (1 or 2 access couplers), FlowerGraph, BowTieGraph
├── ClassicalGraphs/
│   ├── classical_graphs/     ClassicalGraphs, DirectionalClassicalGraphs
│   └── examples/             FlowerGraph, BowTieGraph
├── PeriodicOrbits/
│   ├── periodic_orbits/      PeriodicOrbits
│   └── examples/             FlowerGraph, BowTieGraph
├── wigner/                   RMT reference curves for Σ²(L) and Δ₃(L)
└── data/                     MPB dispersion neff(k), ng(k) of a silicon waveguide
```

## Examples

Every example is a self-contained script: the geometry is written out
explicitly and nothing outside the repository is needed. The scripts are split
into numbered blocks guarded by `if True:`; set one to `False` to skip it.
Expensive results (spectra, σ_min grids, S-matrices) are cached, so a second
run only redraws the figures.

| Script | Graph | Shows | First run |
|--------|-------|-------|-----------|
| `ClosedGraphs/examples/SingleRing/config_nconst.py`   | one ring (Neumann vertex)                | equally spaced (picket-fence) spectrum; generic `ClosedGraphs` class | ~1.5 min |
| `ClosedGraphs/examples/SingleRing/config_nfunc.py`    | same                                      | dispersive `neff(k)`, `ng(k)` from MPB data | ~1.5 min |
| `ClosedGraphs/examples/DoubleRing/config_nconst.py`   | one coupler, two rings                    | full closed-graph pipeline, exact 2-fold degeneracies | ~1.5 min |
| `ClosedGraphs/examples/DoubleRing/config_nfunc.py`    | same                                      | dispersive index | ~2 min |
| `ClosedGraphs/examples/Tetrahedron/config_nconst.py`  | complete graph K₄, 3-port vertices        | arbitrary vertex matrices (Neumann-Kirchhoff) | ~1.5 min |
| `ClosedGraphs/examples/FlowerGraph/config_nconst.py`  | 5 couplers, 10 bonds (all doubled)        | ~1600 levels, statistics on a sub-window | ~9 min |
| `ClosedGraphs/examples/BowTieGraph/config_nconst.py`  | 5 couplers, 10 bonds (2 doubled)          | same pipeline on a second fabricated device | ~6 min |
| `OpenGraphs/examples/ring/config_n*_1access.py`       | ring + 1 access coupler                   | S-matrix, resonances, "experimental" resonance detection | ~17 min (nconst), ~25 min (nfunc) |
| `OpenGraphs/examples/ring/config_n*_2access.py`       | ring + 2 access couplers (add-drop)       | through / drop ports, 4×4 S-matrix | ~17 min (nconst), ~25 min (nfunc) |
| `OpenGraphs/examples/FlowerGraph/config_nconst.py`    | flower graph + access coupler             | complex resonances, transmission, S-matrix plots, resonance wavefunctions, statistics | ~14 min |
| `OpenGraphs/examples/BowTieGraph/config_nconst.py`    | bow-tie graph + access coupler            | same | ~11 min |
| `ClassicalGraphs/examples/FlowerGraph/config.py`      | flower graph                              | Perron-Frobenius spectrum (ergodic, not mixing), ρ snapshots, animation | ~1 min |
| `ClassicalGraphs/examples/BowTieGraph/config.py`      | bow-tie graph                             | ergodic and mixing (spectral gap ≈ 0.12) | ~1 min |
| `PeriodicOrbits/examples/FlowerGraph/config.py`       | flower graph                              | irreducible orbits, all orbits up to L_max, CSV export, orbit plots | seconds |
| `PeriodicOrbits/examples/BowTieGraph/config.py`       | bow-tie graph                             | same | seconds |

First-run times were measured on one CPU core; later runs reload the cached
results. `OpenGraphs/examples/ring/ring_theory.py` gives the analytical
transmission of the single ring, to compare with the numerical S-matrix. The
ring examples use a fine 1 000 000 × 25 resonance grid and 2 000 000 points
for S(k); lower `N_re` and `n_points` for a quicker first run.

The FlowerGraph and BowTieGraph geometries are those of two fabricated silicon
devices (vertex positions, port connections and bond lengths extracted from
the lithography layout).

---

## Physics and numerics

A graph has B bonds of length L_b, hence 2B **directed bonds**. At vacuum
wavenumber k = 2π/λ:

* `D(k)` is the 2B×2B diagonal propagation matrix, `D_bb = exp(i k n_eff L_b)`;
* `Σ(k)` is the 2B×2B bond-to-bond scattering matrix built from the vertex
  matrices σ⁽ᵛ⁾: entry (b' ← b) is σ⁽ᵛ⁾ between the port where b arrives and
  the port where b' leaves, at their common vertex v;
* `U(k) = D(k) Σ(k)` is the quantum evolution map.

**Closed graphs.** The spectrum is the set of real roots of the secular
equation `det(I − U(k)) = 0`. `compute_spectrum()` evaluates
`g(k) = σ_min(I − U(k))` on `N_scan` points, refines every local minimum with
`scipy.optimize.minimize_scalar`, and takes the degeneracy of each root as the
number of singular values of `I − U(k*)` below `nullity_tol`.
`compute_wavefunction(k*)` returns an orthonormal basis of the null space of
`I − U(k*)` (from the SVD): component b is the amplitude of the wave on
directed bond b = (u → v) when it **arrives** at v. With x the distance to u,
the field on bond (u, v) is

    ψ(x) = a_{u→v} exp(i k n_eff (x − L)) + a_{v→u} exp(−i k n_eff x)

**Open graphs.** Ports with `port_map[i][p] = None` are external leads. The
vertex matrices are split into internal (α) and lead (β) channels and

    S(k) = Σ_ββ + Σ_βα (I − D Σ_αα)⁻¹ D Σ_αβ

Leads are numbered in the order in which the `None` entries appear in
`port_map` (vertex by vertex, port by port); `S[j, i]` is the amplitude
leaving lead j for a unit amplitude injected in lead i. The resonances are
the complex poles of S, found as minima of σ_min(I − D Σ_αα) on a grid of the
complex k-plane (`find_resonances`). `find_experimental_resonances` instead
locates peaks of d²|S_ij|²/dk² on the real axis, as one would on a measured
spectrum.

**Spectral statistics** (both quantum modules). Levels are unfolded with the
Weyl law N(k) ≈ n L_tot k / π (with the group index n_g when n_eff depends on k), then compared with the
Poisson, GOE, GUE and GSE predictions: nearest-neighbour spacing distribution
(Wigner surmises), number variance Σ²(L) and spectral rigidity Δ₃(L).
`plot_length_spectrum` Fourier-transforms the spectrum, giving peaks at the
lengths of the periodic orbits.

**Classical dynamics.** The Perron-Frobenius operator `P = |Σ_αα|²`
(element-wise modulus squared) is column-stochastic and evolves probability
vectors on the directed bonds, ρ_{n+1} = P ρ_n.

**Periodic orbits.** For directional-coupler graphs (a ray entering a coupler
on one arm always leaves on the other arm), `PeriodicOrbits` enumerates the
irreducible orbits (no vertex visited twice) and all orbits up to a given
length. Orbits are named by their sequence of real vertices, made unique under
cyclic rotation and time reversal (e.g. `1-4-5`).

## Describing a graph

All lengths are in µm and all wavelengths are vacuum wavelengths in µm.

```python
vertex_positions = [(x0, y0), (x1, y1), ...]   # um, used for drawing (and default lengths)

port_map[i][p] = (j, q)     # port p of vertex i <-> port q of vertex j (give both directions)
port_map[i][p] = None       # OpenGraphs only: external lead

lengths = {((i, p), (j, q)): L_um, ...}        # one entry per bond; None -> Euclidean distances
```

* Vertex `i` has `len(port_map[i])` ports; its scattering matrix must have the
  same size, with the ports in the same order.
* **Self-loops and multiple bonds are allowed.** At construction they are
  replaced by chains of *virtual* 2-port vertices with σ = [[0, 1], [1, 0]]
  (pure transmission, total length kept), so the physics is unchanged. They
  appear as small black dots on the figures; `virtual_vertex_spacing` only
  changes how they are drawn. The original graph stays in `port_map` /
  `lengths`, the virtualised one is in `virtual_port_map` / `virtual_lengths`.
* Directed-bond indexing: for B (virtualised) bonds, indices `0 … B−1` are the
  bonds in one orientation and `B … 2B−1` the same bonds reversed;
  `graph.directed_bond_index[(src, dst)]` gives the index of a directed bond.
* `neff` can be a constant or a function `neff(k)`; the group index `ng` (used
  by the Weyl law) is derived numerically from `neff(k)` when not given.
* Vertex labels on the figures start at 1 (vertex `i` in the code is drawn as
  `i + 1`), except in `open_graphs`, which labels vertices from 0.

### Directional couplers

The `DirectionalGraphs` / `DirectionalClassicalGraphs` classes model every
(non-virtual) vertex as a lossless 4-port directional coupler with power
cross-coupling C. Ports (0, 1) are one arm, ports (2, 3) the other:

```
σ = [[ 0,      0,      √(1−C), i√C    ],
     [ 0,      0,      i√C,    √(1−C) ],
     [ √(1−C), i√C,    0,      0      ],
     [ i√C,    √(1−C), 0,      0      ]]
```

A coupler never reflects backwards: a wave entering through arm (0, 1)
leaves through arm (2, 3), and vice versa. If every vertex can be given an
"input" and an "output" arm such that every bond joins an output arm to an
input arm (true for DoubleRing), the directed bonds split into two decoupled,
time-reversed sectors: every level is exactly doubly degenerate and each
sector alone has no time-reversal symmetry, so GUE statistics are expected.
The statistics methods detect this case (all degeneracies equal to 2), keep
one level per pair and halve the Weyl law; otherwise (FlowerGraph,
BowTieGraph) the full spectrum is used. `mode=` forces a choice.

C can be a float, a dict `{vertex: C}`, or a function `C(k)`. In `open_graphs`, `internal_coupling` and
`external_coupling` set different values for the graph couplers and for the
couplers that carry leads.

For any other junction, use the base classes (`ClosedGraphs`, `OpenGraphs`,
`ClassicalGraphs`) with `vertex_matrices`: one array per vertex, a single
array for all vertices, or a function `σ(k)` (see the SingleRing and
Tetrahedron examples).

## RMT reference curves on their own

The `wigner/` modules can also be used directly, e.g. to compare any spectrum
with random-matrix theory:

```python
import sys
import numpy as np
sys.path.insert(0, "wigner")
from sigma2_RMT import sigma2_RMT
from delta3_RMT import delta3_RMT

L = np.linspace(0, 20, 5000)
sigma2_goe = sigma2_RMT(L, wigner="GOE")     # number variance Σ²(L)
delta3_gue = delta3_RMT(L, wigner="GUE")     # spectral rigidity Δ₃(L)
```

`wigner` is one of `"GOE"`, `"GUE"`, `"GSE"` (interpolated tables) or
`"poissonian"` (analytic). The nearest-neighbour spacing distributions
(Wigner surmises) are available as `NNSD_func(s, wigner)` in `closed_graphs`
and `open_graphs`.

## References

* T. Kottos and U. Smilansky, *Quantum chaos on graphs*, Phys. Rev. Lett. **79**, 4794 (1997).
* T. Kottos and U. Smilansky, *Periodic orbit theory and spectral statistics for quantum graphs*, Ann. Phys. **274**, 76 (1999).
* S. Gnutzmann and U. Smilansky, *Quantum graphs: applications to quantum chaos and universal spectral statistics*, Adv. Phys. **55**, 527 (2006).
* G. Berkolaiko and P. Kuchment, *Introduction to Quantum Graphs*, AMS (2013).

## Acknowledgements

The GOE and GUE reference tables for Σ²(L) and Δ₃(L) in `wigner/` were
computed by Barbara Dietz.

## Author

**Hugo Girin** — Centre de Nanosciences et de Nanotechnologies (C2N),
Université Paris-Saclay. PhD research on quantum graphs and integrated
photonics.

## License

[MIT](LICENSE) © 2026 Hugo Girin
