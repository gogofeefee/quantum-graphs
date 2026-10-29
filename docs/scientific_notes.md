# Scientific notes

Conventions, accuracy of the algorithms, and the places where a choice of
definition changes the numbers. Items marked **[open]** are choices that the
code makes explicitly but that a user should be aware of (they were not
changed, to keep published results reproducible).

## 1. Conventions

* Units: lengths in µm, vacuum wavelength λ in µm, k = 2π/λ in µm⁻¹.
* Phase accumulated on a bond: k n_eff(k) L. The group index
  n_g = n_eff + k dn_eff/dk is only used by the Weyl law (unfolding).
* Vertex matrices are indexed σ[port_out, port_in]; directional coupler:
  through amplitude √(1−C), cross amplitude i√C, no back-reflection.
* Directed-bond amplitudes are the amplitudes **arriving** at the end vertex
  (a = U a with U = D Σ). The field on bond (u, v), x measured from u:
  ψ(x) = a_{u→v} e^{i k n (x−L)} + a_{v→u} e^{−i k n x}.
* Open graphs: S[j, i] = amplitude out of lead j for unit amplitude into
  lead i; leads ordered as in `leads`. Losses can be
  modelled with a complex k (`k_imag`): every bond is attenuated by
  exp(−Im(k) n L).
* Self-loops and multiple bonds are ordinary bonds (2.0): the bond
  scattering matrix is indexed by ports, not by vertex pairs. Up to 1.0 they
  were split by virtual pass-through vertices (σ = [[0,1],[1,0]]), which did
  not change spectra, S or resonances (checked to ≤ 1e-13 when they were
  removed) but made U up to 4× larger and the classical step shorter (§6).

## 2. Closed spectrum

`compute_spectrum` scans σ_min(I − U(k)) on `N_scan` uniform k points and
refines each strict local minimum by bounded Brent minimisation.

* Near a simple root σ_min ≈ c |k − k*| (a V-shaped minimum), so Brent stops
  after ~7 evaluations: the refined root has σ_min ~ 1e-6 – 1e-5 and an
  accuracy |δk| ≈ σ_min / (n_eff L_tot) ~ 1e-9 µm⁻¹, about 10⁻⁵ of the mean
  level spacing. Roots with σ_min > `eps_accept` (1e-4) are rejected.
* Two levels closer than ~2 grid steps merge into one candidate. The method
  now warns when the number of levels (with multiplicity) departs from the
  Weyl estimate by more than 5 % (and 3 levels); choose N_scan so that the
  mean spacing π/(n_eff L_tot) spans ≳ 20 grid points.
* Degeneracy = number of singular values ≤ `nullity_tol` (1e-4) of
  I − U_s(k*), times 2 for partner sectors (§4).
* Single-ring test: levels agree with 2πm/(nL) to 5e-8 µm⁻¹.

## 3. Resonances (open graphs)

* `find_resonances` only finds poles inside the scanned rectangle
  Im(k) ∈ [im_min, im_max]. Broader resonances are missed; the
  `plot_resonances_sigma` heat map shows whether σ_min valleys reach the
  lower edge. Single-access ring test: poles agree with
  (2πm + i ln √(1−C)) / (n L) to 1e-8.
* `find_experimental_resonances` keeps the peaks of d²T/dk² above
  `amp_threshold`, an **absolute** value whose scale depends on n_points,
  k_imag and the graph **[open]**: check the three-panel figures.
* S(k) caches: a cache of a wider window is only reused (by linear
  interpolation) if its grid is at least as fine as the requested one; the
  interpolated result is never written back.

## 4. Spectral statistics

* **Sectors** (`quantum_graphs.core.sectors`, 2.0). The strongly connected
  components of the transition pattern Σ_αα[d', d] ≠ 0 (sampled at 9
  wavenumbers of the window) split U(k) into blocks U_s: block diagonal for a
  closed lossless graph (a unitary block-triangular matrix is block diagonal),
  block triangular for an open one; in both cases det(I − U) = Π det(I − U_s).
  A directed bond that cannot come back to itself carries no level and
  belongs to no sector. If Σ restricted to sector s satisfies
  Σ[r(d), r(d')] = Σ[d', d] (r = reversal of a directed bond; true for
  symmetric vertex matrices), U on the reversed bonds r(s) is a permuted
  transpose of D-similar U_s: same spectrum. r(s) = s: time-reversal symmetry
  inside the sector (GOE expected); r(s) ≠ s: two partner sectors, all their
  levels doubly degenerate, no time-reversal symmetry inside each (GUE
  expected). Measured: double ring and single ring = two partner sectors;
  flower and bow-tie graphs (closed and open), tetrahedron = one
  self-reverse sector.
* **Spectrum per sector.** `compute_spectrum` scans one sector of each pair
  of partners; the multiplicity of a level is (number of sectors sharing the
  spectrum) × (nullity of I − U_s, threshold `nullity_tol`). `find_resonances`
  still searches the whole U and assigns each resonance to the independent
  sector where σ_min(I − U_s) is smallest, with the same multiplicity rule.
  A level degenerate *inside* a sector (geometric symmetry, accidental
  near-degeneracy) is reported by a warning: it gives zero spacings.
  Before 2.0 the degeneracy was measured only, and a single level found with
  g = 1 among pairs (merged pair, window edge) switched the whole analysis to
  the expanded spectrum, silently.
* **Unfolding**: x_i = N(k_i) − N(k_1), N(k) = (L_sel/2π) ∫ n_g dk, with
  L_sel the summed length of the directed bonds of the selected sectors. The
  default selection is the desymmetrised spectrum (one sector per pair of
  partners; for the double ring L_sel = L, the "Weyl × 1/2" of 1.0); an
  explicit `spectrum` (measured peaks) is unfolded with the same L_sel, so
  give each degenerate pair once. Statistics assume unit mean spacing;
  `quantum_graphs.stats.mean_spacing` checks it. Values are those of 1.0 for
  every reference graph (reference_v5).
* **Thresholds.** One degeneracy threshold, `NULLITY_TOL` = 1e-4 on the
  singular values of I − U(k*), for the spectra and the resonances; the
  eigenvectors of a level of a computed spectrum (entropy, IPR, figures) use
  its stored multiplicity. Up to 1.0 the open-graph wavefunctions used a
  relative threshold of 1e-3, so a resonance could be simple in the
  statistics and degenerate in the entropy (no reference value changed).
* NNSD histograms use `density=True` over the plotted range `xlim`: if
  spacings exceed `xlim[1]`, the histogram is renormalised to that range.
* Σ²(L): windows on a uniform grid of step `L_step`. The GOE/GUE tables
  (Barbara Dietz) agree with the exact Dyson-Mehta formula
  Σ² = L − 2∫(L−r)Y₂ to within 2.2 % (L = 0.5) and 0.7 % (L = 10).
* **Δ₃(L) [open].** The package estimator (`windows="level"`, historical
  default) starts one window on each unfolded level (the level itself is
  excluded). The textbook Dyson-Mehta Δ₃ averages over uniformly placed
  windows (`windows="uniform"`). The two differ for L ≲ 3. Measured on 30
  GOE matrices (N = 600, central half):

  | L | level windows | GOE table | uniform windows | exact Dyson-Mehta |
  |---|---|---|---|---|
  | 1 | 0.039 | 0.039 | 0.060 | 0.0605 |
  | 2 | 0.089 | 0.089 | 0.102 | 0.1024 |
  | 5 | 0.168 | 0.169 | 0.173 | 0.1738 |
  | 10 | 0.234 | 0.235 | 0.235 | 0.2357 |

  The level-anchored estimator is therefore consistent with the GOE/GUE
  tables used in the figures (so the comparison data vs theory is
  apples-to-apples), but both differ from the standard Δ₃ at small L, which
  matters when comparing with values from the literature. Use
  `windows="uniform"` with `stats.delta3_dyson_mehta` for the standard
  definition. The Poisson curve L/15 is the same in both conventions.
  **Decision (author, 2026-10-24):** the level-anchored estimator with the
  Monte-Carlo RMT tables of Barbara Dietz remains the reference; the uniform
  estimator and the analytic curves are kept as a cross-check.
* The closed form used for Δ₃ in one window equals the least-squares
  definition to numerical precision (tested).

## 5. Wavefunction measures

Decided by the author (2026-10-24, plan validated 2026-10-26), implemented in
`quantum_graphs.stats.localization` (reference_v3). Before, the measures were
computed on the virtualised half-bonds, from one arbitrary eigenvector.

* **Bonds.** The distribution is p_b = I_b / Σ I_b over the N_b bonds, with
  I_b = |a_{u→v}|² + |a_{v→u}|², the two amplitudes arriving at the end
  vertices of bond b = (u, v) (directed bonds b and b + N_b). Up to 1.0 the
  same quantity was computed through `physical_bond_arrivals` on the
  virtualised graph; the values are unchanged (reference_v4). At complex k
  the arrival convention makes the result independent of a pass-through
  vertex inserted on the bond (tested).
* I_b is an **intensity**, not an energy (which would be ∝ L_b I_b), as in
  the THG measurement, where p_b is the mean intensity on each branch.
* S_norm = −Σ p_b ln p_b / ln N_b ∈ [0, 1]; IPR = N_b Σ p_b² ∈ [1, N_b]
  (1 when uniform). The unnormalised Σ p_b² ∈ [1/N_b, 1], often quoted in
  the literature, is IPR / N_b. S_norm is undefined (NaN) for one bond.
* **Degenerate levels** (multiplicity g): the eigenvectors returned by the
  SVD are an arbitrary, LAPACK-dependent basis of the eigenspace. The
  intensity is the diagonal of the eigenspace projector divided by g,
  (1/g) Σ_m |a_b^(m)|², i.e. the maximally mixed state of the level; it is
  basis-independent and equals |a_b|² for g = 1. Each level counts once in
  `plot_entropy` / `plot_ipr`. **Caveat:** the entropy is concave, so the
  entropy of this average is ≥ the average entropy of any basis: degenerate
  levels look at least as delocalised as any choice of eigenvectors would
  make them. Alternatives (a symmetry-adapted basis, excluding degenerate
  levels) were discussed and not retained. For directional-coupler graphs
  whose levels are doubly degenerate (two counter-propagating sectors),
  the intensity per bond is the same for every vector of the eigenspace, so
  the choice does not matter there.
* Open graphs: the null space of I − U_b(k) at a resonance has the
  multiplicity found by `find_resonances` (§4); a nearly degenerate pair of
  resonances closer than the threshold is treated as one degenerate level.
* Change of values (paper BTG, tl = 85 µm, closed model, 110 levels in
  1520–1540 nm): S_norm 0.892 ± 0.036 on the 14 half-bonds →
  0.888 ± 0.037 on the 10 physical bonds; IPR 1.55 → 1.52. For a graph
  whose bonds are all split in two (closed flower graph),
  S(half-bonds) = S(physical) + ln 2 exactly.
* Before 2026-10-24 `compute_wavefunction` returned the complex conjugate of
  the eigenvector; |a|² (all figures, entropy, IPR) was unaffected.

## 6. Classical dynamics

P = |Σ_αα|² ignores bond lengths: one step = one bond traversal. Up to 1.0
the virtual vertices made a multiple bond take 2 steps and a self-loop 4,
which changed the classical spectrum (BTG gap 0.121 instead of 1 − 1/√2 ≈
0.293; FG eigenvalues ±i instead of −1). The paper's supplementary material
(Δ_BTG ≈ 0.29, FG eigenvalue −1) was computed without virtual vertices and
agrees with 2.0. For unitary vertex
matrices P is doubly stochastic and the uniform vector is invariant.
Ergodicity is tested as "eigenvalue 1 is simple" and mixing as "all other
eigenvalues have modulus < 1" (tolerance 1e-10).

## 7. Periodic orbits

Orbits are closed walks on directed bonds obeying the coupler rule (enter on
one arm, leave on the other). Before 2026-10-24,
`find_orbits_up_to_length(L_max)` missed every orbit of length between
L_max − (its shortest bond) and L_max (9 of 99 flower-graph orbits up to
1500 µm); the enumeration is now tested against brute force. Searches stopped
by `max_orbits` / `max_search_nodes` raise a warning and set
`orbits_complete = False`. An irreducible orbit visits no vertex twice; up to
1.0 the rule also counted the virtual vertices, which is equivalent
(traversing a bond twice means leaving one of its end vertices twice): the
catalogues of reference_v4 are those of reference_v3. Names "1-4[a]",
"1-4[b]" of orbits sharing a vertex sequence follow the search order, which
depends on the bond order.
