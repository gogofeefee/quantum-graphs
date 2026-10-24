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
  lead i; leads ordered as the `None` entries of `port_map`. Losses can be
  modelled with a complex k (`k_imag`): every bond is attenuated by
  exp(−Im(k) n L).
* Virtual vertices (pass-through σ = [[0,1],[1,0]]) split self-loops (4
  segments) and multiple bonds (2 segments); they never change the physics.
  Quantities resolved per bond (entropy, IPR, energy distribution) are,
  however, computed on the virtualised half-bonds (§5).

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
* Degeneracy = number of singular values ≤ `nullity_tol` (1e-4).
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

* Unfolding: x_i = N_Weyl(k_i) = (L_eff/π) ∫ n_g dk. Statistics assume unit
  mean spacing; `quantum_graphs.stats.mean_spacing` checks it.
* Directional graphs with all levels exactly doubly degenerate: one level per
  pair and L_eff = L/2 (`mode="directional"`); otherwise the expanded
  spectrum (each level repeated by its degeneracy). A *mixture* of single and
  double levels is treated as expanded, which puts zero spacings in the
  NNSD **[open]**.
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

## 5. Wavefunction measures **[open — decided, not yet implemented]**

Decision (author, 2026-10-24): entropy and IPR should be computed per
**physical** bond (independent of the virtualisation); to be implemented.


* Entropy and IPR use p_b ∝ |a_{b,fwd}|² + |a_{b,bwd}|² per **virtualised**
  bond, i.e. intensity, not energy (which would be ∝ L_b × intensity), and
  each half of a split bond counts separately (normalisation ln N_b with the
  virtual count). For the flower graph every physical bond is split, so
  S(half-bonds) = S(physical) + ln 2 exactly.
* For a degenerate level the eigenvectors form an arbitrary orthonormal basis
  of the eigenspace (from the SVD): entropy, IPR and plots of `vecs[0]` then
  depend on that arbitrary choice.
* Before 2026-10-24 `compute_wavefunction` returned the complex conjugate of
  the eigenvector; |a|² (all figures, entropy, IPR) was unaffected.

## 6. Classical dynamics

P = |Σ_αα|² ignores bond lengths (one step per bond). For unitary vertex
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
`orbits_complete = False`.
