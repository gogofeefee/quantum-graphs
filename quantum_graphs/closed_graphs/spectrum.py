"""
SpectrumMixin — U(k), secular function, eigenwavenumbers and eigenvectors.

Algorithm of :meth:`compute_spectrum`
-------------------------------------
For each independent sector s (one per pair of reciprocal partners, see
:mod:`quantum_graphs.core.sectors`):

1. g(k) = σ_min(I − U_s(k)) on ``N_scan`` points uniformly spaced in k
   (batched SVDs);
2. every strict local minimum of g is refined by bounded Brent minimisation
   on the two neighbouring grid intervals (``xatol=tol``);
3. roots with g(k*) ≤ ``eps_accept`` are kept; the multiplicity of a root is
   the number of singular values of I − U_s(k*) at or below ``nullity_tol``,
   times 2 if the sector has a partner.

Two levels closer than ~2 grid steps can merge into one candidate: choose
``N_scan`` so that the mean level spacing (π / (n_eff L_tot) in k) spans many
grid points. :meth:`compute_spectrum` warns when the number of roots departs
from the Weyl estimate of its sector.
"""

import os
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from ..core.persistence import use_cache, write_cache_meta
from ..core.scattering import NULLITY_TOL


class SpectrumMixin:

    # =========================================================================
    # Spectrum computation
    # =========================================================================

    def compute_spectrum(self, N_scan=1_000_000, tol=1e-16,
                         show_progress=False, save_spectrum=True,
                         overwrite=False, maxiter=500, nullity_tol=NULLITY_TOL,
                         eps_accept=1e-4):
        """
        Closed-graph spectrum on the real k-axis via σ_min(I − U_s(k)), sector
        by sector.

        Only one sector of each pair of reciprocal partners is scanned (see
        :mod:`quantum_graphs.core.sectors`): its levels are those of the
        partner, so their multiplicity is known by construction.

        Parameters
        ----------
        N_scan : int
            Number of coarse grid points between k_min and k_max.
        tol : float
            x-accuracy target for bounded refinement (xatol).
        show_progress : bool
            Print scan progress and summary.
        save_spectrum : bool
            Cache result as .npz (with a ``.meta.json`` parameter record).
        overwrite : bool
            Recompute even if a valid cached file exists.
        maxiter : int
            Max iterations per refinement call.
        nullity_tol : float
            Singular values of I − U_s(k*) at or below it count in the
            multiplicity of a level.
        eps_accept : float
            Maximum σ_min(I − U_s(k*)) for a refined root to be accepted.

        Returns
        -------
        spectrum : np.ndarray
            Sorted real eigen-wavenumbers in [k_min, k_max], one per distinct level.
        degeneracies : np.ndarray (int)
            Multiplicity of each level (2 for the levels of partner sectors).

        Also sets ``self.spectrum``, ``self.degeneracies`` and
        ``self.level_sectors`` (representative sector of each level).
        """
        lam_min, lam_max = self.wavelength_range

        base = (f"spectrum_sigmaMin_lamMin_{lam_min:.3f}_lamMax_{lam_max:.3f}"
                f"_N{N_scan}_tol{tol:.0e}_maxiter{maxiter:.0f}"
                f"_nulltol{nullity_tol:.0e}")
        if eps_accept != 1e-4:
            base += f"_eps{eps_accept:.0e}"
        filepath_npz = os.path.join(self.save_path, f"{base}.npz")
        params = dict(N_scan=N_scan, tol=tol, maxiter=maxiter, nullity_tol=nullity_tol,
                      eps_accept=eps_accept, wavelength_range=list(self.wavelength_range),
                      method="sectors")

        if save_spectrum and use_cache(filepath_npz, self.fingerprint, params, overwrite,
                                       label="spectrum"):
            self._log(f"✓ Spectrum loaded from file: {filepath_npz}", verbose=show_progress)
            data = np.load(filepath_npz)
            spectrum = np.array(data["spectrum"], dtype=float)
            if "level_sectors" in data.files:
                level_sectors = np.array(data["level_sectors"], dtype=int)
                degeneracies = np.array(data["degeneracies"], dtype=int)
            else:                               # written before the sector analysis
                level_sectors, degeneracies = self._classify_levels(spectrum, nullity_tol)
            self._set_levels(spectrum, degeneracies, level_sectors)
            return spectrum, degeneracies

        roots, nullities, sectors = [], [], []
        for sec in self.sectors.independent:
            model = self.model.sector(sec.directed)
            self._log(f"Sector {sec.index} ({sec.directed.size} directed bonds, "
                      f"×{self.sectors.multiplicity(sec.index)}): scanning σ_min(I − U_s(k)) ...",
                      verbose=show_progress)
            k_s = self._sector_roots(model, N_scan, tol, maxiter, eps_accept, show_progress)
            n_s = np.array([model.nullity(k, nullity_tol) for k in k_s], dtype=int)
            self._check_weyl_count(sec, k_s, n_s)
            roots.append(k_s)
            nullities.append(n_s * self.sectors.multiplicity(sec.index))
            sectors.append(np.full(k_s.size, sec.index))

        order = np.argsort(np.concatenate(roots), kind="stable")
        spectrum = np.concatenate(roots)[order]
        degeneracies = np.concatenate(nullities)[order].astype(int)
        level_sectors = np.concatenate(sectors)[order].astype(int)

        if save_spectrum:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez(filepath_npz, spectrum=spectrum, degeneracies=degeneracies,
                     level_sectors=level_sectors)
            write_cache_meta(filepath_npz, self.fingerprint, params)
            self._log(f"✓ Spectrum saved to {filepath_npz}", verbose=show_progress)

        self._set_levels(spectrum, degeneracies, level_sectors)
        return spectrum, degeneracies

    def _sector_roots(self, model, N_scan, tol, maxiter, eps_accept, show_progress):
        """Roots of σ_min(I − U_s(k)) in the window: grid scan, then Brent
        refinement of every strict local minimum."""
        k_min, k_max = self._k_range()

        # 1. coarse scan (batched)
        k_vals = np.linspace(k_min, k_max, N_scan)
        g_vals = np.empty(N_scan)
        chunk = 20_000
        for s in range(0, N_scan, chunk):
            g_vals[s:s + chunk] = model.sigma_min_many(k_vals[s:s + chunk])
            self._log(f"Scanning σ_min: {int(100 * min(s + chunk, N_scan) / N_scan):3d}% done",
                      end="\r", verbose=show_progress)
        self._log("Scanning σ_min: 100% ✓\nFinding candidate wells...", verbose=show_progress)

        if not np.any(np.isfinite(g_vals)):
            raise RuntimeError("σ_min is not finite anywhere on the scan grid.")
        self._log(f"   min σ_min on grid: {np.min(g_vals[np.isfinite(g_vals)]):.3e}\n"
                  f"   accept ≤ {eps_accept:.3e}", verbose=show_progress)

        # 2. strict local minima, refined on the two neighbouring intervals
        inner = g_vals[1:-1]
        cand_idx = np.nonzero((inner < g_vals[:-2]) & (inner < g_vals[2:]))[0] + 1
        roots = []
        for i in cand_idx:
            res = minimize_scalar(model.sigma_min, bounds=(k_vals[i - 1], k_vals[i + 1]),
                                  method="bounded", options={"xatol": tol, "maxiter": maxiter})
            if res.success and (k_min <= res.x <= k_max) and (res.fun <= eps_accept):
                roots.append(float(res.x))
        self._log(f"Refined {len(cand_idx)} candidates → kept {len(roots)} roots.",
                  verbose=show_progress)
        return np.array(sorted(roots), dtype=float)

    def _check_weyl_count(self, sector, roots, nullities, rel_tol=0.05):
        """Warn when the number of levels of a sector departs from its Weyl estimate."""
        if roots.size < 20:
            return
        expected = self.weyl_count(sector=sector.index)
        found = float(np.sum(nullities))
        if expected > 0 and abs(found - expected) > max(rel_tol * expected, 3.0):
            warnings.warn(
                f"compute_spectrum found {found:.0f} levels in sector {sector.index} where the "
                f"Weyl law predicts {expected:.1f}: levels may be missing (increase N_scan) or "
                "the index / group index may be inconsistent.", RuntimeWarning, stacklevel=3)

    # =========================================================================
    # Wavefunction
    # =========================================================================

    def compute_wavefunction(self, k, tol=NULLITY_TOL):
        """
        Basis of ker(I − U(k)) at wavenumber k (singular values ≤ ``tol``).

        Component b of each vector is the amplitude of the wave on directed
        bond b = (u → v) when it arrives at v, so that a = U(k) a. Vectors are
        normalised (Σ|a_b|² = 1), with an arbitrary global phase; for a
        degenerate level they form an arbitrary orthonormal basis of the
        eigenspace. For level i of the computed spectrum,
        ``null_vectors(spectrum[i], degeneracies[i])`` uses its multiplicity.

        Returns
        -------
        null_space : ndarray, shape (n_modes, 2 n_bonds)
            Each row is one wavefunction coefficient vector.
        """
        rows = self.null_vectors(k, tol=tol)
        if rows.size == 0:
            self._log(f"⚠ Warning: No null space found at k = {k:.6f} (tol={tol}).")
        else:
            self._log(f"✓ Found {rows.shape[0]} wavefunction(s) at k = {k:.6f}.")
        return rows
