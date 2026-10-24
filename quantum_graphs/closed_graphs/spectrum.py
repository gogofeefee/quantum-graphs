"""
SpectrumMixin — U(k), secular function, eigenwavenumbers and eigenvectors.

Algorithm of :meth:`compute_spectrum`
-------------------------------------
1. g(k) = σ_min(I − U(k)) on ``N_scan`` points uniformly spaced in k
   (batched SVDs);
2. every strict local minimum of g is refined by bounded Brent minimisation
   on the two neighbouring grid intervals (``xatol=tol``);
3. roots with g(k*) ≤ ``eps_accept`` are kept; the degeneracy of each root is
   the number of singular values of I − U(k*) below ``nullity_tol``.

Two levels closer than ~2 grid steps can merge into one candidate: choose
``N_scan`` so that the mean level spacing (π / (n_eff L_tot) in k) spans many
grid points. :meth:`compute_spectrum` warns when the number of roots departs
from the Weyl estimate.
"""

import os
import warnings

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize_scalar

from ..core.persistence import use_cache, write_cache_meta


class SpectrumMixin:

    # =========================================================================
    # Matrices
    # =========================================================================

    def _build_propagation_matrix(self, k):
        """Diagonal propagation matrix D(k), D_bb = exp(i k n_eff L_b)."""
        return self.model.D(k)

    def _build_sigma_alpha_alpha(self, k):
        """Internal scattering block Σ_αα(k)."""
        return np.array(self.model.block("aa", k))

    def U(self, k):
        """Quantum evolution map U(k) = D(k) Σ_αα(k)."""
        return self.model.U(k)

    def sigma_min(self, k):
        """Smallest singular value of I − U(k); scalar or array of k."""
        if np.ndim(k) == 0:
            return self.model.sigma_min(k)
        return self.model.sigma_min_many(np.asarray(k))

    # =========================================================================
    # Secular determinant and nullity
    # =========================================================================

    def secular_det(self, k):
        """Secular determinant f(k) = det(I − D(k) Σ_αα(k))."""
        return np.linalg.det(np.eye(self.n_directed_bonds) - self.model.U(k))

    def _nullity_at_k(self, k, tol_abs=1e-4):
        """Dimension of ker(I − U(k)): number of singular values ≤ tol_abs."""
        s = np.linalg.svd(np.eye(self.n_directed_bonds) - self.model.U(k), compute_uv=False)
        return int(np.sum(s <= tol_abs))

    def weyl_count(self, k_min=None, k_max=None):
        """Weyl estimate of the number of levels (with multiplicity) in [k_min, k_max]."""
        lo, hi = self._k_range()
        k_min = lo if k_min is None else k_min
        k_max = hi if k_max is None else k_max
        return float(self.N_weyl_scaled(k_max) - self.N_weyl_scaled(k_min))

    # =========================================================================
    # Spectrum computation
    # =========================================================================

    def compute_spectrum(self, N_scan=1_000_000, tol=1e-16,
                         show_progress=False, save_spectrum=True,
                         overwrite=False, maxiter=500, nullity_tol=1e-4,
                         eps_accept=1e-4):
        """
        Closed-graph spectrum on the real k-axis via σ_min(I − U(k)).

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
            SVD threshold for degeneracy detection.
        eps_accept : float
            Maximum σ_min(I − U(k*)) for a refined root to be accepted.

        Returns
        -------
        spectrum : np.ndarray
            Sorted real eigen-wavenumbers in [k_min, k_max].
        degeneracies : np.ndarray (int)
            Degeneracy of each eigenvalue.
        """
        lam_min, lam_max = self.wavelength_range
        k_min, k_max = self._k_range()

        base = (f"spectrum_sigmaMin_lamMin_{lam_min:.3f}_lamMax_{lam_max:.3f}"
                f"_N{N_scan}_tol{tol:.0e}_maxiter{maxiter:.0f}"
                f"_nulltol{nullity_tol:.0e}")
        if eps_accept != 1e-4:
            base += f"_eps{eps_accept:.0e}"
        filepath_npz = os.path.join(self.save_path, f"{base}.npz")
        params = dict(N_scan=N_scan, tol=tol, maxiter=maxiter, nullity_tol=nullity_tol,
                      eps_accept=eps_accept, wavelength_range=list(self.wavelength_range))

        if save_spectrum and use_cache(filepath_npz, self.fingerprint, params, overwrite,
                                       label="spectrum"):
            self._log(f"✓ Spectrum loaded from file: {filepath_npz}", verbose=show_progress)
            data = np.load(filepath_npz)
            spectrum = np.array(data["spectrum"], dtype=float)
            degeneracies = np.array(data["degeneracies"], dtype=int)
            self._set_spectrum(spectrum, degeneracies)
            return spectrum, degeneracies

        self._log("Starting closed-graph spectrum via σ_min(I−U(k)) ...", verbose=show_progress)

        # 1. coarse scan (batched)
        k_vals = np.linspace(k_min, k_max, N_scan)
        g_vals = np.empty(N_scan)
        chunk = 20_000
        for s in range(0, N_scan, chunk):
            g_vals[s:s + chunk] = self.model.sigma_min_many(k_vals[s:s + chunk])
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
            res = minimize_scalar(self.model.sigma_min, bounds=(k_vals[i - 1], k_vals[i + 1]),
                                  method="bounded", options={"xatol": tol, "maxiter": maxiter})
            if res.success and (k_min <= res.x <= k_max) and (res.fun <= eps_accept):
                roots.append(float(res.x))
        self._log(f"Refined {len(cand_idx)} candidates → kept {len(roots)} roots.",
                  verbose=show_progress)
        spectrum = np.array(sorted(roots), dtype=float)

        # 3. degeneracies
        degeneracies = np.array([self._nullity_at_k(k, tol_abs=nullity_tol) for k in spectrum],
                                dtype=int)
        self._check_weyl_count(spectrum, degeneracies)

        if save_spectrum:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez(filepath_npz, spectrum=spectrum, degeneracies=degeneracies)
            write_cache_meta(filepath_npz, self.fingerprint, params)
            self._log(f"✓ Spectrum saved to {filepath_npz}", verbose=show_progress)

        self._set_spectrum(spectrum, degeneracies)
        return spectrum, degeneracies

    def _set_spectrum(self, spectrum, degeneracies):
        self.spectrum = spectrum
        self.degeneracies = degeneracies
        self.degenerate_spectrum = np.repeat(spectrum, degeneracies)
        zero = spectrum[degeneracies == 0]
        if zero.size:
            warnings.warn(
                f"{zero.size} level(s) have nullity 0: consider loosening nullity_tol or "
                f"reducing tol. Examples: k = {', '.join(f'{k:.8g}' for k in zero[:5])}.",
                RuntimeWarning, stacklevel=3)

    def _check_weyl_count(self, spectrum, degeneracies, rel_tol=0.05):
        """Warn when the number of levels departs from the Weyl estimate."""
        if spectrum.size < 20:
            return
        expected = self.weyl_count()
        found = float(np.sum(degeneracies))
        if expected > 0 and abs(found - expected) > max(rel_tol * expected, 3.0):
            warnings.warn(
                f"compute_spectrum found {found:.0f} levels (with multiplicity) where the Weyl "
                f"law predicts {expected:.1f}: levels may be missing (increase N_scan) or the "
                "index / group index may be inconsistent.", RuntimeWarning, stacklevel=3)

    # =========================================================================
    # Wavefunction
    # =========================================================================

    def compute_wavefunction(self, k, tol=1e-4):
        """
        Basis of ker(I − U(k)) at wavenumber k.

        Component b of each vector is the amplitude of the wave on directed
        bond b = (u → v) when it arrives at v, so that a = U(k) a. Vectors are
        normalised (Σ|a_b|² = 1), with an arbitrary global phase; for a
        degenerate level they form an arbitrary orthonormal basis of the
        eigenspace.

        Returns
        -------
        null_space : ndarray, shape (n_modes, n_directed_bonds)
            Each row is one wavefunction coefficient vector.
        """
        A = np.eye(self.n_directed_bonds, dtype=complex) - self.model.U(k)
        _, S, Vh = svd(A)
        rows = Vh[S < tol].conj()          # rows of Vh are v^H
        if rows.size == 0:
            self._log(f"⚠ Warning: No null space found at k = {k:.6f} (tol={tol}).")
        else:
            self._log(f"✓ Found {rows.shape[0]} wavefunction(s) at k = {k:.6f}.")
        return rows
