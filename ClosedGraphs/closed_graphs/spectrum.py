#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectrumMixin — propagation matrix, scattering block, secular determinant,
spectrum computation, and wavefunction reconstruction.
"""

import os
import warnings
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize_scalar


class SpectrumMixin:

    # =========================================================================
    # Core matrices
    # =========================================================================

    def _build_propagation_matrix(self, k):
        """
        Diagonal propagation matrix D(k).

        D[idx, idx] = exp(i k n_eff L)  for each directed bond.
        """
        n_eff = self.neff(np.real(k)) if callable(self.neff) else self.neff

        D = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)
        for dbond, idx in self.directed_bond_index.items():
            L = self.directed_lengths[dbond]
            D[idx, idx] = np.exp(1j * k * n_eff * L)
        return D

    def _build_sigma_alpha_alpha(self, k):
        """
        Internal scattering block Σ_{αα}(k).

        Σ_{αα}[out, in] = σ_v[port_out, port_in] for each directed bond pair
        meeting at vertex v.
        """
        Sigma_aa = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)

        for dbond_in in self.directed_bonds:
            src, dst  = dbond_in
            v         = dst[0]
            port_in   = dst[1]

            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(k)

            idx_in = self.directed_bond_index[dbond_in]

            for port_out in range(self.valencies[v]):
                neighbor, n_port = self.virtual_port_map[v][port_out]
                dbond_out = ((v, port_out), (neighbor, n_port))
                idx_out   = self.directed_bond_index[dbond_out]
                Sigma_aa[idx_out, idx_in] = sigma_v[port_out, port_in]

        return Sigma_aa

    # =========================================================================
    # Secular determinant and nullity
    # =========================================================================

    def secular_det(self, k):
        """
        Secular determinant  f(k) = det(I − D(k) Σ_{αα}(k)).

        Parameters
        ----------
        k : float or complex

        Returns
        -------
        complex
        """
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        D        = self._build_propagation_matrix(k)
        return np.linalg.det(np.eye(self.n_directed_bonds) - D @ Sigma_aa)

    def _nullity_at_k(self, k, tol_abs=1e-4):
        """
        Dimension of ker(I − U(k)) at real wavenumber k.

        Parameters
        ----------
        k : float
        tol_abs : float
            Singular values ≤ tol_abs are treated as zero.

        Returns
        -------
        int
        """
        D_k      = self._build_propagation_matrix(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        U_k      = D_k @ Sigma_aa
        A        = np.eye(self.n_directed_bonds, dtype=complex) - U_k
        s        = np.linalg.svd(A, compute_uv=False)
        return int(np.sum(s <= tol_abs))

    # =========================================================================
    # Spectrum computation
    # =========================================================================

    def compute_spectrum(self, N_scan=1_000_000, tol=1e-16,
                         show_progress=False, save_spectrum=True,
                         overwrite=False, maxiter=500, nullity_tol=1e-4):
        """
        Closed-graph spectrum on the real k-axis via σ_min(I − U(k)).

        Scans k-space for local minima of g(k) = σ_min(I − U(k)), refines each
        candidate with bounded minimization, and determines degeneracies via SVD.

        Parameters
        ----------
        N_scan : int
            Number of coarse grid points between k_min and k_max.
        tol : float
            x-accuracy target for bounded refinement (xatol).
        show_progress : bool
            Print scan progress and summary.
        save_spectrum : bool
            Cache result as .npz.
        overwrite : bool
            Recompute even if a cached file exists.
        maxiter : int
            Max iterations per refinement call.
        nullity_tol : float
            SVD threshold for degeneracy detection.

        Returns
        -------
        spectrum : np.ndarray
            Sorted real eigen-wavenumbers in [k_min, k_max].
        degeneracies : np.ndarray (int)
            Degeneracy of each eigenvalue.
        """
        lam_min, lam_max = self.wavelength_range
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min

        base = (f"spectrum_sigmaMin_lamMin_{lam_min:.3f}_lamMax_{lam_max:.3f}"
                f"_N{N_scan}_tol{tol:.0e}_maxiter{maxiter:.0f}"
                f"_nulltol{nullity_tol:.0e}")
        filepath_npz = os.path.join(self.save_path, f"{base}.npz")

        if save_spectrum and (not overwrite) and os.path.exists(filepath_npz):
            if show_progress:
                print(f"✓ Spectrum loaded from file: {filepath_npz}")
            data = np.load(filepath_npz)
            spectrum     = np.array(data["spectrum"],     dtype=float)
            degeneracies = np.array(data["degeneracies"], dtype=int)
            self.spectrum          = spectrum
            self.degeneracies      = degeneracies
            self.degenerate_spectrum = np.repeat(spectrum, degeneracies)

            zero_idx = np.nonzero(degeneracies == 0)[0]
            if zero_idx.size > 0:
                ks_bad = ", ".join(f"{spectrum[i]:.8g}" for i in zero_idx[:5])
                tail   = "" if zero_idx.size <= 5 else f" (and {zero_idx.size-5} more)"
                warnings.warn(
                    "Some loaded modes have nullity 0. Consider loosening nullity_tol "
                    f"or refining k further. Examples: k = {ks_bad}{tail}.",
                    RuntimeWarning,
                )
            return spectrum, degeneracies

        if show_progress:
            print("Starting closed-graph spectrum via σ_min(I−U(k)) ...")

        def _sigma_min(k):
            try:
                Sigma = self._build_sigma_alpha_alpha(k)
                D     = self._build_propagation_matrix(k)
                U     = D @ Sigma
                svals = np.linalg.svd(np.eye(U.shape[0]) - U, compute_uv=False)
                smin  = float(svals[-1])
                return smin if np.isfinite(smin) else np.inf
            except Exception:
                return np.inf

        # Coarse scan
        k_vals = np.linspace(k_min, k_max, N_scan)
        g_vals = np.empty(N_scan, dtype=float)

        last_pct = -1
        for i, k in enumerate(k_vals):
            g_vals[i] = _sigma_min(k)
            if show_progress:
                pct = int(i * 100 / max(1, N_scan - 1))
                if pct % 5 == 0 and pct != last_pct:
                    print(f"Scanning σ_min: {pct:3d}% done", end="\r")
                    last_pct = pct
        if show_progress:
            print("Scanning σ_min: 100% ✓\nFinding candidate wells...")

        finite_mask  = np.isfinite(g_vals)
        finite_vals  = g_vals[finite_mask]
        if finite_vals.size == 0:
            if show_progress:
                print("No finite values; returning empty spectrum.")
            self.spectrum = self.degeneracies = np.array([])
            self.degenerate_spectrum = np.array([])
            return self.spectrum, self.degeneracies

        eps_accept = 1e-4
        if show_progress:
            print(f"   min σ_min on grid: {np.min(finite_vals):.3e}")
            print(f"   accept ≤ {eps_accept:.3e}")

        cand_idx = [i for i in range(1, N_scan - 1)
                    if g_vals[i] < g_vals[i - 1] and g_vals[i] < g_vals[i + 1]]

        # Refine
        roots = []
        for i in cand_idx:
            a, b = k_vals[i - 1], k_vals[i + 1]
            res  = minimize_scalar(_sigma_min, bounds=(a, b), method="bounded",
                                   options={"xatol": tol, "maxiter": maxiter})
            k_star, g_star = res.x, res.fun
            if res.success and (k_min <= k_star <= k_max) and (g_star <= eps_accept):
                roots.append(k_star)

        if show_progress:
            print(f"Refined {len(cand_idx)} candidates → kept {len(roots)} roots.")

        spectrum = np.array(sorted(roots), dtype=float)

        # Degeneracies
        if spectrum.size == 0:
            degeneracies = np.array([], dtype=int)
            deg_spectrum = np.array([], dtype=float)
        else:
            degeneracies = np.empty(spectrum.size, dtype=int)
            zero_list    = []
            for idx, k in enumerate(spectrum):
                d = int(self._nullity_at_k(float(k), tol_abs=nullity_tol))
                degeneracies[idx] = d
                if d == 0:
                    zero_list.append((idx, k))

            if zero_list:
                examples = ", ".join(f"{k:.8g}" for _, k in zero_list[:5])
                tail     = "" if len(zero_list) <= 5 else f" (and {len(zero_list)-5} more)"
                warnings.warn(
                    "Some refined roots have nullity 0. Consider loosening nullity_tol "
                    f"or reducing tol. Examples: k = {examples}{tail}.",
                    RuntimeWarning,
                )

            deg_spectrum = np.repeat(spectrum, degeneracies)

        if save_spectrum:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez(filepath_npz, spectrum=spectrum, degeneracies=degeneracies)
            if show_progress:
                print(f"✓ Spectrum saved to {filepath_npz}")

        self.spectrum            = spectrum
        self.degeneracies        = degeneracies
        self.degenerate_spectrum = deg_spectrum
        return spectrum, degeneracies

    # =========================================================================
    # Wavefunction
    # =========================================================================

    def compute_wavefunction(self, k, tol=1e-4):
        """
        Basis of ker(I − U(k)) at wavenumber k.

        Parameters
        ----------
        k : float
        tol : float
            SVD threshold for near-zero singular values.

        Returns
        -------
        null_space : ndarray, shape (n_modes, n_directed_bonds)
            Each row is one wavefunction coefficient vector.
        """
        D_k      = self._build_propagation_matrix(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        U_k      = D_k @ Sigma_aa
        A        = np.eye(self.n_directed_bonds, dtype=complex) - U_k

        _, S, Vh = svd(A)
        null_mask  = S < tol
        null_space = Vh[null_mask].conj().T   # columns = basis vectors (rows of Vh are v^H)

        if null_space.size == 0:
            print(f"⚠ Warning: No null space found at k = {k:.6f} (tol={tol}).")
        else:
            print(f"✓ Found {null_space.shape[1]} wavefunction(s) at k = {k:.6f}.")

        return null_space.T   # rows = eigenvectors
