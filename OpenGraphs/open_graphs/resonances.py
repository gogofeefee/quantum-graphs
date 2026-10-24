#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResonancesMixin — σ_min grid, candidate detection, refinement, find_resonances.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from scipy.optimize import minimize


class ResonancesMixin:

    # ── Grid path helpers ─────────────────────────────────────────────────────

    def _sigma_grid_path(self, N_re: int, N_im: int,
                         im_min: float, im_max: float) -> str:
        lam_min, lam_max = self.wavelength_range
        fname = (f"sigma_grid"
                 f"_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_Nre_{N_re}_Nim_{N_im}"
                 f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}.npy")
        return os.path.join(self.save_path, fname)

    def _resonances_path(self, N_re: int, N_im: int,
                         im_min: float, im_max: float,
                         tol_step: float, maxiter: int) -> str:
        lam_min, lam_max = self.wavelength_range
        fname = (f"resonances_raw"
                 f"_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_Nre_{N_re}_Nim_{N_im}"
                 f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}"
                 f"_tolstep_{tol_step:.1e}_maxiter_{int(maxiter)}.npz")
        return os.path.join(self.save_path, fname)

    def _spectrum_path(self, N_re: int, N_im: int,
                       im_min: float, im_max: float,
                       tol_step: float, maxiter: int,
                       eps_accept: float, nullity_tol: float) -> str:
        lam_min, lam_max = self.wavelength_range
        fname = (f"spectrum"
                 f"_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_Nre_{N_re}_Nim_{N_im}"
                 f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}"
                 f"_tolstep_{tol_step:.1e}_maxiter_{int(maxiter)}"
                 f"_eps_{eps_accept:.1e}_nulltol_{nullity_tol:.1e}.npz")
        return os.path.join(self.save_path, fname)

    # ── Grid open / fill ──────────────────────────────────────────────────────

    def _open_sigma_grid(self, N_re: int, N_im: int,
                         im_min: float, im_max: float,
                         grid_dtype: str = "float32",
                         overwrite: bool = False,
                         verbose: bool = True):
        lam_min, lam_max = self.wavelength_range
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
        re_vals = np.linspace(k_min, k_max, N_re, dtype=np.float64)
        im_vals = np.linspace(im_max, im_min, N_im, dtype=np.float64)
        grid_path = self._sigma_grid_path(N_re, N_im, im_min, im_max)
        os.makedirs(self.save_path, exist_ok=True)
        if overwrite and os.path.exists(grid_path):
            os.remove(grid_path)
            if verbose:
                print(f"[grid] Removed existing grid: {grid_path}")
        if not os.path.exists(grid_path):
            if verbose:
                print(f"[grid] Creating  ({N_im}×{N_re}, {grid_dtype}): {grid_path}")
            mem = np.lib.format.open_memmap(grid_path, mode="w+",
                                            dtype=grid_dtype, shape=(N_im, N_re))
            mem[:] = np.nan
            mem.flush()
        else:
            if verbose:
                print(f"[grid] Reusing existing grid: {grid_path}")
            mem = np.lib.format.open_memmap(grid_path, mode="r+")
            if mem.shape != (N_im, N_re):
                raise ValueError(f"Existing grid shape {mem.shape} != requested ({N_im}, {N_re}).")
        return mem, re_vals, im_vals, grid_path

    def _fill_sigma_grid(self, mem, re_vals, im_vals, grid_path,
                         verbose: bool = True, flush_every: int = 1):
        N_im, N_re = mem.shape
        n_done = int(np.sum(~np.isnan(mem).any(axis=1)))
        if verbose:
            print(f"[grid] Filling σ_min grid ({n_done}/{N_im} rows already done) ...")
        flush_counter = 0
        for i in range(N_im):
            row = mem[i]
            if not np.isnan(row).any():
                continue
            im       = im_vals[i]
            nan_cols = np.where(np.isnan(row))[0]
            for j in nan_cols:
                row[j] = self._sigma_min(re_vals[j] + 1j * im)
            flush_counter += 1
            if flush_counter >= flush_every:
                mem.flush()
                flush_counter = 0
            if verbose:
                done = int(np.sum(~np.isnan(mem).any(axis=1)))
                print(f"\r[grid]  {done}/{N_im} rows ({100*done/N_im:.1f}%)", end="", flush=True)
        mem.flush()
        if verbose:
            print(f"\r[grid] Done – grid complete.              ")

    # ── Candidate detection ───────────────────────────────────────────────────

    def _find_candidates(self, mem, re_vals, im_vals,
                         peak_prominence: float = 0.0,
                         neighbor_rows: int = 1,
                         verbose: bool = True):
        N_im, N_re = mem.shape
        g      = np.array(mem, dtype=np.float64)
        g_safe = np.where(np.isfinite(g), g, np.inf)
        row_candidates = []
        for i in range(N_im):
            row     = g_safe[i]
            neg_row = -row
            peaks, _ = find_peaks(neg_row, prominence=peak_prominence)
            for j in peaks:
                row_candidates.append((i, j))
        if verbose:
            print(f"[candidates] {len(row_candidates)} per-row minima found.")
        promoted = []
        for i, j in row_candidates:
            val = g_safe[i, j]
            ok  = True
            for di in range(1, neighbor_rows + 1):
                for sign in (-1, +1):
                    ii = i + sign * di
                    if not (0 <= ii < N_im):
                        continue
                    j_lo = max(j - 1, 0)
                    j_hi = min(j + 1, N_re - 1)
                    if np.any(val > g_safe[ii, j_lo:j_hi + 1]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                promoted.append((i, j))
        if verbose:
            print(f"[candidates] {len(promoted)} 2D candidates promoted.")
        if not promoted:
            return np.array([], dtype=np.complex128)
        k0_list = np.array(
            [re_vals[j] + 1j * im_vals[i] for i, j in promoted],
            dtype=np.complex128,
        )
        return k0_list

    # ── Refinement ────────────────────────────────────────────────────────────

    def _refine_one(self, k0: complex, re_vals, im_vals,
                    box_re: int = 3, box_im: int = 3,
                    tol_step: float = 1e-12, maxiter: int = 500) -> tuple:
        x0, y0 = float(np.real(k0)), float(np.imag(k0))
        dx = float(re_vals[1] - re_vals[0]) if re_vals.size >= 2 else 1e-12
        dy = float(abs(im_vals[1] - im_vals[0])) if im_vals.size >= 2 else 1e-12
        x_lo = max(x0 - box_re * dx, float(re_vals.min()))
        x_hi = min(x0 + box_re * dx, float(re_vals.max()))
        y_lo = max(y0 - box_im * dy, float(im_vals.min()))
        y_hi = min(y0 + box_im * dy, float(im_vals.max()))
        for lo, hi, center, step in [
            (x_lo, x_hi, x0, dx), (y_lo, y_hi, y0, dy)
        ]:
            if hi <= lo:
                eps = max(step * 1e-4, 1e-15)
                x_lo, x_hi = (center - eps, center + eps) if lo == x_lo else (x_lo, x_hi)
                y_lo, y_hi = (center - eps, center + eps) if lo == y_lo else (y_lo, y_hi)

        def obj(xy):
            return self._sigma_min(float(xy[0]) + 1j * float(xy[1]))

        res = minimize(
            obj, x0=np.array([x0, y0]), method="L-BFGS-B",
            bounds=[(x_lo, x_hi), (y_lo, y_hi)],
            options={"ftol": 1e-15, "gtol": tol_step,
                     "maxiter": maxiter, "maxfun": maxiter * 10},
        )
        k_star = complex(float(res.x[0]), float(res.x[1]))
        g_star = float(res.fun)
        res2 = minimize(
            obj, x0=res.x, method="Nelder-Mead",
            options={"xatol": tol_step * 0.1, "fatol": 1e-15,
                     "maxiter": maxiter, "adaptive": True},
        )
        if res2.fun < g_star:
            k_star = complex(float(res2.x[0]), float(res2.x[1]))
            g_star = float(res2.fun)
        success = g_star < float(np.inf) and np.isfinite(g_star)
        return k_star, g_star, success

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, k_arr: np.ndarray, g_arr: np.ndarray,
                     min_spacing_re: float, min_spacing_im: float) -> tuple:
        if k_arr.size == 0:
            return k_arr, g_arr
        order = np.argsort(g_arr)
        k_arr = k_arr[order]
        g_arr = g_arr[order]
        kept  = np.ones(len(k_arr), dtype=bool)
        for i in range(len(k_arr)):
            if not kept[i]:
                continue
            for j in range(i + 1, len(k_arr)):
                if not kept[j]:
                    continue
                if (abs(np.real(k_arr[j]) - np.real(k_arr[i])) < min_spacing_re and
                        abs(np.imag(k_arr[j]) - np.imag(k_arr[i])) < min_spacing_im):
                    kept[j] = False
        return k_arr[kept], g_arr[kept]

    # ── Degeneracy ────────────────────────────────────────────────────────────

    def _nullity(self, k: complex, atol: float) -> int:
        """Nullity of (I − U(k))."""
        U = self._U(k)
        A = np.eye(U.shape[0], dtype=np.complex128) - U
        s = np.linalg.svd(A, compute_uv=False)
        return int(np.sum(s <= atol))


    # ── Public method ─────────────────────────────────────────────────────────

    def find_resonances(self,
                        N_re: int         = 500_000,
                        N_im: int         = 100,
                        im_min: float     = -1e-4,
                        im_max: float     =  0.0,
                        peak_prominence: float = 0.0,
                        neighbor_rows: int     = 1,
                        box_re: int       = 3,
                        box_im: int       = 3,
                        tol_step: float   = 1e-13,
                        maxiter: int      = 500,
                        eps_accept: float  = 1e-4,
                        nullity_tol: float = 1e-4,
                        grid_dtype: str        = "float32",
                        overwrite_grid: bool     = False,
                        overwrite_refined: bool  = False,
                        overwrite_spectrum: bool = False,
                        plot: bool    = True,
                        verbose: bool = True):
        """
        Find the complex resonances of the open quantum graph via the σ_min method.

        Returns
        -------
        spectrum      : np.ndarray, complex128, shape (N,)
        degeneracies  : np.ndarray, int, shape (N,)
        """
        os.makedirs(self.save_path, exist_ok=True)

        lam_min, lam_max = self.wavelength_range
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
        re_vals = np.linspace(k_min, k_max, N_re, dtype=np.float64)
        im_vals = np.linspace(im_max, im_min, N_im, dtype=np.float64)

        raw_path  = self._resonances_path(N_re, N_im, im_min, im_max, tol_step, maxiter)
        spec_path = self._spectrum_path(N_re, N_im, im_min, im_max, tol_step, maxiter,
                                        eps_accept, nullity_tol)

        # ── 0. Early-load from final spectrum ─────────────────────────────────
        if (not overwrite_spectrum) and (not overwrite_refined) and (not overwrite_grid) \
                and os.path.exists(spec_path):
            if verbose:
                print(f"[spectrum] Loading final spectrum from: {spec_path}")
            npz = np.load(spec_path)
            self.spectrum            = npz["spectrum"].astype(np.complex128)
            self.degeneracies        = npz["degeneracies"].astype(int)
            self.degenerate_spectrum = npz["degenerate_spectrum"].astype(np.complex128)
            if verbose:
                print(f"[spectrum] {self.spectrum.size} resonances loaded, "
                      f"total multiplicity = {self.degenerate_spectrum.size}.")
            if plot:
                grid_path = self._sigma_grid_path(N_re, N_im, im_min, im_max)
                has_grid  = os.path.exists(grid_path)
                if verbose and not has_grid:
                    print("[plot] σ_min grid not found → plotting resonances only (no heatmap).")
                self.plot_resonances_sigma(
                    N_re=N_re, N_im=N_im,
                    im_min=im_min, im_max=im_max,
                    plot_heatmap=has_grid,
                    overwrite_grid=False,
                    grid_dtype=grid_dtype,
                    save=True, verbose=verbose,
                )
            return self.spectrum, self.degeneracies

        # ── 1. σ_min grid ─────────────────────────────────────────────────────
        mem, re_vals, im_vals, grid_path = self._open_sigma_grid(
            N_re, N_im, im_min, im_max,
            grid_dtype=grid_dtype, overwrite=overwrite_grid, verbose=verbose,
        )
        self._fill_sigma_grid(mem, re_vals, im_vals, grid_path, verbose=verbose)

        # ── 2. Refined candidates ─────────────────────────────────────────────
        if (not overwrite_refined) and (not overwrite_grid) and os.path.exists(raw_path):
            if verbose:
                print(f"[refine] Loading raw candidates from: {raw_path}")
            npz   = np.load(raw_path)
            k_all = npz["k_refined"].astype(np.complex128)
            g_all = npz["g_refined"].astype(np.float64)
        else:
            k0_list = self._find_candidates(
                mem, re_vals, im_vals,
                peak_prominence=peak_prominence, neighbor_rows=neighbor_rows, verbose=verbose,
            )
            if k0_list.size == 0:
                if verbose:
                    print("[refine] No candidates found. Returning empty spectrum.")
                self.spectrum            = np.array([], dtype=np.complex128)
                self.degeneracies        = np.array([], dtype=int)
                self.degenerate_spectrum = np.array([], dtype=np.complex128)
                return self.spectrum, self.degeneracies

            if verbose:
                print(f"[refine] Refining {k0_list.size} candidates ...")
            k_all = np.empty(k0_list.size, dtype=np.complex128)
            g_all = np.empty(k0_list.size, dtype=np.float64)
            for idx, k0 in enumerate(k0_list):
                k_star, g_star, _ = self._refine_one(
                    k0, re_vals, im_vals,
                    box_re=box_re, box_im=box_im,
                    tol_step=tol_step, maxiter=maxiter,
                )
                k_all[idx] = k_star
                g_all[idx] = g_star
                if verbose:
                    print(f"\r[refine]  {idx+1}/{k0_list.size}  "
                          f"k*={np.real(k_star):.6f}{np.imag(k_star):+.3e}i  "
                          f"g={g_star:.2e}", end="", flush=True)
            if verbose:
                print()
            np.savez(raw_path, k_refined=k_all, g_refined=g_all,
                     re_vals=re_vals, im_vals=im_vals)
            if verbose:
                print(f"[refine] Raw candidates saved to: {raw_path}")

        # ── 3. Deduplicate ────────────────────────────────────────────────────
        dx = float(re_vals[1] - re_vals[0]) if re_vals.size >= 2 else 1e-12
        dy = float(abs(im_vals[1] - im_vals[0])) if im_vals.size >= 2 else 1e-12
        k_all, g_all = self._deduplicate(k_all, g_all, 0.5 * dx, 0.5 * dy)
        if verbose:
            print(f"[filter] {k_all.size} candidates after deduplication.")

        # ── 4. Accept by eps_accept ───────────────────────────────────────────
        mask     = g_all <= eps_accept
        spectrum = k_all[mask]
        if verbose:
            print(f"[filter] {spectrum.size} resonances accepted (σ_min ≤ {eps_accept:.1e}).")
        if spectrum.size == 0:
            if verbose:
                print("[filter] No resonances accepted. Returning empty spectrum.")
            self.spectrum            = np.array([], dtype=np.complex128)
            self.degeneracies        = np.array([], dtype=int)
            self.degenerate_spectrum = np.array([], dtype=np.complex128)
            return self.spectrum, self.degeneracies

        # ── 5. Sort ───────────────────────────────────────────────────────────
        order    = np.argsort(np.real(spectrum))
        spectrum = spectrum[order]

        # ── 6. Degeneracies ───────────────────────────────────────────────────
        if verbose:
            print("[degen] Computing degeneracies ...")
        degeneracies = np.array(
            [self._nullity(k, atol=nullity_tol) for k in spectrum], dtype=int
        )
        zero_mask = degeneracies == 0
        if zero_mask.any():
            n_zero   = int(zero_mask.sum())
            examples = ", ".join(
                f"{k.real:.6g}{k.imag:+.6g}i" for k in spectrum[zero_mask][:5]
            )
            warnings.warn(
                f"{n_zero} resonance(s) have nullity 0 "
                f"(nullity_tol={nullity_tol:.1e} may be too small). "
                f"Examples: {examples}.",
                RuntimeWarning,
            )

        # ── 7. Attributes ─────────────────────────────────────────────────────
        self.spectrum            = spectrum
        self.degeneracies        = degeneracies
        self.degenerate_spectrum = np.repeat(spectrum, degeneracies)

        # ── 8. Save ───────────────────────────────────────────────────────────
        if (not os.path.exists(spec_path)) or overwrite_spectrum:
            np.savez(spec_path, spectrum=spectrum, degeneracies=degeneracies,
                     degenerate_spectrum=self.degenerate_spectrum)
            if verbose:
                print(f"[spectrum] Final spectrum saved to: {spec_path}")
        else:
            if verbose:
                print(f"[spectrum] Spectrum file exists, not overwriting: {spec_path}")

        # ── 9. Plot ───────────────────────────────────────────────────────────
        if plot:
            self.plot_resonances_sigma(
                N_re=N_re, N_im=N_im,
                im_min=im_min, im_max=im_max,
                plot_heatmap=True,
                overwrite_grid=False,
                grid_dtype=grid_dtype,
                save=True, verbose=verbose,
            )

        if verbose:
            print(f"[done] {spectrum.size} resonances found, "
                  f"total multiplicity = {self.degenerate_spectrum.size}.")

        return spectrum, degeneracies

    # ── Resonance heatmap plot ────────────────────────────────────────────────

    def plot_resonances_sigma(self,
                              lam_min: float = None,
                              lam_max: float = None,
                              N_re: int = 1_000_000,
                              N_im: int = 25,
                              im_min: float = -1e-4,
                              im_max: float = 0.0,
                              plot_heatmap: bool = True,
                              overwrite_grid: bool = False,
                              grid_dtype: str = "float32",
                              max_cols: int = None,
                              save: bool = True,
                              file_format: str = "pdf",
                              verbose: bool = True):
        """
        Plot complex resonances in the k-plane, optionally overlaying the
        σ_min(I − U(k)) heatmap.

        Requires ``self.spectrum`` to exist (run ``find_resonances`` first).

        Parameters
        ----------
        lam_min, lam_max : float or None
            Wavelength bounds (µm). Defaults to ``self.wavelength_range``.
        N_re, N_im : int
            Grid size (must match the cached grid if ``plot_heatmap=True``).
        im_min, im_max : float
            Imaginary-k axis bounds.
        plot_heatmap : bool
            If True, load (or compute) the σ_min grid and display as heatmap.
        overwrite_grid : bool
            If True, recompute the grid even if a file already exists.
        grid_dtype : str
            Storage dtype for the σ_min grid (``"float32"`` or ``"float64"``).
        max_cols : int or None
            Cap on the number of image columns.  Above it, the grid is reduced
            by taking the **minimum** over blocks of columns.  ``None`` (default)
            plots the grid as it stands.  Worth setting whenever the grid is far
            wider than the figure: matplotlib's own resampling *averages*, which
            erases σ_min valleys narrower than a pixel — that is, the very
            resonances the plot is about — while the block minimum keeps them.
        save : bool
            Save the figure to ``self.save_path``.
        file_format : str
            Figure format (``"pdf"``, ``"png"``, …).
        verbose : bool
            Print progress messages.
        """
        from matplotlib.ticker import ScalarFormatter

        if not hasattr(self, "spectrum"):
            raise RuntimeError(
                "'self.spectrum' is required but missing. "
                "Run 'find_resonances' first."
            )

        spectrum = np.asarray(self.spectrum, dtype=np.complex128)

        lam_min0, lam_max0 = self.wavelength_range
        if lam_min is None:
            lam_min = lam_min0
        if lam_max is None:
            lam_max = lam_max0

        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_heatmap:
            mem, re_vals, im_vals, grid_path = self._open_sigma_grid(
                N_re=N_re, N_im=N_im, im_min=im_min, im_max=im_max,
                grid_dtype=grid_dtype, overwrite=overwrite_grid, verbose=verbose,
            )
            # Fill any missing rows
            self._fill_sigma_grid(mem, re_vals, im_vals, grid_path, verbose=verbose)

            # Restrict to the displayed range first, so that a zoom does not
            # carry (and resample) the whole grid.
            j0 = max(int(np.searchsorted(re_vals, k_min, side="left")) - 1, 0)
            j1 = min(int(np.searchsorted(re_vals, k_max, side="right")) + 1,
                     re_vals.size)
            img    = np.asarray(mem[:, j0:j1], dtype=np.float64)
            img    = np.where(np.isfinite(img), img, np.inf)
            re_img = np.asarray(re_vals[j0:j1], dtype=np.float64)

            # Then reduce the column count by block MINIMUM (see max_cols).
            if max_cols is not None and img.shape[1] > int(max_cols):
                block  = int(np.ceil(img.shape[1] / int(max_cols)))
                n_keep = (img.shape[1] // block) * block
                img    = img[:, :n_keep].reshape(img.shape[0], -1, block).min(axis=2)
                re_img = re_img[:n_keep].reshape(-1, block).mean(axis=1)
                if verbose:
                    print(f"[plot] {j1 - j0} grid columns → {img.shape[1]} image "
                          f"columns by block minimum (block = {block}).")

            finite = img[np.isfinite(img)]
            vmin = float(np.nanmin(finite)) if finite.size else 1e-16
            if vmin <= 0.0:
                vmin = 1e-16
            vmax = float(np.nanmax(finite)) if finite.size else 1.0

            imshow_obj = ax.imshow(
                img,
                extent=[re_img[0], re_img[-1], im_vals[-1], im_vals[0]],
                aspect="auto",
                cmap="inferno_r",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                origin="upper",
            )
            cbar = plt.colorbar(imshow_obj, ax=ax)
            cbar.set_label(r"$\sigma_{\min}(I - U(k))$", fontsize=22)
            cbar.ax.tick_params(labelsize=18)

        if spectrum.size > 0:
            cross_color = "red" if plot_heatmap else "black"
            ax.scatter(np.real(spectrum), np.imag(spectrum),
                       marker="x", color=cross_color, s=40, linewidths=1.5,
                       zorder=5)

        ax.set_xlim(k_min, k_max)
        ax.set_ylim(im_min, im_max)
        ax.set_xlabel(r"Re$(k)$  [µm$^{-1}$]", fontsize=22)
        ax.set_ylabel(r"Im$(k)$  [µm$^{-1}$]", fontsize=22)

        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(yfmt)
        ax.tick_params(axis="both", which="major", labelsize=18,
                       direction="in", top=True, right=True)

        plt.tight_layout()

        if save:
            os.makedirs(self.save_path, exist_ok=True)
            tag = "heatmap" if plot_heatmap else "crossesOnly"
            fname = (f"ResonancesSigma_{tag}"
                     f"_lam{lam_min:.3f}-{lam_max:.3f}.{file_format}")
            fig_path = os.path.join(self.save_path, fname)
            fig.savefig(fig_path, format=file_format, dpi=300,
                        bbox_inches="tight")
            if verbose:
                print(f"✓ Figure saved to {fig_path}")

        plt.show()
