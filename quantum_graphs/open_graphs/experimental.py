#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExperimentalMixin — experimental resonance detection via d²T/dk² peaks.
"""

from ..core.config import resolve
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ..core.persistence import use_cache, write_cache_meta


class ExperimentalMixin:

    # -------------------------------------------------------------------------
    # Derivative helpers
    # -------------------------------------------------------------------------

    def finite_diff_derivatives_uniform_x(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple:
        """
        Compute (y, dy/dx, d2y/dx2) on a *uniformly spaced* x-grid.

        - Interior points: 5-point, 4th-order accurate centered stencils
        - First 2 and last 2 points: 2nd-order one-sided stencils

        Parameters
        ----------
        x : ndarray
            1D array, strictly increasing and uniformly spaced.
        y : ndarray
            1D array (same shape as x).

        Returns
        -------
        y : ndarray
        dydx : ndarray
        d2ydx2 : ndarray
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if x.size != y.size:
            raise ValueError("x and y must have the same length.")
        if x.size < 5:
            raise ValueError("Need at least 5 points for 5-point stencils.")

        dx = float(x[1] - x[0])
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("Invalid spacing dx.")

        if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-14):
            raise ValueError("x is not uniformly spaced; provide a uniform grid.")

        dydx   = np.empty_like(y)
        d2ydx2 = np.empty_like(y)

        # interior: 5-point, 4th-order
        dydx[2:-2] = (-y[4:] + 8.0 * y[3:-1] - 8.0 * y[1:-3] + y[:-4]) / (12.0 * dx)
        d2ydx2[2:-2] = (
            -y[4:] + 16.0 * y[3:-1] - 30.0 * y[2:-2] + 16.0 * y[1:-3] - y[:-4]
        ) / (12.0 * dx * dx)

        # edges: 2nd-order one-sided
        dydx[0]  = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dx)
        dydx[1]  = (-3.0 * y[1] + 4.0 * y[2] - y[3]) / (2.0 * dx)
        dydx[-2] = (3.0 * y[-2] - 4.0 * y[-3] + y[-4]) / (2.0 * dx)
        dydx[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * dx)

        d2ydx2[0]  = (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / (dx * dx)
        d2ydx2[1]  = (2.0 * y[1] - 5.0 * y[2] + 4.0 * y[3] - y[4]) / (dx * dx)
        d2ydx2[-2] = (2.0 * y[-2] - 5.0 * y[-3] + 4.0 * y[-4] - y[-5]) / (dx * dx)
        d2ydx2[-1] = (2.0 * y[-1] - 5.0 * y[-2] + 4.0 * y[-3] - y[-4]) / (dx * dx)

        return y, dydx, d2ydx2

    def k_to_lambda_um(self, k_real: np.ndarray) -> np.ndarray:
        """λ (µm) from real k (µm⁻¹), with λ = 2π/k."""
        k_real = np.asarray(k_real, dtype=float)
        return 2.0 * np.pi / k_real

    def transform_derivatives_k_to_lambda(
        self,
        k_real: np.ndarray,
        dT_dk: np.ndarray,
        d2T_dk2: np.ndarray,
    ) -> tuple:
        """
        Convert derivatives wrt k to derivatives wrt λ using the chain rule
        for λ = 2π/k (k real).

        dk/dλ = -k²/(2π)
        d²k/dλ² = k³/(2π²)
        """
        k       = np.asarray(k_real, dtype=float)
        dTdk    = np.asarray(dT_dk, dtype=float)
        d2Tdk2  = np.asarray(d2T_dk2, dtype=float)

        dk_dlam  = -(k**2) / (2.0 * np.pi)
        d2k_dlam2 = (k**3) / (2.0 * np.pi**2)

        dT_dlam   = dTdk * dk_dlam
        d2T_dlam2 = (dk_dlam**2) * d2Tdk2 + dTdk * d2k_dlam2
        return dT_dlam, d2T_dlam2

    # -------------------------------------------------------------------------
    # Three-panel plot helper
    # -------------------------------------------------------------------------

    def plot_three_panel_derivatives(
        self,
        lam_um: np.ndarray,
        T: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
        component: tuple,
        peaks_lam_um=None,
        title: str = None,
        xlim=None,
    ):
        """
        3 stacked subplots: top T, middle first derivative, bottom second derivative.
        Optionally overlay peak markers (red stars) at peaks_lam_um on all three panels.
        """
        i, j = component

        lam_um = np.asarray(lam_um, dtype=float).ravel()
        T  = np.asarray(T,  dtype=float).ravel()
        d1 = np.asarray(d1, dtype=float).ravel()
        d2 = np.asarray(d2, dtype=float).ravel()

        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.05},
        )

        ax0.plot(lam_um, T, "k-", lw=1.5)
        ax0.set_ylabel(rf"$|S_{{{i}{j}}}|^2$", fontsize=22)
        if title is not None:
            ax0.set_title(title, fontsize=18)

        ax1.plot(lam_um, d1, "k-", lw=1.5)
        ax1.set_ylabel(r"$d/d\lambda$ (arb.)", fontsize=22)

        ax2.plot(lam_um, d2, "k-", lw=1.5)
        ax2.set_ylabel(r"$d^2/d\lambda^2$ (arb.)", fontsize=22)
        ax2.set_xlabel("Wavelength (µm)", fontsize=22)

        if xlim is not None:
            ax2.set_xlim(*xlim)
        else:
            ax2.set_xlim(lam_um.min(), lam_um.max())

        def _set_ylim_with_margin(ax, y, frac=0.05):
            y = np.asarray(y, dtype=float)
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                return
            m = frac * (ymax - ymin) if ymax > ymin else frac
            ax.set_ylim(ymin - m, ymax + m)

        _set_ylim_with_margin(ax0, T,  frac=0.05)
        _set_ylim_with_margin(ax1, d1, frac=0.05)
        _set_ylim_with_margin(ax2, d2, frac=0.05)

        if peaks_lam_um is not None and len(peaks_lam_um) > 0:
            peaks_lam_um = np.asarray(peaks_lam_um, dtype=float).ravel()
            T_pk  = np.interp(peaks_lam_um, lam_um, T)
            d1_pk = np.interp(peaks_lam_um, lam_um, d1)
            d2_pk = np.interp(peaks_lam_um, lam_um, d2)
            ax0.plot(peaks_lam_um, T_pk,  "r*", markersize=14)
            ax1.plot(peaks_lam_um, d1_pk, "r*", markersize=14)
            ax2.plot(peaks_lam_um, d2_pk, "r*", markersize=14)

        for ax in (ax0, ax1, ax2):
            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", direction="in",
                           length=6, width=2, labelsize=22, top=True, right=True)
            ax.tick_params(axis="both", which="minor", direction="in",
                           length=3, width=1.5, top=True, right=True)

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Main method: find_experimental_resonances
    # -------------------------------------------------------------------------

    def find_experimental_resonances(
        self,
        n_points: int = 5_000_000,
        component: tuple = (1, 0),
        k_imag: float = 1e-7,
        prominence=None,
        distance: int = 5,
        amp_threshold: float = 2e8,
        manual_lambda_peaks=None,
        exclude_lambda_peaks=None,
        exclude_tol_nm: float = 2e-4,
        plot_figures: bool = True,
        save: bool = True,
        overwrite: bool = False,
        verbose: bool = None,
        file_format: str = "pdf",
    ):
        """
        Detect resonances from the theoretical spectrum by using peaks of d²T/dk²,
        where T(k) = |S_ij(k)|² evaluated on a uniform real k grid.

        Caching logic
        -------------
        - Cache #1: resonances (k values) saved as .npy
        - Cache #2: computed complex S_ij(k) saved as .npz (stores k_real + S_ij)

        Loading rules
        -------------
        - If plot_figures=False:
            * If resonance .npy exists and overwrite=False: load and return immediately.
            * Else compute S → T → derivatives → resonances; optionally save.
        - If plot_figures=True:
            * Always ensure S is available (load S cache if possible; else compute).
            * Resonances may be loaded from .npy if available.
            * Plot: λ-domain (full + 2 zooms) and k-domain (full-range 3-panel).

        Peak logic
        ----------
        - Peaks are found on d²T/dk² using scipy.signal.find_peaks.
        - Peaks with height < amp_threshold are rejected.
        - manual_lambda_peaks (µm) are injected (λ → k) within the computed window.
        - exclude_lambda_peaks (nm) are removed if |λ_peak - λ_excl| <= exclude_tol_nm.

        Returns
        -------
        self.exp_spectrum : np.ndarray
            Sorted array of detected resonances (k in µm⁻¹).
        """
        verbose = resolve("verbose", verbose)

        # ------------------------------------------------------------------
        # Local helpers
        # ------------------------------------------------------------------
        def _sanitize_float(x: float) -> str:
            return f"{x:.12g}".replace(".", "p").replace("+", "").replace("-", "m")

        def _progress_print(idx, total, last_pct, prefix="Computing S..."):
            if not verbose:
                return last_pct
            pct = int(100 * (idx + 1) / total)
            if pct != last_pct and (pct % 1 == 0):
                print(f"\r{prefix} {pct:3d}%", end="", flush=True)
                return pct
            return last_pct

        def _finish_progress(prefix="Computing S..."):
            if verbose:
                print(f"\r{prefix} 100%")

        def _dedup_sorted(x: np.ndarray, tol: float) -> np.ndarray:
            x = np.asarray(x, dtype=float).ravel()
            if x.size <= 1 or tol <= 0:
                return x
            out = [x[0]]
            for v in x[1:]:
                if abs(v - out[-1]) > tol:
                    out.append(v)
            return np.asarray(out, dtype=float)

        def _exclude_by_lambda(lam_um_all: np.ndarray) -> np.ndarray:
            if exclude_lambda_peaks is None or len(exclude_lambda_peaks) == 0:
                return lam_um_all
            excl_um  = 1e-3 * np.asarray(exclude_lambda_peaks, dtype=float).ravel()
            excl_um  = np.sort(excl_um)
            tol_um   = float(exclude_tol_nm) * 1e-3
            x        = lam_um_all
            idx      = np.searchsorted(excl_um, x)
            idx0     = np.clip(idx - 1, 0, excl_um.size - 1)
            idx1     = np.clip(idx,     0, excl_um.size - 1)
            d0       = np.abs(x - excl_um[idx0])
            d1_      = np.abs(x - excl_um[idx1])
            dmin     = np.minimum(d0, d1_)
            mask_keep = dmin > tol_um
            removed   = int(np.sum(~mask_keep))
            if verbose:
                print(f"✓ Exclusion: removed {removed} peaks within {exclude_tol_nm:g} nm of exclude list.")
            return x[mask_keep]

        # ------------------------------------------------------------------
        # Basic quantities / paths
        # ------------------------------------------------------------------
        i, j = component
        lam_min, lam_max = self.wavelength_range

        workdir = getattr(self, "save_path", None)
        if workdir is None:
            workdir = os.getcwd()
        os.makedirs(workdir, exist_ok=True)

        k_min, k_max = 2.0 * np.pi / lam_max, 2.0 * np.pi / lam_min
        k_real = np.linspace(k_min, k_max, int(n_points))
        k_eval = k_real + 1j * float(k_imag)

        prom_tag     = f"prom{_sanitize_float(prominence)}" if prominence is not None else "promNone"
        dist_tag     = f"dist{int(distance)}"
        thr_tag      = f"thr{_sanitize_float(amp_threshold)}"
        kim_tag      = f"kIm{_sanitize_float(k_imag)}"
        lam_tag      = f"lam{_sanitize_float(lam_min)}-{_sanitize_float(lam_max)}"
        N_tag        = f"N{int(n_points)}"
        manual_count = len(manual_lambda_peaks) if manual_lambda_peaks is not None else 0
        manual_tag   = f"manual{manual_count}"
        excl_count   = len(exclude_lambda_peaks) if exclude_lambda_peaks is not None else 0
        excl_tag     = f"excl{excl_count}_tol{_sanitize_float(exclude_tol_nm)}nm"

        res_filename = (
            f"experimental_resonances_d2dk2__S{i}{j}__{lam_tag}__{N_tag}__{kim_tag}__"
            f"{prom_tag}__{dist_tag}__{thr_tag}__{manual_tag}__{excl_tag}.npy"
        )
        res_path = os.path.join(workdir, res_filename)

        S_filename = f"cache_Sij__S{i}{j}__{lam_tag}__{N_tag}__{kim_tag}.npz"
        S_path     = os.path.join(workdir, S_filename)
        S_params   = dict(component=[i, j], wavelength_range=list(self.wavelength_range),
                          n_points=int(n_points), k_imag=float(k_imag))
        res_params = dict(S_params, prominence=prominence, distance=int(distance),
                          amp_threshold=float(amp_threshold),
                          manual_lambda_peaks=sorted(manual_lambda_peaks or []),
                          exclude_lambda_peaks=sorted(exclude_lambda_peaks or []),
                          exclude_tol_nm=float(exclude_tol_nm))

        # ------------------------------------------------------------------
        # Fast path: load resonances only (no plotting requested)
        # ------------------------------------------------------------------
        if (not plot_figures) and save and use_cache(res_path, self.fingerprint, res_params,
                                                     overwrite, label="experimental resonances"):
            if verbose:
                print(f"✓ Loaded experimental resonances from file: {res_path}")
            self.exp_spectrum = np.load(res_path)
            return self.exp_spectrum

        # ------------------------------------------------------------------
        # Load resonances if available
        # ------------------------------------------------------------------
        loaded_resonances = False
        k_resonances      = None
        if save and use_cache(res_path, self.fingerprint, res_params, overwrite,
                              label="experimental resonances"):
            try:
                k_resonances      = np.load(res_path)
                loaded_resonances = True
                if verbose:
                    print(f"✓ Loaded experimental resonances from file: {res_path}")
            except Exception:
                loaded_resonances = False
                k_resonances      = None

        # ------------------------------------------------------------------
        # Ensure S is available
        # ------------------------------------------------------------------
        need_S = plot_figures or (not loaded_resonances) or overwrite

        S_ij = None
        if need_S:
            loaded_S = False
            if use_cache(S_path, self.fingerprint, S_params, overwrite, label="S_ij cache"):
                try:
                    data = np.load(S_path, allow_pickle=False)
                    if "k_real" in data and "S_ij" in data:
                        k_cached = np.asarray(data["k_real"], dtype=float)
                        S_cached = np.asarray(data["S_ij"], dtype=np.complex128)
                        if (k_cached.shape == k_real.shape and
                                np.allclose(k_cached, k_real, rtol=0.0, atol=0.0)):
                            S_ij     = S_cached
                            loaded_S = True
                            if verbose:
                                print(f"✓ Loaded cached S[{i},{j}](k) from: {S_path}")
                except Exception:
                    loaded_S = False

            if not loaded_S:
                if verbose:
                    print(f"✓ Computing S[{i},{j}](k) on {k_real.size} points (uniform k grid).")
                S_ij = self.model.S_many(k_eval)[:, i, j]

                if save:
                    payload = {
                        "k_real":          k_real,
                        "S_ij":            S_ij,
                        "meta_component":  np.array([i, j], dtype=int),
                        "meta_lambda_min": float(lam_min),
                        "meta_lambda_max": float(lam_max),
                        "meta_n_points":   int(n_points),
                        "meta_k_imag":     float(k_imag),
                    }
                    np.savez_compressed(S_path, **payload)
                    write_cache_meta(S_path, self.fingerprint, S_params)
                    if verbose:
                        print(f"✓ Cached S[{i},{j}](k) to: {S_path}")

        # ------------------------------------------------------------------
        # Compute resonances from T''(k) if needed
        # ------------------------------------------------------------------
        if (not loaded_resonances) or overwrite:
            if S_ij is None:
                raise RuntimeError("Internal error: S_ij is None but resonances need computation.")

            T = np.abs(S_ij) ** 2
            _, dT_dk, d2T_dk2 = self.finite_diff_derivatives_uniform_x(k_real, T)

            kwargs = {"distance": int(distance)}
            if prominence is not None:
                kwargs["prominence"] = float(prominence)

            peaks_all, _props = find_peaks(d2T_dk2, **kwargs)

            if peaks_all.size > 0:
                heights  = d2T_dk2[peaks_all]
                keep     = heights >= float(amp_threshold)
                peaks    = peaks_all[keep]
                rejected = int(peaks_all.size - peaks.size)
            else:
                peaks    = peaks_all
                rejected = 0

            if verbose:
                print(
                    f"✓ d²T/dk² peaks found: {int(peaks_all.size)} total, "
                    f"{int(peaks.size)} kept, {rejected} rejected "
                    f"(amp_threshold={amp_threshold:g})."
                )

            k_resonances = k_real[peaks]

            lam_um_auto = 2.0 * np.pi / k_resonances

            lam_um_manual = np.array([], dtype=float)
            if manual_lambda_peaks is not None and len(manual_lambda_peaks) > 0:
                manual_um = np.asarray(manual_lambda_peaks, dtype=float).ravel()
                lo_um     = float(min(lam_min, lam_max))
                hi_um     = float(max(lam_min, lam_max))
                mask      = (manual_um >= lo_um) & (manual_um <= hi_um)
                lam_um_manual = manual_um[mask]
                if verbose:
                    print(
                        f"✓ Manual peaks: {manual_um.size} provided, "
                        f"{lam_um_manual.size} within [{lo_um:.6f}, {hi_um:.6f}] µm."
                    )

            lam_um_all = np.concatenate([lam_um_auto, lam_um_manual])
            if lam_um_all.size == 0:
                if verbose:
                    print("⚠ No peaks found and no manual peaks within window. Spectrum empty.")
                self.exp_spectrum = np.array([], dtype=float)
                return self.exp_spectrum

            lam_um_all = np.sort(lam_um_all)
            lam_grid   = 2.0 * np.pi / k_real
            dlam       = float(np.median(np.abs(np.diff(lam_grid))))
            dedup_tol_um = 0.5 * dlam if np.isfinite(dlam) and dlam > 0 else 0.0
            lam_um_all   = _dedup_sorted(lam_um_all, dedup_tol_um)
            lam_um_all   = _exclude_by_lambda(lam_um_all)

            if lam_um_all.size == 0:
                if verbose:
                    print("⚠ After exclusion, no peaks remain. Spectrum empty.")
                self.exp_spectrum = np.array([], dtype=float)
                return self.exp_spectrum

            k_resonances = np.sort(2.0 * np.pi / lam_um_all)

            if verbose:
                print(f"✓ Total resonances (auto + manual - excluded): {len(k_resonances)}")

            if save:
                np.save(res_path, k_resonances)
                write_cache_meta(res_path, self.fingerprint, res_params)
                if verbose:
                    print(f"✓ Experimental resonances saved to {res_path}")

        self.exp_spectrum = np.asarray(k_resonances, dtype=float)

        # ------------------------------------------------------------------
        # Plotting
        # ------------------------------------------------------------------
        if not plot_figures:
            return self.exp_spectrum

        if S_ij is None:
            raise RuntimeError("Internal error: plot_figures=True but S_ij is None.")

        T = np.abs(S_ij) ** 2
        _, dT_dk, d2T_dk2 = self.finite_diff_derivatives_uniform_x(k_real, T)

        k_res_plot  = np.asarray(self.exp_spectrum, dtype=float)
        lam_res_plot = np.sort(2.0 * np.pi / k_res_plot)

        # (A) λ-domain plots
        lam_um       = self.k_to_lambda_um(k_real)
        dT_dlam, d2T_dlam2 = self.transform_derivatives_k_to_lambda(k_real, dT_dk, d2T_dk2)

        lam_plot = lam_um[::-1]
        T_plot   = T[::-1]
        d1_plot  = dT_dlam[::-1]
        d2_plot  = d2T_dlam2[::-1]

        title_full = (
            rf"Spectrum + derivatives vs $\lambda$ (S$_{{{i}{j}}}$), "
            rf"peaks from $d^2T/dk^2$ (thr={amp_threshold:g})"
        )
        fig_full = self.plot_three_panel_derivatives(
            lam_um=lam_plot, T=T_plot, d1=d1_plot, d2=d2_plot,
            component=component, peaks_lam_um=lam_res_plot,
            title=title_full, xlim=(lam_min, lam_max),
        )
        if save:
            fig_name = (
                f"Spectrum_3panel_lambda__d2dk2__S{i}{j}__{lam_tag}__{N_tag}__{kim_tag}__"
                f"{prom_tag}__{dist_tag}__{thr_tag}__{manual_tag}__{excl_tag}.{file_format}"
            )
            fig_full.savefig(os.path.join(workdir, fig_name),
                             format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ λ full-range 3-panel saved to {os.path.join(workdir, fig_name)}")
        plt.show()

        lam_range = lam_max - lam_min
        if lam_range > 0.02:
            center = 0.5 * (lam_max + lam_min)
            zoom_min, zoom_max = center - 0.01, center + 0.01
        else:
            zoom_min, zoom_max = lam_min, lam_max

        fig_z0 = self.plot_three_panel_derivatives(
            lam_um=lam_plot, T=T_plot, d1=d1_plot, d2=d2_plot,
            component=component, peaks_lam_um=None,
            title=f"Zoom vs λ (no markers) [{zoom_min:.3f}, {zoom_max:.3f}] µm",
            xlim=(zoom_min, zoom_max),
        )
        if save:
            zoom_name = (
                f"Zoom_3panel_lambda_nopeaks__S{i}{j}__lam{_sanitize_float(zoom_min)}-{_sanitize_float(zoom_max)}__"
                f"{N_tag}__{kim_tag}__{prom_tag}__{dist_tag}__{thr_tag}__{manual_tag}__{excl_tag}.{file_format}"
            )
            fig_z0.savefig(os.path.join(workdir, zoom_name),
                           format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ λ zoom (no markers) saved to {os.path.join(workdir, zoom_name)}")
        plt.show()

        mask_zoom    = (lam_res_plot >= zoom_min) & (lam_res_plot <= zoom_max)
        lam_res_zoom = lam_res_plot[mask_zoom]

        fig_z1 = self.plot_three_panel_derivatives(
            lam_um=lam_plot, T=T_plot, d1=d1_plot, d2=d2_plot,
            component=component, peaks_lam_um=lam_res_zoom,
            title=f"Zoom vs λ (with markers) [{zoom_min:.3f}, {zoom_max:.3f}] µm",
            xlim=(zoom_min, zoom_max),
        )
        if save:
            zoom_name2 = (
                f"Zoom_3panel_lambda_peaks__S{i}{j}__lam{_sanitize_float(zoom_min)}-{_sanitize_float(zoom_max)}__"
                f"{N_tag}__{kim_tag}__{prom_tag}__{dist_tag}__{thr_tag}__{manual_tag}__{excl_tag}.{file_format}"
            )
            fig_z1.savefig(os.path.join(workdir, zoom_name2),
                           format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ λ zoom (with markers) saved to {os.path.join(workdir, zoom_name2)}")
        plt.show()

        # (B) k-domain plot
        def _plot_three_panel_k(k, T, d1k, d2k, k_peaks=None, title=None):
            k   = np.asarray(k,   dtype=float).ravel()
            T   = np.asarray(T,   dtype=float).ravel()
            d1k = np.asarray(d1k, dtype=float).ravel()
            d2k = np.asarray(d2k, dtype=float).ravel()

            fig, (ax0, ax1, ax2) = plt.subplots(
                3, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.05},
            )
            ax0.plot(k, T,   "k-", lw=1.5)
            ax0.set_ylabel(rf"$|S_{{{i}{j}}}|^2$", fontsize=22)
            if title is not None:
                ax0.set_title(title, fontsize=18)
            ax1.plot(k, d1k, "k-", lw=1.5)
            ax1.set_ylabel(r"$dT/dk$ (arb.)", fontsize=22)
            ax2.plot(k, d2k, "k-", lw=1.5)
            ax2.set_ylabel(r"$d^2T/dk^2$ (arb.)", fontsize=22)
            ax2.set_xlabel(r"$k$ (µm$^{-1}$)", fontsize=22)

            def _ylim(ax, y, frac=0.05):
                y = np.asarray(y, dtype=float)
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
                if not np.isfinite(ymin) or not np.isfinite(ymax):
                    return
                m = frac * (ymax - ymin) if ymax > ymin else frac
                ax.set_ylim(ymin - m, ymax + m)

            _ylim(ax0, T,   frac=0.05)
            _ylim(ax1, d1k, frac=0.05)
            _ylim(ax2, d2k, frac=0.05)

            if k_peaks is not None and len(k_peaks) > 0:
                k_peaks = np.asarray(k_peaks, dtype=float).ravel()
                T_pk  = np.interp(k_peaks, k, T)
                d1_pk = np.interp(k_peaks, k, d1k)
                d2_pk = np.interp(k_peaks, k, d2k)
                ax0.plot(k_peaks, T_pk,  "r*", markersize=14)
                ax1.plot(k_peaks, d1_pk, "r*", markersize=14)
                ax2.plot(k_peaks, d2_pk, "r*", markersize=14)

            for ax in (ax0, ax1, ax2):
                ax.minorticks_on()
                ax.tick_params(axis="both", which="major", direction="in",
                               length=6, width=2, labelsize=22, top=True, right=True)
                ax.tick_params(axis="both", which="minor", direction="in",
                               length=3, width=1.5, top=True, right=True)
            fig.tight_layout()
            return fig

        title_k = rf"Spectrum + derivatives vs $k$ (S$_{{{i}{j}}}$), markers at extracted resonances"
        fig_k   = _plot_three_panel_k(k_real, T, dT_dk, d2T_dk2,
                                      k_peaks=k_res_plot, title=title_k)
        if save:
            figk_name = (
                f"Spectrum_3panel_k__d2dk2__S{i}{j}__{lam_tag}__{N_tag}__{kim_tag}__"
                f"{prom_tag}__{dist_tag}__{thr_tag}__{manual_tag}__{excl_tag}.{file_format}"
            )
            fig_k.savefig(os.path.join(workdir, figk_name),
                          format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ k full-range 3-panel saved to {os.path.join(workdir, figk_name)}")
        plt.show()

        return self.exp_spectrum
