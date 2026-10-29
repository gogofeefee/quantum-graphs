#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExperimentalMixin — resonances located as an experiment would: peaks of the
curvature d²T/dk² of a transmission T(k) = |S_ij(k)|² sampled on a real k grid
(slightly above the axis, ``k_imag``), as for a measured spectrum.

``find_experimental_resonances`` returns (and stores in ``exp_spectrum``) the
peak wavenumbers; S_ij(k) and the peaks are cached next to the graph.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from ..core.config import log, resolve, show_figure
from ..core.persistence import use_cache, write_cache_meta
from ..plotting.style import box_ticks, save_figure


def derivatives_uniform(x, y):
    """
    (y, dy/dx, d²y/dx²) on a uniformly spaced, increasing grid ``x``: 5-point
    4th-order centred stencils inside, 2nd-order one-sided ones on the first
    and last two points.
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

    dydx = np.empty_like(y)
    d2ydx2 = np.empty_like(y)
    dydx[2:-2] = (-y[4:] + 8.0 * y[3:-1] - 8.0 * y[1:-3] + y[:-4]) / (12.0 * dx)
    d2ydx2[2:-2] = (-y[4:] + 16.0 * y[3:-1] - 30.0 * y[2:-2] + 16.0 * y[1:-3] - y[:-4]) \
        / (12.0 * dx * dx)
    dydx[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dx)
    dydx[1] = (-3.0 * y[1] + 4.0 * y[2] - y[3]) / (2.0 * dx)
    dydx[-2] = (3.0 * y[-2] - 4.0 * y[-3] + y[-4]) / (2.0 * dx)
    dydx[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * dx)
    d2ydx2[0] = (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / (dx * dx)
    d2ydx2[1] = (2.0 * y[1] - 5.0 * y[2] + 4.0 * y[3] - y[4]) / (dx * dx)
    d2ydx2[-2] = (2.0 * y[-2] - 5.0 * y[-3] + 4.0 * y[-4] - y[-5]) / (dx * dx)
    d2ydx2[-1] = (2.0 * y[-1] - 5.0 * y[-2] + 4.0 * y[-3] - y[-4]) / (dx * dx)
    return y, dydx, d2ydx2


def derivatives_k_to_lambda(k, dT_dk, d2T_dk2):
    """dT/dλ and d²T/dλ² from the k derivatives (λ = 2π/k, k real)."""
    k = np.asarray(k, dtype=float)
    dk_dlam = -(k ** 2) / (2.0 * np.pi)
    d2k_dlam2 = (k ** 3) / (2.0 * np.pi ** 2)
    return dT_dk * dk_dlam, dk_dlam ** 2 * d2T_dk2 + dT_dk * d2k_dlam2


def _tag(x):
    return f"{x:.12g}".replace(".", "p").replace("+", "").replace("-", "m")


def _dedup_sorted(x, tol):
    """Drop the values closer than ``tol`` to the previous kept one."""
    if x.size <= 1 or tol <= 0:
        return x
    out = [x[0]]
    for v in x[1:]:
        if abs(v - out[-1]) > tol:
            out.append(v)
    return np.asarray(out, dtype=float)


def _exclude_near(lam_um, exclude_nm, tol_nm, verbose):
    """Remove the wavelengths within ``tol_nm`` of one of ``exclude_nm``."""
    if exclude_nm is None or len(exclude_nm) == 0 or lam_um.size == 0:
        return lam_um
    excl_um = np.sort(1e-3 * np.asarray(exclude_nm, dtype=float).ravel())
    idx = np.searchsorted(excl_um, lam_um)
    nearest = np.minimum(np.abs(lam_um - excl_um[np.clip(idx - 1, 0, excl_um.size - 1)]),
                         np.abs(lam_um - excl_um[np.clip(idx, 0, excl_um.size - 1)]))
    keep = nearest > float(tol_nm) * 1e-3
    log(f"✓ Exclusion: removed {int(np.sum(~keep))} peaks within {tol_nm:g} nm of exclude list.",
        verbose=verbose)
    return lam_um[keep]


def _three_panel(x, T, d1, d2, component, peaks=None, title=None, xlim=None,
                 xlabel="Wavelength (µm)", d1label=r"$d/d\lambda$ (arb.)",
                 d2label=r"$d^2/d\lambda^2$ (arb.)"):
    """T (top), its first and second derivatives (below) on a shared x axis,
    with red stars at ``peaks``. Returns the figure."""
    i, j = component
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]}, layout="constrained")
    fig.get_layout_engine().set(hspace=0.0)
    curves = [np.asarray(c, dtype=float).ravel() for c in (T, d1, d2)]
    for ax, y, ylabel in zip(axes, curves, (rf"$|S_{{{i}{j}}}|^2$", d1label, d2label)):
        ax.plot(x, y, "k-", lw=1.5)
        ax.set_ylabel(ylabel, fontsize=22)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        if np.isfinite(lo) and np.isfinite(hi):
            margin = 0.05 * (hi - lo) if hi > lo else 0.05
            ax.set_ylim(lo - margin, hi + margin)
        if peaks is not None and len(peaks) > 0:
            ax.plot(peaks, np.interp(peaks, x, y), "r*", markersize=14)
        box_ticks(ax)
    if title is not None:
        axes[0].set_title(title, fontsize=18)
    axes[-1].set_xlabel(xlabel, fontsize=22)
    axes[-1].set_xlim(*(xlim if xlim is not None else (np.min(x), np.max(x))))
    return fig


class ExperimentalMixin:
    """Resonances from the curvature of a transmission, as for measured data."""

    @property
    def best_spectrum(self):
        """The resonances of ``find_resonances`` if computed, else the peaks of
        ``find_experimental_resonances``, else None (e.g. to mark them on S plots)."""
        if np.asarray(getattr(self, "spectrum", [])).size > 0:
            return self.spectrum
        if np.asarray(getattr(self, "exp_spectrum", [])).size > 0:
            return self.exp_spectrum
        return None

    def find_experimental_resonances(
        self,
        n_points: int = 5_000_000,
        component: tuple = (1, 0),
        k_imag: float = 1e-7,
        prominence=None,
        distance: int = 5,
        amp_threshold: float = 0.0,
        manual_lambda_peaks=None,
        exclude_lambda_peaks=None,
        exclude_tol_nm: float = 2e-4,
        plot: bool = False,
        cache: bool = True,
        save_fig: bool = False,
        overwrite: bool = False,
        verbose: bool = None,
        file_format: str = "pdf",
        show: bool = None,
    ):
        """
        Resonances from the peaks of d²T/dk², T(k) = |S_ij(k)|² on a uniform
        grid of ``n_points`` real k (plus ``k_imag``) spanning ``wavelength_range``.

        Parameters
        ----------
        n_points : int
        component : (i, j)
            S-matrix element (lead i ← lead j).
        k_imag : float
            Imaginary part added to k (smooths the narrowest resonances).
        prominence, distance
            Passed to ``scipy.signal.find_peaks`` (distance in grid points).
        amp_threshold : float
            Peaks with d²T/dk² below this value (µm², absolute) are rejected.
            The peak height scales as T_peak / Γ², so the scale depends on the
            graph: the log reports the range of heights found. Default 0: every
            peak of ``find_peaks`` is kept.
        manual_lambda_peaks : list of float or None
            Wavelengths (µm) added to the detected peaks (those inside the window).
        exclude_lambda_peaks : list of float or None
            Wavelengths (nm) whose peaks are removed, within ``exclude_tol_nm``.
        plot : bool
            Draw T and its derivatives vs λ (full window, and a 20 nm zoom
            without and with markers) and vs k; ``save_fig`` saves them.
        cache : bool
            Load / save S_ij(k) and the peaks next to the graph (verified
            against the physics fingerprint; ``overwrite`` recomputes).
        verbose, file_format, show

        Returns
        -------
        ndarray : sorted resonance wavenumbers (µm⁻¹), also ``self.exp_spectrum``.
        """
        verbose = resolve("verbose", verbose)
        i, j = component
        lam_min, lam_max = self.wavelength_range
        k_real = np.linspace(2.0 * np.pi / lam_max, 2.0 * np.pi / lam_min, int(n_points))

        n_manual = len(manual_lambda_peaks) if manual_lambda_peaks is not None else 0
        n_excl = len(exclude_lambda_peaks) if exclude_lambda_peaks is not None else 0
        grid_tag = f"lam{_tag(lam_min)}-{_tag(lam_max)}__N{int(n_points)}__kIm{_tag(k_imag)}"
        peak_tag = (f"prom{_tag(prominence) if prominence is not None else 'None'}__"
                    f"dist{int(distance)}__thr{_tag(amp_threshold)}__manual{n_manual}__"
                    f"excl{n_excl}_tol{_tag(exclude_tol_nm)}nm")
        S_params = dict(component=[i, j], wavelength_range=list(self.wavelength_range),
                        n_points=int(n_points), k_imag=float(k_imag),
                        **(self._complex_k_params() if k_imag else {}))
        res_params = dict(S_params, prominence=prominence, distance=int(distance),
                          amp_threshold=float(amp_threshold),
                          manual_lambda_peaks=sorted(manual_lambda_peaks or []),
                          exclude_lambda_peaks=sorted(exclude_lambda_peaks or []),
                          exclude_tol_nm=float(exclude_tol_nm))
        res_path = os.path.join(self.save_path,
                                f"experimental_resonances_d2dk2__S{i}{j}__{grid_tag}__"
                                f"{peak_tag}.npy")

        k_res = None
        if cache and use_cache(res_path, self.fingerprint, res_params, overwrite,
                               label="experimental resonances"):
            k_res = np.load(res_path)
            log(f"✓ Loaded experimental resonances from file: {res_path}", verbose=verbose)
        if k_res is not None and not plot:
            self.exp_spectrum = k_res
            return self.exp_spectrum

        S_path = os.path.join(self.save_path, f"cache_Sij__S{i}{j}__{grid_tag}.npz")
        S_ij = self._component_on_grid(k_real, k_imag, component, S_path, S_params,
                                       cache, overwrite, verbose)
        T = np.abs(S_ij) ** 2
        _, dT_dk, d2T_dk2 = derivatives_uniform(k_real, T)

        if k_res is None:
            k_res = self._curvature_peaks(k_real, d2T_dk2, prominence, distance, amp_threshold,
                                          manual_lambda_peaks, exclude_lambda_peaks,
                                          exclude_tol_nm, verbose)
            if cache:
                np.save(res_path, k_res)
                write_cache_meta(res_path, self.fingerprint, res_params)
                log(f"✓ Experimental resonances saved to {res_path}", verbose=verbose)
        self.exp_spectrum = np.asarray(k_res, dtype=float)

        if plot:
            self._plot_experimental(k_real, T, dT_dk, d2T_dk2, component, amp_threshold,
                                    f"{grid_tag}__{peak_tag}", save_fig, file_format,
                                    verbose, show)
        return self.exp_spectrum

    def _component_on_grid(self, k_real, k_imag, component, path, params, cache, overwrite,
                           verbose):
        """S_ij(k_real + i k_imag), from the cache when it holds this grid."""
        i, j = component
        if cache and use_cache(path, self.fingerprint, params, overwrite, label="S_ij cache"):
            data = np.load(path, allow_pickle=False)
            if "S_ij" in data and data["S_ij"].shape == k_real.shape:
                log(f"✓ Loaded cached S[{i},{j}](k) from: {path}", verbose=verbose)
                return np.asarray(data["S_ij"], dtype=np.complex128)
        log(f"✓ Computing S[{i},{j}](k) on {k_real.size} points (uniform k grid).",
            verbose=verbose)
        S_ij = self.model.S_many(k_real + 1j * float(k_imag))[:, i, j]
        if cache:
            np.savez_compressed(path, k_real=k_real, S_ij=S_ij)
            write_cache_meta(path, self.fingerprint, params)
            log(f"✓ Cached S[{i},{j}](k) to: {path}", verbose=verbose)
        return S_ij

    def _curvature_peaks(self, k_real, d2T_dk2, prominence, distance, amp_threshold,
                         manual_lambda_peaks, exclude_lambda_peaks, exclude_tol_nm, verbose):
        """Peaks of d²T/dk² above ``amp_threshold``, plus the manual ones, minus
        the excluded ones, merged within half a grid step; sorted k."""
        kwargs = {"distance": int(distance)}
        if prominence is not None:
            kwargs["prominence"] = float(prominence)
        peaks_all, _ = find_peaks(d2T_dk2, **kwargs)
        heights = d2T_dk2[peaks_all]
        peaks = peaks_all[heights >= float(amp_threshold)]
        if verbose:
            span = (f", heights {heights.min():.3g} … {heights.max():.3g} µm²"
                    if heights.size else "")
            print(f"✓ d²T/dk² peaks found: {peaks_all.size} total, {peaks.size} kept, "
                  f"{peaks_all.size - peaks.size} rejected (amp_threshold={amp_threshold:g}"
                  f"{span}).")

        lam_min, lam_max = self.wavelength_range
        lam_um = 2.0 * np.pi / k_real[peaks]
        if manual_lambda_peaks is not None and len(manual_lambda_peaks) > 0:
            manual = np.asarray(manual_lambda_peaks, dtype=float).ravel()
            inside = manual[(manual >= lam_min) & (manual <= lam_max)]
            log(f"✓ Manual peaks: {manual.size} provided, {inside.size} within "
                f"[{lam_min:.6f}, {lam_max:.6f}] µm.", verbose=verbose)
            lam_um = np.concatenate([lam_um, inside])

        dlam = float(np.median(np.abs(np.diff(2.0 * np.pi / k_real))))
        lam_um = _dedup_sorted(np.sort(lam_um), 0.5 * dlam if np.isfinite(dlam) else 0.0)
        lam_um = _exclude_near(lam_um, exclude_lambda_peaks, exclude_tol_nm, verbose)
        if lam_um.size == 0:
            log("⚠ No resonance left (no peak, or all excluded).", verbose=verbose)
        else:
            log(f"✓ Total resonances (auto + manual - excluded): {lam_um.size}",
                verbose=verbose)
        return np.sort(2.0 * np.pi / lam_um)

    def _plot_experimental(self, k_real, T, dT_dk, d2T_dk2, component, amp_threshold, tag,
                           save_fig, file_format, verbose, show):
        """λ-domain figures (full window, 20 nm zoom without / with markers) and
        the k-domain figure of T and its derivatives, peaks as red stars."""
        i, j = component
        lam_min, lam_max = self.wavelength_range
        k_res = np.asarray(self.exp_spectrum, dtype=float)
        lam_res = np.sort(2.0 * np.pi / k_res)
        dT_dlam, d2T_dlam2 = derivatives_k_to_lambda(k_real, dT_dk, d2T_dk2)
        lam = (2.0 * np.pi / k_real)[::-1]
        curves = (T[::-1], dT_dlam[::-1], d2T_dlam2[::-1])

        if lam_max - lam_min > 0.02:
            centre = 0.5 * (lam_max + lam_min)
            zoom = (centre - 0.01, centre + 0.01)
        else:
            zoom = (lam_min, lam_max)
        in_zoom = lam_res[(lam_res >= zoom[0]) & (lam_res <= zoom[1])]
        zoom_tag = f"lam{_tag(zoom[0])}-{_tag(zoom[1])}"
        grid_free_tag = tag.split("__", 1)[1]      # tag without the window

        figures = [
            (_three_panel(lam, *curves, component, lam_res, xlim=(lam_min, lam_max),
                          title=rf"Spectrum + derivatives vs $\lambda$ (S$_{{{i}{j}}}$), "
                                rf"peaks from $d^2T/dk^2$ (thr={amp_threshold:g})"),
             f"Spectrum_3panel_lambda__d2dk2__S{i}{j}__{tag}", "λ full-range 3-panel"),
            (_three_panel(lam, *curves, component, None, xlim=zoom,
                          title=f"Zoom vs λ (no markers) [{zoom[0]:.3f}, {zoom[1]:.3f}] µm"),
             f"Zoom_3panel_lambda_nopeaks__S{i}{j}__{zoom_tag}__{grid_free_tag}",
             "λ zoom (no markers)"),
            (_three_panel(lam, *curves, component, in_zoom, xlim=zoom,
                          title=f"Zoom vs λ (with markers) [{zoom[0]:.3f}, {zoom[1]:.3f}] µm"),
             f"Zoom_3panel_lambda_peaks__S{i}{j}__{zoom_tag}__{grid_free_tag}",
             "λ zoom (with markers)"),
            (_three_panel(k_real, T, dT_dk, d2T_dk2, component, k_res,
                          title=rf"Spectrum + derivatives vs $k$ (S$_{{{i}{j}}}$), "
                                "markers at extracted resonances",
                          xlabel=r"$k$ (µm$^{-1}$)", d1label=r"$dT/dk$ (arb.)",
                          d2label=r"$d^2T/dk^2$ (arb.)"),
             f"Spectrum_3panel_k__d2dk2__S{i}{j}__{tag}", "k full-range 3-panel"),
        ]
        for fig, name, what in figures:
            if save_fig:
                save_figure(fig, self.save_path, f"{name}.{file_format}", what, verbose)
            show_figure(fig, show)
        return [fig for fig, _, _ in figures]
