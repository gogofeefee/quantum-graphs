#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMatrixPlotsMixin — S-matrix plotting methods.

Public methods
--------------
plot_S_amplitude        — |S_ij|² vs wavelength or frequency.
plot_S_phase            — arg(S_ij) vs wavelength or frequency.
plot_S_amplitude_phase  — combined two-panel figure.
plot_S_fourier          — Fourier transform of |S_ij|² (optical-length domain).
plot_S_complex_plane    — S_ij trajectory in the complex plane.

All methods share the same S-matrix cache via ``open_graphs.cache.get_S_full``.
Style helpers are imported from ``open_graphs.plotting.style``.
"""

from ...core.config import show_figure
from ...core.config import resolve
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.fft import fft, fftfreq, next_fast_len
from scipy.signal import windows

from ..cache import get_S_full
from .style import (
    lambda_to_freq, freq_to_lambda,
    validate_x_axis, format_bottom_xlabel,
    configure_y_axis, configure_phase_y_axis_pi,
    configure_primary_x_axis, add_secondary_top_axis,
    set_x_limits, break_wrapped_phase_curve,
    MAJOR_TICK_LEN, MINOR_TICK_LEN, MAJOR_TICK_W, MINOR_TICK_W,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _components_to_list(comps):
    """Accept a single ``(i, j)`` tuple or a list of tuples."""
    if (isinstance(comps, tuple) and len(comps) == 2
            and all(isinstance(x, int) for x in comps)):
        return [comps]
    return list(comps)


def _build_plot_arrays(lambda_values, S_dict, choice):
    """
    Sort S arrays onto the display axis (wavelength or frequency).

    The raw dict is kept in k-uniform order (needed for FFT).
    Returns ``(x_plot, lambda_plot, freq_plot, S_plot, mod_sq_plot,
               phase_plot, phase_plot_broken)``.
    """
    lambda_values = np.asarray(lambda_values, dtype=float)
    freq_values   = lambda_to_freq(lambda_values)
    x_raw         = lambda_values if choice == "wavelength" else freq_values
    order         = np.argsort(x_raw)

    x_plot      = x_raw[order]
    lambda_plot = lambda_values[order]
    freq_plot   = freq_values[order]

    S_plot            = {}
    mod_sq_plot       = {}
    phase_plot        = {}
    phase_plot_broken = {}

    for key, Svals in S_dict.items():
        S_sorted = np.asarray(Svals)[order]
        phi      = np.angle(S_sorted)
        S_plot[key]            = S_sorted
        mod_sq_plot[key]       = np.abs(S_sorted)**2
        phase_plot[key]        = phi
        phase_plot_broken[key] = break_wrapped_phase_curve(x_plot, phi)

    return (x_plot, lambda_plot, freq_plot,
            S_plot, mod_sq_plot, phase_plot, phase_plot_broken)


def _convert_spectrum_to_x(spectrum, choice):
    if spectrum is None:
        return None
    spec        = np.asarray(spectrum)
    lambda_spec = 2.0 * np.pi / np.real(spec)
    return lambda_spec if choice == "wavelength" else lambda_to_freq(lambda_spec)


def _spectrum_vlines(ax, x_spec):
    if x_spec is not None:
        for x0 in np.asarray(x_spec):
            ax.axvline(x0, color="red", linestyle="--", linewidth=1)


def _amp_ylabel(components):
    if len(components) == 1:
        i, j = components[0]
        return rf"$|S_{{{i}{j}}}|^2$"
    return r"$|S_{ij}|^2$"


def _phase_ylabel(components):
    if len(components) == 1:
        i, j = components[0]
        return rf"$\arg(S_{{{i}{j}}})$"
    return r"$\arg(S_{ij})$"


def _comp_tag(components):
    return "_".join(f"S{i}{j}" for i, j in components)


def _plot_filename(prefix, components, spectrum, x_axis,
                   lambda_min, lambda_max, n_points, k_imag, file_format):
    spec_tag = "withSpec" if spectrum is not None else "noSpec"
    x_tag    = "vsLambda" if x_axis == "wavelength" else "vsFreq"
    return (f"{prefix}_{_comp_tag(components)}_{spec_tag}_{x_tag}_"
            f"lam{lambda_min:.3f}-{lambda_max:.3f}_"
            f"N{int(n_points)}_kIm{k_imag:.2e}.{file_format}")


# ---------------------------------------------------------------------------
# Helpers for complex-plane crossing detection
# ---------------------------------------------------------------------------

def _prepare_spectrum_arrays(spectrum):
    if spectrum is None:
        return None
    spec = np.asarray(spectrum)
    spec = np.real(spec[np.isfinite(spec)])
    if spec.size == 0:
        return None
    spec = np.sort(spec)
    return {
        "k":         spec,
        "lambda_um": 2.0 * np.pi / spec,
        "freq_thz":  lambda_to_freq(2.0 * np.pi / spec),
    }


def _nearest_resonance(k_cross, spectrum_data):
    if spectrum_data is None:
        return None
    k_spec = spectrum_data["k"]
    idx    = int(np.argmin(np.abs(k_spec - k_cross)))
    k_res  = float(k_spec[idx])
    lam_cross = 2.0 * np.pi / k_cross
    lam_res   = float(spectrum_data["lambda_um"][idx])
    return {
        "idx":       idx,
        "k":         k_res,
        "lambda_um": lam_res,
        "freq_thz":  float(spectrum_data["freq_thz"][idx]),
        "delta_k":   float(k_cross - k_res),
        "delta_lam": float(lam_cross - lam_res),
    }


def _find_negative_real_crossings(Svals, lambda_vals, real_k, tol_im=1e-14):
    """
    Detect crossings of the negative real half-axis (Im(S)=0, Re(S)≤0)
    by linear interpolation between consecutive sample points.
    """
    re_v = np.real(Svals)
    im_v = np.imag(Svals)
    n    = len(Svals)
    crossings = []

    for m in range(n - 1):
        x0, y0 = re_v[m], im_v[m]
        x1, y1 = re_v[m+1], im_v[m+1]

        if abs(y0) <= tol_im and x0 <= 0:
            crossings.append({
                "re": float(x0), "im": float(y0),
                "lambda_um": float(lambda_vals[m]),
                "k_real":    float(real_k[m]),
                "freq_thz":  float(lambda_to_freq(lambda_vals[m])),
            })

        if ((y0 < 0 < y1) or (y0 > 0 > y1)
                or (abs(y1) <= tol_im and abs(y0) > tol_im)):
            denom = y1 - y0
            if abs(denom) <= tol_im:
                continue
            t = -y0 / denom
            if 0.0 <= t <= 1.0:
                x_cross = x0 + t * (x1 - x0)
                if x_cross <= 0:
                    lam_c = lambda_vals[m] + t * (lambda_vals[m+1] - lambda_vals[m])
                    k_c   = real_k[m]      + t * (real_k[m+1]      - real_k[m])
                    crossings.append({
                        "re": float(x_cross), "im": 0.0,
                        "lambda_um": float(lam_c),
                        "k_real":    float(k_c),
                        "freq_thz":  float(lambda_to_freq(lam_c)),
                    })

    # Remove duplicates (same wavelength within 1e-12 µm)
    dedup = []
    for c in crossings:
        if dedup and abs(c["lambda_um"] - dedup[-1]["lambda_um"]) < 1e-12:
            continue
        dedup.append(c)
    return dedup


# ===========================================================================
# Mixin
# ===========================================================================

class SMatrixPlotsMixin:
    """Mixin providing S-matrix spectral and complex-plane plots."""

    # ------------------------------------------------------------------
    # Shared data preparation
    # ------------------------------------------------------------------

    def _get_S_data(self, lambda_min, lambda_max, n_points, k_imag=0.0,
                    reuse_if_included=True, verbose=None):
        """
        Return ``(lambda_values, real_k, delta_k, S_full)`` for the grid.

        ``S_full`` shape: ``(n_points, n_leads, n_leads)``, complex128.
        Delegates caching to ``cache.get_S_full``.
        """
        verbose = resolve("verbose", verbose)
        return get_S_full(self, lambda_min, lambda_max, n_points, k_imag,
                          reuse_if_included=reuse_if_included, verbose=verbose)

    def _extract_components(self, S_full, components):
        """Validate component indices and return ``{(i,j): array}``."""
        n_ports = S_full.shape[1]
        for i, j in components:
            if not (0 <= i < n_ports and 0 <= j < n_ports):
                raise IndexError(
                    f"Component S[{i},{j}] out of bounds for "
                    f"{n_ports}×{n_ports} S-matrix."
                )
        return {(i, j): S_full[:, i, j] for i, j in components}

    # ------------------------------------------------------------------
    # plot_S_amplitude
    # ------------------------------------------------------------------

    def plot_S_amplitude(self, lambda_min, lambda_max, n_points, components,
                         spectrum=None, k_imag=0.0, x_axis="wavelength",
                         save=False, save_fig=None, file_format="pdf",
                         reuse_if_included=True, verbose=None):
        """
        Plot |S_ij|² versus wavelength or frequency.

        Parameters
        ----------
        lambda_min, lambda_max : float
            Wavelength range (µm).
        n_points : int
            Number of uniformly-spaced k-grid points.
        components : (i, j) or list of (i, j)
            S-matrix components to plot.
        spectrum : array-like or None
            Resonance wavenumbers (µm⁻¹); drawn as vertical red dashed lines.
        k_imag : float
            Imaginary part added to k.
        x_axis : ``'wavelength'`` or ``'frequency'``
            Horizontal axis variable.
        save : bool
            Save the figure to ``self.save_path``.
        file_format : str
            Output format (``'pdf'``, ``'svg'``, ``'png'``, …).
        reuse_if_included : bool
            Re-use a broader cached S(λ) if it covers the range.
        verbose : bool
            Print progress.
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        verbose = resolve("verbose", verbose)
        x_axis     = validate_x_axis(x_axis)
        components = _components_to_list(components)

        lambda_vals, real_k, _, S_full = self._get_S_data(
            lambda_min, lambda_max, n_points, k_imag, reuse_if_included, verbose)
        S_dict = self._extract_components(S_full, components)

        x_plot, _, _, _, mod_sq, _, _ = _build_plot_arrays(lambda_vals, S_dict, x_axis)
        x_spec = _convert_spectrum_to_x(spectrum, x_axis)

        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")

        for i, j in components:
            kw = (dict(color="black", linewidth=1.2) if len(components) == 1
                  else dict(linewidth=2, label=rf"$|S_{{{i}{j}}}|^2$"))
            ax.plot(x_plot, mod_sq[(i, j)], **kw)

        _spectrum_vlines(ax, x_spec)
        configure_primary_x_axis(ax, show_bottom=True, labelbottom=True,
                                  xlabel=format_bottom_xlabel(x_axis))
        configure_y_axis(ax, ylabel=_amp_ylabel(components))
        set_x_limits(ax, x_axis, lambda_min, lambda_max)
        add_secondary_top_axis(ax, x_axis, labelpad=16)

        if len(components) > 1:
            ax.legend(fontsize=18, frameon=False)

        if save:
            fname    = _plot_filename("Samp", components, spectrum, x_axis,
                                      lambda_min, lambda_max, n_points, k_imag, file_format)
            filepath = os.path.join(self.save_path, fname)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Amplitude figure saved to {filepath}")

        show_figure()

    # ------------------------------------------------------------------
    # plot_S_phase
    # ------------------------------------------------------------------

    def plot_S_phase(self, lambda_min, lambda_max, n_points, components,
                     spectrum=None, k_imag=0.0, x_axis="wavelength",
                     save=False, save_fig=None, file_format="pdf",
                     reuse_if_included=True, verbose=None):
        """
        Plot arg(S_ij) versus wavelength or frequency.

        Phase is wrapped to [−π, π] and branch jumps appear as
        discontinuities (no spurious vertical lines).

        Parameters
        ----------
        (same as ``plot_S_amplitude``)
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        verbose = resolve("verbose", verbose)
        x_axis     = validate_x_axis(x_axis)
        components = _components_to_list(components)

        lambda_vals, _, _, S_full = self._get_S_data(
            lambda_min, lambda_max, n_points, k_imag, reuse_if_included, verbose)
        S_dict = self._extract_components(S_full, components)

        x_plot, _, _, _, _, _, phase_broken = _build_plot_arrays(lambda_vals, S_dict, x_axis)
        x_spec = _convert_spectrum_to_x(spectrum, x_axis)

        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")

        for i, j in components:
            xp, yp = phase_broken[(i, j)]
            kw = (dict(color="black", linewidth=1.2) if len(components) == 1
                  else dict(linewidth=2, label=rf"$\arg(S_{{{i}{j}}})$"))
            ax.plot(xp, yp, **kw)

        _spectrum_vlines(ax, x_spec)
        configure_primary_x_axis(ax, show_bottom=True, labelbottom=True,
                                  xlabel=format_bottom_xlabel(x_axis))
        configure_phase_y_axis_pi(ax, ylabel=_phase_ylabel(components))
        set_x_limits(ax, x_axis, lambda_min, lambda_max)
        add_secondary_top_axis(ax, x_axis, labelpad=16)

        if len(components) > 1:
            ax.legend(fontsize=18, frameon=False)

        if save:
            fname    = _plot_filename("Sphase", components, spectrum, x_axis,
                                      lambda_min, lambda_max, n_points, k_imag, file_format)
            filepath = os.path.join(self.save_path, fname)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Phase figure saved to {filepath}")

        show_figure()

    # ------------------------------------------------------------------
    # plot_S_amplitude_phase
    # ------------------------------------------------------------------

    def plot_S_amplitude_phase(self, lambda_min, lambda_max, n_points, components,
                                spectrum=None, k_imag=0.0, x_axis="wavelength",
                                save=False, save_fig=None, file_format="pdf",
                                reuse_if_included=True, verbose=None):
        """
        Two-panel figure: |S_ij|² (top) and arg(S_ij) (bottom), shared x-axis.

        Parameters
        ----------
        (same as ``plot_S_amplitude``)
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        verbose = resolve("verbose", verbose)
        x_axis     = validate_x_axis(x_axis)
        components = _components_to_list(components)

        lambda_vals, _, _, S_full = self._get_S_data(
            lambda_min, lambda_max, n_points, k_imag, reuse_if_included, verbose)
        S_dict = self._extract_components(S_full, components)

        x_plot, _, _, _, mod_sq, _, phase_broken = _build_plot_arrays(
            lambda_vals, S_dict, x_axis)
        x_spec = _convert_spectrum_to_x(spectrum, x_axis)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
            layout="constrained",
        )
        # Top: |S|²
        for i, j in components:
            lw = 2 if len(components) > 1 else 1.2
            kw = dict(color="black", linewidth=lw)
            if len(components) > 1:
                kw["label"] = rf"$|S_{{{i}{j}}}|^2$"
            ax_top.plot(x_plot, mod_sq[(i, j)], **kw)

        _spectrum_vlines(ax_top, x_spec)
        configure_primary_x_axis(ax_top, show_bottom=True, labelbottom=False)
        configure_y_axis(ax_top, ylabel=_amp_ylabel(components))
        set_x_limits(ax_top, x_axis, lambda_min, lambda_max)
        add_secondary_top_axis(ax_top, x_axis, labelpad=12)

        # Bottom: phase
        for i, j in components:
            xp, yp = phase_broken[(i, j)]
            lw = 2 if len(components) > 1 else 1.2
            kw = dict(color="black", linewidth=lw)
            if len(components) > 1:
                kw["label"] = f"S[{i},{j}]"
            ax_bot.plot(xp, yp, **kw)

        _spectrum_vlines(ax_bot, x_spec)
        configure_primary_x_axis(ax_bot, show_bottom=True, show_top=True,
                                  labelbottom=True,
                                  xlabel=format_bottom_xlabel(x_axis))
        configure_phase_y_axis_pi(ax_bot, ylabel=_phase_ylabel(components))
        set_x_limits(ax_bot, x_axis, lambda_min, lambda_max)

        if len(components) > 1:
            ax_top.legend(fontsize=18, frameon=False)
            ax_bot.legend(fontsize=16, frameon=False)

        if save:
            fname    = _plot_filename("SampPhase", components, spectrum, x_axis,
                                      lambda_min, lambda_max, n_points, k_imag, file_format)
            filepath = os.path.join(self.save_path, fname)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Amplitude+phase figure saved to {filepath}")

        show_figure()

    # ------------------------------------------------------------------
    # plot_S_fourier
    # ------------------------------------------------------------------

    def plot_S_fourier(self, lambda_min, lambda_max, n_points, components,
                       k_imag=0.0, xlim_um=(0, 8000),
                       save=False, save_fig=None, file_format="pdf",
                       reuse_if_included=True, verbose=None):
        """
        Fourier transform of |S_ij|² in the optical-length domain.

        Uses a Hann window on the k-uniform grid and zero-pads to the next
        fast FFT size (≥ 2²²).

        Parameters
        ----------
        lambda_min, lambda_max : float
            Wavelength range (µm).
        n_points : int
            Number of k-grid points.
        components : (i, j) or list of (i, j)
            S-matrix components to transform.
        k_imag : float
            Imaginary part of k.
        xlim_um : (float, float)
            Optical-length axis range (µm).
        save : bool
            Save the figure.
        file_format : str
            Output format.
        reuse_if_included : bool
            Re-use broader cached S(λ) if available.
        verbose : bool
            Print progress.
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        verbose = resolve("verbose", verbose)
        components = _components_to_list(components)

        lambda_vals, real_k, delta_k, S_full = self._get_S_data(
            lambda_min, lambda_max, n_points, k_imag, reuse_if_included, verbose)

        n_fft = max(next_fast_len(int(n_points)), 2**22)
        win   = windows.hann(len(real_k))

        fig, axs = plt.subplots(len(components), 1, figsize=(14, 5 * len(components)), sharex=True)
        if len(components) == 1:
            axs = [axs]
        fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.12, hspace=0)

        max_val = 0.0

        for idx, (i, j) in enumerate(components):
            # Use k-uniform order (NOT sorted by λ) for correct FFT
            signal   = np.abs(S_full[:, i, j])**2 * win
            fft_vals = np.abs(fft(signal, n=n_fft))[:n_fft // 2]
            opt_len  = 2.0 * np.pi * fftfreq(n_fft, d=delta_k)[:n_fft // 2]

            axs[idx].plot(opt_len, fft_vals, color="black", lw=1)
            axs[idx].set_xlim(*xlim_um)
            axs[idx].set_yticks([])

            non_dc  = fft_vals[50:] if fft_vals.size > 50 else fft_vals
            max_val = max(max_val, float(np.max(non_dc)))

            if len(components) > 1:
                axs[idx].text(0.98, 0.90, rf"$S_{{{i}{j}}}$",
                              ha="right", va="top",
                              transform=axs[idx].transAxes, fontsize=20)

        for ax_ in axs:
            ax_.set_ylim(0, 1.1 * max_val)
            ax_.xaxis.set_major_locator(MultipleLocator(1000))
            ax_.xaxis.set_minor_locator(MultipleLocator(500))
            ax_.tick_params(axis="x", which="major",
                            bottom=True, top=False, direction="out",
                            length=MAJOR_TICK_LEN, width=MAJOR_TICK_W, labelsize=20)
            ax_.tick_params(axis="x", which="minor",
                            bottom=True, top=False, direction="out",
                            length=MINOR_TICK_LEN, width=MINOR_TICK_W)
            ax_.tick_params(axis="y", which="major",
                            left=True, right=True, direction="in",
                            length=MAJOR_TICK_LEN, width=MAJOR_TICK_W)

            # Mirror top ticks without labels
            ax_top = ax_.twiny()
            ax_top.set_xlim(ax_.get_xlim())
            ax_top.xaxis.set_major_locator(MultipleLocator(1000))
            ax_top.xaxis.set_minor_locator(MultipleLocator(500))
            ax_top.tick_params(axis="x", which="major",
                               bottom=False, top=True, direction="in",
                               length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                               labeltop=False)
            ax_top.tick_params(axis="x", which="minor",
                               bottom=False, top=True, direction="in",
                               length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                               labeltop=False)
            for spine in ("bottom", "left", "right"):
                ax_top.spines[spine].set_visible(False)
            ax_top.patch.set_alpha(0)

        axs[-1].set_xlabel("Optical length (µm)", fontsize=22)
        if len(components) == 1:
            axs[0].set_ylabel("Fourier Transform", fontsize=22)
        else:
            fig.text(0.08, 0.5, "Fourier Transform", fontsize=22,
                     va="center", rotation="vertical")

        if save:
            fname    = _plot_filename("Sfourier", components, None, "wavelength",
                                      lambda_min, lambda_max, n_points, k_imag, file_format)
            filepath = os.path.join(self.save_path, fname)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Fourier figure saved to {filepath}")

        show_figure()

    # ------------------------------------------------------------------
    # plot_S_complex_plane
    # ------------------------------------------------------------------

    def plot_S_complex_plane(self, lambda_min, lambda_max, n_points, components,
                              spectrum=None, k_imag=0.0,
                              save=False, save_fig=None, file_format="pdf",
                              reuse_if_included=False, verbose=None):
        """
        Plot S_ij(λ) as a curve in the complex plane.

        Detects crossings of the negative real half-axis (Im(S)=0, Re(S)≤0)
        and marks them with red dots.  Prints wavelength, frequency, and
        nearest resonance when ``spectrum`` is provided.

        Parameters
        ----------
        lambda_min, lambda_max : float
            Wavelength range (µm).
        n_points : int
            Number of k-grid points.
        components : (i, j) or list of (i, j)
            S-matrix components to plot.
        spectrum : array-like or None
            Resonance wavenumbers (µm⁻¹) for nearest-resonance lookup.
        k_imag : float
            Imaginary part of k.
        save : bool
            Save the figure.
        file_format : str
            Output format.
        reuse_if_included : bool
            Re-use broader cached S(λ) if available.  Default ``False``
            to avoid inadvertently mixing different ``k_imag`` values.
        verbose : bool
            Print crossing info.

        Returns
        -------
        fig : Figure
        axs : ndarray of Axes  (shape ``(1, n_components)``)
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        verbose = resolve("verbose", verbose)
        components = _components_to_list(components)

        lambda_vals, real_k, _, S_full = self._get_S_data(
            lambda_min, lambda_max, n_points, k_imag, reuse_if_included, verbose)
        S_dict        = self._extract_components(S_full, components)
        spectrum_data = _prepare_spectrum_arrays(spectrum)

        n_comp = len(components)
        fig, axs = plt.subplots(1, n_comp, figsize=(8 * n_comp, 7), squeeze=False)
        theta_unit = np.linspace(0, 2*np.pi, 500)

        for col, (i, j) in enumerate(components):
            ax = axs[0, col]
            Sv = S_dict[(i, j)]

            ax.plot(np.real(Sv), np.imag(Sv), color="steelblue", lw=0.8)
            ax.axhline(0, color="gray", lw=0.6, zorder=0)
            ax.axvline(0, color="gray", lw=0.6, zorder=0)
            ax.plot(np.cos(theta_unit), np.sin(theta_unit),
                    "k--", lw=0.6, alpha=0.4)

            crossings = _find_negative_real_crossings(Sv, lambda_vals, real_k)
            for c in crossings:
                ax.scatter(c["re"], c["im"], color="red", s=60, zorder=5)
                if verbose:
                    print(f"  Crossing S[{i},{j}]: "
                          f"λ = {c['lambda_um']:.5f} µm  "
                          f"({c['lambda_um']*1e3:.3f} nm)  "
                          f"f = {c['freq_thz']:.4f} THz")
                    nr = _nearest_resonance(c["k_real"], spectrum_data)
                    if nr is not None:
                        print(f"    → nearest resonance (idx={nr['idx']}): "
                              f"λ = {nr['lambda_um']:.5f} µm  "
                              f"Δλ = {nr['delta_lam']*1e3:+.4e} nm")

            title = (rf"$S_{{{i}{j}}}$ in the complex plane"
                     if n_comp == 1 else rf"$S_{{{i}{j}}}$")
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(r"$\mathrm{Re}(S_{ij})$", fontsize=18)
            ax.set_ylabel(r"$\mathrm{Im}(S_{ij})$", fontsize=18)
            ax.set_aspect("equal")
            ax.tick_params(axis="both", which="major", direction="in",
                           length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                           labelsize=14, top=True, right=True)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="minor", direction="in",
                           length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                           top=True, right=True)

        plt.tight_layout()

        if save:
            fname    = _plot_filename("ScomplexPlane", components, spectrum, "wavelength",
                                      lambda_min, lambda_max, n_points, k_imag, file_format)
            filepath = os.path.join(self.save_path, fname)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Complex-plane figure saved to {filepath}")

        show_figure()
        return fig, axs
