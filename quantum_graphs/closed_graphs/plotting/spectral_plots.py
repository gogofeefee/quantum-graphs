#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralPlotsMixin — staircase, NNSD, long-range correlations, length spectrum.

All statistics accept ``wavelength_range_stats`` to restrict the analysis to a
sub-window (same API as open_graphs). Staircase and Weyl-deviation plots carry
a secondary x-axis (λ ↔ frequency).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .style import (
    C_UM_THz,
    MAJOR_TICK_LEN, MINOR_TICK_LEN, MAJOR_TICK_W, MINOR_TICK_W,
    lambda_to_freq, freq_to_lambda,
    add_secondary_top_axis,
    regular_major_locator,
)
from ..spectral_stats import NNSD_func, _get_rmt_funcs


def _tick_params(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in",
                   length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=22, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                   top=True, right=True)


def _filter_spectrum(spectrum, wavelength_range_stats):
    """Return (filtered_spectrum, range_tag)."""
    spectrum = np.asarray(spectrum, dtype=float)
    if wavelength_range_stats is None:
        return spectrum, ""
    lam_min_s, lam_max_s = wavelength_range_stats
    k_lo  = 2 * np.pi / lam_max_s
    k_hi  = 2 * np.pi / lam_min_s
    mask  = (spectrum >= k_lo) & (spectrum <= k_hi)
    tag   = f"_lam{lam_min_s:.3f}-{lam_max_s:.3f}"
    return spectrum[mask], tag


class SpectralPlotsMixin:

    # =========================================================================
    # Staircase
    # =========================================================================

    def plot_staircase_function(self, plot_weyl=True, weyl_diff=False,
                                mode=None,
                                wavelength_range_stats=None,
                                save_fig=False, file_format="pdf"):
        """
        Plot the spectral staircase N(k) with optional Weyl overlay.

        Parameters
        ----------
        plot_weyl : bool
        weyl_diff : bool
            Also produce N_Weyl(k) − N(k) vs mode index and vs wavelength.
        mode : str or None
        wavelength_range_stats : (float, float) or None
            Restrict to modes whose vacuum wavelength falls in this range (µm).
        save_fig : bool
        file_format : str
        """
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum. Run compute_spectrum() first.")

        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(mode=chosen_mode)
        spectrum_used, range_tag = _filter_spectrum(spectrum_used, wavelength_range_stats)
        spectrum = np.sort(np.real(spectrum_used))
        if spectrum.size == 0:
            raise ValueError("Empty spectrum after selection / filtering.")

        print(f"[Staircase:{method}] {note}")

        N_modes       = spectrum.size
        N_k           = np.arange(N_modes + 1)
        plot_spectrum = np.insert(spectrum, 0, spectrum[0])

        # ---- Main staircase ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
        ax.step(plot_spectrum, N_k, where="post", label=r"$N(k)$", color="k", lw=2)

        if plot_weyl:
            k_grid = np.linspace(spectrum[0], spectrum[-1], 1000)
            Nw     = self.N_weyl_scaled(k_grid, length_scale=weyl_scale)
            Nw    -= self.N_weyl_scaled(spectrum[0], length_scale=weyl_scale)
            Nw    += 0.5
            ax.plot(k_grid, Nw, "r--", lw=1.5,
                    label=r"$N_{\mathrm{Weyl}}(k)$")

        # secondary x-axis: convert k → λ (µm) on top
        # functions: k → λ = 2π/k, and λ → k = 2π/λ
        def k_to_lam(k_val):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 2 * np.pi / np.asarray(k_val, dtype=float)
        secax = ax.secondary_xaxis("top",
                                   functions=(k_to_lam, k_to_lam))
        secax.set_xlabel(r"Wavelength $\lambda$ (µm)", fontsize=22, labelpad=14)
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator
        secax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4,
                                                  steps=[1, 2, 2.5, 5, 10]))
        secax.xaxis.set_minor_locator(AutoMinorLocator(2))
        secax.tick_params(axis="x", which="major", top=True, bottom=False,
                          direction="in", length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                          labelsize=18, labeltop=True)
        secax.tick_params(axis="x", which="minor", top=True, bottom=False,
                          direction="in", length=MINOR_TICK_LEN, width=MINOR_TICK_W)

        L_eff = self.total_length * weyl_scale
        ax.set_title(fr"$L_{{\rm eff}} = {L_eff:.2f}\ \mu m$   ({method})",
                     fontsize=20)
        ax.set_xlabel(r"$k$ (µm$^{-1}$)", fontsize=22)
        ax.set_ylabel(r"$N(k)$", fontsize=22)
        ax.set_xlim(spectrum[0], spectrum[-1])
        ax.set_ylim(0, N_modes)
        _tick_params(ax)

        if save_fig:
            fname = f"staircase_function__{method}{range_tag}.{file_format}"
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
            print(f"✓ Staircase plot saved to: {fpath}")
        plt.show()

        # ---- Weyl deviation plots (optional) ----------------------------------
        if weyl_diff:
            N_weyl_k = self.N_weyl_scaled(spectrum, length_scale=weyl_scale)
            N_weyl_k -= self.N_weyl_scaled(spectrum[0], length_scale=weyl_scale)
            N_weyl_k += 0.5
            N_k_mid   = np.arange(1, N_modes + 1)
            delta     = N_weyl_k - N_k_mid

            # (a) Δ vs mode index
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.plot(np.arange(N_modes), delta, "k-", lw=2)
            ax2.axhline(0, color="gray", linestyle="--")
            ax2.set_xlabel("Mode index", fontsize=22)
            ax2.set_ylabel(r"$N_{\mathrm{Weyl}}(k) - N(k)$", fontsize=22)
            ax2.set_xlim(0, N_modes - 1)
            _tick_params(ax2)
            fig2.tight_layout()
            if save_fig:
                fname2 = f"staircase_diff_index__{method}{range_tag}.{file_format}"
                fpath2 = os.path.join(self.save_path, fname2)
                fig2.savefig(fpath2, format=file_format, dpi=300, bbox_inches="tight")
                print(f"✓ Staircase diff (index) saved to: {fpath2}")
            plt.show()

            # (b) Δ vs wavelength (nm)
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            lambda_nm = 2 * np.pi / spectrum * 1e3
            ax3.plot(lambda_nm, delta, "k-", lw=2)
            ax3.axhline(0, color="gray", linestyle="--")
            ax3.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=22)
            ax3.set_ylabel(r"$N_{\mathrm{Weyl}}(k) - N(k)$", fontsize=22)
            ax3.set_xlim(float(np.min(lambda_nm)), float(np.max(lambda_nm)))
            _tick_params(ax3)
            fig3.tight_layout()
            if save_fig:
                fname3 = f"staircase_diff_lambda__{method}{range_tag}.{file_format}"
                fpath3 = os.path.join(self.save_path, fname3)
                fig3.savefig(fpath3, format=file_format, dpi=300, bbox_inches="tight")
                print(f"✓ Staircase diff (λ) saved to: {fpath3}")
            plt.show()

    # =========================================================================
    # Short-range correlations (NNSD)
    # =========================================================================

    def short_range_correlations(self, width=0.1, xlim=None,
                                 plot_wigner=None,
                                 cumulative_spacing=False,
                                 color_list=None,
                                 mode=None,
                                 wavelength_range_stats=None,
                                 yscale="linear",
                                 save_fig=False, file_format="pdf"):
        """
        Nearest-Neighbour Spacing Distribution from the unfolded spectrum.

        Parameters
        ----------
        width : float            Histogram bin width.
        xlim : tuple or None     x-axis limits.
        plot_wigner : list       Ensembles to overlay: 'poissonian', 'goe', 'gue', 'gse'.
        cumulative_spacing : bool  Add CDF subplot.
        color_list : list or None  Matplotlib style strings for Wigner curves.
        mode : str or None
        wavelength_range_stats : (float, float) or None
        yscale : str             'linear' or 'log'.
        save_fig : bool
        file_format : str
        """
        if plot_wigner is None:
            plot_wigner = ["poissonian", "goe"]

        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum. Run compute_spectrum() first.")

        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(mode=chosen_mode)
        spectrum_used, range_tag = _filter_spectrum(spectrum_used, wavelength_range_stats)
        spectrum_used = np.sort(np.real(spectrum_used))
        if spectrum_used.size < 2:
            raise ValueError("Too few modes after selection / filtering.")
        print(f"[NNSD:{method}] {note}")

        unfolded = self._compute_unfolded_spectrum(spectrum_used, mode=method)
        spacings = np.diff(unfolded)

        if xlim is None:
            xlim = (0.0, float(np.max(spacings)) if spacings.size else 1.0)

        bins = np.arange(xlim[0], xlim[1] + width, width)

        if cumulative_spacing:
            fig, (ax, ax_cdf) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [2, 1], "hspace": 0},
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        hist_values, bin_edges = np.histogram(spacings, bins=bins, density=True)

        ax.step(bin_edges[1:], hist_values, where="pre", color="black", linewidth=2)
        if hist_values.size > 0:
            ax.plot([bins[0], bins[0] + width], [hist_values[0], hist_values[0]],
                    "-", lw=2, color="black")

        s_vals = np.linspace(xlim[0], xlim[1], 1000)
        if color_list is None:
            color_list = ["r-", "b-.", "g--", "orange", "purple"]
        wigner_labels = {"poissonian": "Poisson", "goe": "GOE",
                         "gue": "GUE", "gse": "GSE"}

        for i, name_func in enumerate(plot_wigner):
            fmt = color_list[i % len(color_list)]
            ax.plot(s_vals, NNSD_func(s_vals, wigner=name_func), fmt, lw=2,
                    label=wigner_labels.get(name_func, name_func.upper()))

        ax.set_ylabel(r"$P(s)$", fontsize=22)
        ax.set_xlabel("Spacing $s$", fontsize=22)
        ax.set_xlim(xlim)
        ymax_data = float(np.max(hist_values)) if hist_values.size else 1.0
        # also consider the max of plotted Wigner/Poisson curves
        if plot_wigner:
            wigner_max = max(
                float(np.max(NNSD_func(s_vals, wigner=n))) for n in plot_wigner
            )
            ymax_data = max(ymax_data, wigner_max)
        ax.set_yscale(yscale)
        if yscale == "log":
            ymin_log = max(1e-4, float(np.min(hist_values[hist_values > 0])) * 0.5) \
                       if hist_values.size and np.any(hist_values > 0) else 1e-4
            ax.set_ylim(ymin_log, max(ymax_data * 1.8, 1e-9))
        else:
            ymin_lin = -0.04 * ymax_data if cumulative_spacing else 0.0
            ax.set_ylim(ymin_lin, max(ymax_data * 1.05, 1e-9))
        ax.legend(fontsize=16, loc="upper right")
        _tick_params(ax)

        if cumulative_spacing:
            sp_sorted = np.sort(spacings)
            cdf_vals  = np.arange(1, sp_sorted.size + 1) / sp_sorted.size
            ax_cdf.scatter(sp_sorted, cdf_vals, marker="s",
                           edgecolor="black", facecolor="none",
                           s=30, linewidths=2, zorder=2)

            s_full = np.linspace(xlim[0], xlim[1], 1000)
            ds     = s_full[1] - s_full[0] if s_full.size > 1 else 1.0
            for i, name_func in enumerate(plot_wigner):
                fmt = color_list[i % len(color_list)]
                pdf = NNSD_func(s_full, wigner=name_func)
                ax_cdf.plot(s_full, np.cumsum(pdf) * ds, fmt, lw=2,
                            label=wigner_labels.get(name_func, name_func.upper()))

            ax_cdf.set_ylabel(r"Cumulative $P(s)$", fontsize=22)
            ax_cdf.set_xlabel("Spacing $s$", fontsize=22)
            ax_cdf.set_xlim(xlim)
            ax_cdf.set_ylim(0, 1.1)
            ax_cdf.legend(fontsize=14, loc="lower right")
            _tick_params(ax_cdf)

        fig.tight_layout()

        if save_fig:
            wigner_part    = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            cumul_part     = "_cumulative" if cumulative_spacing else ""
            fname = (f"NNSD_{wigner_part}_bin{width:.2f}"
                     f"{cumul_part}__{method}{range_tag}.{file_format}")
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
            print(f"✓ NNSD plot saved to: {fpath}")
        plt.show()

    # =========================================================================
    # Long-range correlations (Σ² and Δ₃)
    # =========================================================================

    def long_range_correlations(self, L_max=10, L_step=0.2,
                                plot_wigner=None,
                                correlation_type="both",
                                color_list=None,
                                mode=None,
                                wavelength_range_stats=None,
                                save_fig=False, file_format="pdf"):
        """
        Number variance Σ²(L) and spectral rigidity Δ₃(L).

        Parameters
        ----------
        L_max : float
        L_step : float
        plot_wigner : list
        correlation_type : {'number_variance', 'spectral_rigidity', 'both'}
        color_list : list or None
        mode : str or None
        wavelength_range_stats : (float, float) or None
        save_fig : bool
        file_format : str
        """
        if plot_wigner is None:
            plot_wigner = ["poissonian", "goe"]

        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum. Run compute_spectrum() first.")

        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(mode=chosen_mode)
        spectrum_used, range_tag = _filter_spectrum(spectrum_used, wavelength_range_stats)
        spectrum_used = np.sort(np.real(spectrum_used))
        if spectrum_used.size < 2:
            raise ValueError("Too few modes after selection / filtering.")
        print(f"[LongRange:{method}] {note}")

        unfolded  = self._compute_unfolded_spectrum(spectrum_used, mode=method)
        has_sigma = correlation_type in ("number_variance", "both")
        has_delta = correlation_type in ("spectral_rigidity", "both")

        L_sigma = sigma2_vals = L_delta = delta3_vals = None
        if has_sigma:
            L_sigma, sigma2_vals = self._compute_sigma2(unfolded, L_step, L_max)
        if has_delta:
            L_delta, delta3_vals = self._compute_delta3(unfolded, L_step, L_max)

        sigma2_RMT, delta3_RMT = _get_rmt_funcs()
        L_plot = np.linspace(0.0, L_max, 1000)

        if color_list is None:
            color_list = ["r-", "b-.", "g--", "orange", "purple"]
        wigner_labels = {"poissonian": "Poisson", "goe": "GOE",
                         "gue": "GUE", "gse": "GSE"}

        if correlation_type == "both":
            fig, (ax_sigma, ax_delta) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0},
            )
        else:
            fig, ax_single = plt.subplots(figsize=(10, 8))

        # ---- Σ²(L) -----------------------------------------------------------
        if has_sigma:
            ax_s = ax_sigma if correlation_type == "both" else ax_single
            ax_s.scatter(L_sigma, sigma2_vals,
                         edgecolor="black", facecolor="white",
                         s=60, linewidth=1.5, zorder=1)

            ymax = max(float(np.max(sigma2_vals)),
                       max((float(np.max(sigma2_RMT(L_plot, n)))
                            for n in (plot_wigner or [])
                            if n.lower() != "poissonian"), default=0)) * 1.05

            for i, name in enumerate(plot_wigner or []):
                ax_s.plot(L_plot, sigma2_RMT(L_plot, name),
                          color_list[i % len(color_list)], lw=2,
                          label=wigner_labels.get(name, name.upper()))

            ax_s.set_ylabel(r"$\Sigma^2(L)$", fontsize=22)
            ax_s.set_xlim(0.0, L_max)
            ax_s.set_ylim(0.0, max(ymax, 0.1))
            ax_s.legend(fontsize=18, loc="lower right")
            _tick_params(ax_s)
            if correlation_type == "number_variance":
                ax_s.set_xlabel("Interval length $L$", fontsize=22)

        # ---- Δ₃(L) -----------------------------------------------------------
        if has_delta:
            ax_d = ax_delta if correlation_type == "both" else ax_single
            ax_d.scatter(L_delta, delta3_vals,
                         edgecolor="black", facecolor="white",
                         s=60, linewidth=1.5, zorder=1)

            ymax = max(float(np.max(delta3_vals)),
                       max((float(np.max(delta3_RMT(L_plot, n)))
                            for n in (plot_wigner or [])
                            if n.lower() != "poissonian"), default=0)) * 1.05

            for i, name in enumerate(plot_wigner or []):
                ax_d.plot(L_plot, delta3_RMT(L_plot, name),
                          color_list[i % len(color_list)], lw=2,
                          label=wigner_labels.get(name, name.upper()))

            ax_d.set_ylabel(r"$\Delta_3(L)$", fontsize=22)
            ax_d.set_xlabel("Interval length $L$", fontsize=22)
            ax_d.set_xlim(0.0, L_max)
            ax_d.set_ylim(0.0, max(ymax, 0.1))
            ax_d.legend(fontsize=18, loc="lower right")
            _tick_params(ax_d)

        fig.tight_layout()

        if save_fig:
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            fname = (f"LongRange_{correlation_type}_{wigner_part}_"
                     f"Lmax{L_max}_Lstep{L_step}__{method}{range_tag}.{file_format}")
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
            print(f"✓ Long-range correlation plot saved to: {fpath}")
        plt.show()

    # =========================================================================
    # Length spectrum
    # =========================================================================

    def plot_length_spectrum(self, L_max=4999, n_points=100_000,
                             save_fig=False, file_format="pdf",
                             ignore_zero_peak=True, zero_window_um=None):
        """
        ρ̂(ℓ) = |Σ_j exp(−i k_j n_eff ℓ)|  — periodic-orbit length spectrum.

        Prefers the degenerate spectrum when available.

        Parameters
        ----------
        L_max : float
        n_points : int
        save_fig : bool
        file_format : str
        ignore_zero_peak : bool
            Exclude the ℓ ≈ 0 DC peak from ylim computation.
        zero_window_um : float or None
        """
        if hasattr(self, "degenerate_spectrum") and self.degenerate_spectrum is not None and \
                np.asarray(self.degenerate_spectrum).size > 0:
            k_vals     = np.asarray(self.degenerate_spectrum, dtype=np.complex128)
            source_tag = "__deg"
        elif hasattr(self, "spectrum") and self.spectrum is not None and \
                np.asarray(self.spectrum).size > 0:
            k_vals     = np.asarray(self.spectrum, dtype=np.complex128)
            source_tag = ""
        else:
            raise AttributeError("No spectrum. Run compute_spectrum() first.")

        ell   = np.linspace(0.0, float(L_max), int(n_points))
        d_ell = float(ell[1] - ell[0]) if ell.size >= 2 else float(L_max)

        # Vectorised n_eff(k)
        if callable(self.neff):
            try:
                n_eff_vec = self.neff(np.real(k_vals))
                if np.isscalar(n_eff_vec):
                    n_eff_vec = np.full(k_vals.shape, float(n_eff_vec))
                else:
                    n_eff_vec = np.asarray(n_eff_vec, dtype=float)
                    if n_eff_vec.shape != k_vals.shape:
                        n_eff_vec = np.array([float(self.neff(float(np.real(k))))
                                              for k in k_vals])
            except Exception:
                n_eff_vec = np.array([float(self.neff(float(np.real(k))))
                                      for k in k_vals])
        else:
            n_eff_vec = np.full(k_vals.shape, float(self.neff))

        rho_hat = np.zeros_like(ell, dtype=np.complex128)
        for kj, neff_j in zip(k_vals, n_eff_vec):
            rho_hat += np.exp(-1j * kj * neff_j * ell)
        rho_hat = np.abs(rho_hat)

        if ignore_zero_peak:
            if zero_window_um is None:
                zero_window_um = max(100.0, 50 * d_ell)
            mask_ylim = ell > float(zero_window_um)
            ymax = float(np.max(rho_hat[mask_ylim])) if np.any(mask_ylim) else \
                   float(np.max(rho_hat[1:]))
        else:
            ymax = float(rho_hat.max())

        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(ell, rho_hat, color="black", lw=1.5)
        ax.set_xlim(0.0, float(L_max))
        ax.set_ylim(0.0, 1.05 * ymax)
        ax.set_xlabel(r"Length $\ell$ ($\mu$m)", fontsize=22)
        ax.set_ylabel(r"$|\hat{\rho}(\ell)|$", fontsize=22)
        _tick_params(ax)
        fig.tight_layout()

        if save_fig:
            fname = f"LengthSpectrum_Lmax{L_max}_N{n_points}{source_tag}.{file_format}"
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
            print(f"✓ Length spectrum saved to {fpath}")
        plt.show()
