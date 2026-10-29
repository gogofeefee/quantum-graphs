#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralPlotsMixin — staircase, NNSD, and long-range correlation plots.

Depends on SpectralStatsMixin (spectral_stats.py) for Weyl-law and
unfolding computations.
"""

from ...core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt

from ..spectral_stats import NNSD_func, _get_rmt_funcs


class SpectralPlotsMixin:

    # -------------------------------------------------------------------------
    # Staircase
    # -------------------------------------------------------------------------

    def plot_staircase_function(self, plot_weyl=True, weyl_diff=False,
                                mode=None, use_experimental=False,
                                wavelength_range_stats=None,
                                save_fig=False, file_format="pdf"):
        """
        Plot the spectral staircase N(k) with optional Weyl overlay and
        deviation sub-plots.

        Parameters
        ----------
        plot_weyl : bool
            Overlay the smooth Weyl prediction.
        weyl_diff : bool
            Also produce N_Weyl(k) − N(k) vs mode index and vs wavelength.
        mode : str or None
            Spectrum selection mode (None → auto).
        use_experimental : bool
            Use ``self.exp_spectrum`` instead of ``self.spectrum``.
        wavelength_range_stats : (float, float) or None
            Restrict the statistics to modes whose vacuum wavelength (µm) falls
            within this range.  ``None`` uses the full spectrum.
        save_fig : bool
            Save figures to ``self.save_path``.
        file_format : str
            Output format.
        """
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(
            mode=chosen_mode, use_experimental=use_experimental)

        spectrum_used = np.asarray(spectrum_used, dtype=float)
        range_tag = ""
        if wavelength_range_stats is not None:
            lam_min_s, lam_max_s = wavelength_range_stats
            k_lo = 2 * np.pi / lam_max_s
            k_hi = 2 * np.pi / lam_min_s
            mask = (spectrum_used >= k_lo) & (spectrum_used <= k_hi)
            spectrum_used = spectrum_used[mask]
            range_tag = f"_lam{lam_min_s:.3f}-{lam_max_s:.3f}"

        spectrum = np.sort(np.real(spectrum_used))
        if spectrum.size == 0:
            raise ValueError("Empty spectrum after selection.")

        N_modes       = spectrum.size
        N_k           = np.arange(N_modes + 1)
        plot_spectrum = np.insert(spectrum, 0, spectrum[0])
        log(f"[Staircase:{method}] {note}")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.step(plot_spectrum, N_k, where="post",
                label=r"$N(k)$", color="k", lw=2)

        if plot_weyl:
            k_vals = np.linspace(spectrum[0], spectrum[-1], 1000)
            Nw     = self.N_weyl_scaled(k_vals, length_scale=weyl_scale, k0=spectrum[0])
            Nw    += 0.5
            ax.plot(k_vals, Nw, "r--", lw=1.5,
                    label=r"$N_{\mathrm{Weyl}}(k)$")

        L_eff = self.total_length * weyl_scale
        tag   = "__exp" if use_experimental else ""
        ax.set_title(
            rf"$L_{{\rm eff}} = {L_eff:.2f}\ \mu m$   "
            rf"({method}{', exp' if use_experimental else ''})",
            fontsize=20)
        ax.set_xlabel(r"$k$  (µm$^{-1}$)", fontsize=22)
        ax.set_ylabel(r"$N(k)$", fontsize=22)
        ax.set_xlim(spectrum[0], spectrum[-1])
        ax.set_ylim(0, N_modes)
        ax.legend(fontsize=16)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in",
                       length=6, width=2, labelsize=18, top=True, right=True)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=3, width=1.5, top=True, right=True)
        fig.tight_layout()

        if save_fig:
            fp = os.path.join(self.save_path,
                              f"staircase_function__{method}{tag}{range_tag}.{file_format}")
            fig.savefig(fp, format=file_format, dpi=300, bbox_inches="tight")
            log(f"✓ Staircase plot saved to: {fp}")
        show_figure()

        if weyl_diff:
            N_weyl_k  = self.N_weyl_scaled(spectrum,
                                           length_scale=weyl_scale, k0=spectrum[0])
            N_weyl_k += 0.5
            delta     = N_weyl_k - np.arange(1, N_modes + 1)

            # vs mode index
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.plot(np.arange(N_modes), delta, "k-", lw=2)
            ax2.axhline(0, color="gray", linestyle="--")
            ax2.set_xlabel("Mode index", fontsize=22)
            ax2.set_ylabel(r"$N_{\mathrm{Weyl}}(k) - N(k)$", fontsize=22)
            ax2.set_xlim(0, N_modes - 1)
            ax2.minorticks_on()
            ax2.tick_params(axis="both", which="major", direction="in",
                            length=6, width=2, labelsize=18, top=True, right=True)
            ax2.tick_params(axis="both", which="minor", direction="in",
                            length=3, width=1.5, top=True, right=True)
            fig2.tight_layout()
            if save_fig:
                fp2 = os.path.join(self.save_path,
                                   f"staircase_diff_index__{method}{tag}{range_tag}.{file_format}")
                fig2.savefig(fp2, format=file_format, dpi=300, bbox_inches="tight")
                log(f"✓ Staircase diff (index) saved to: {fp2}")
            show_figure()

            # vs wavelength
            lambda_nm = 2 * np.pi / spectrum * 1e3
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ax3.plot(lambda_nm, delta, "k-", lw=2)
            ax3.axhline(0, color="gray", linestyle="--")
            ax3.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=22)
            ax3.set_ylabel(r"$N_{\mathrm{Weyl}}(k) - N(k)$", fontsize=22)
            ax3.set_xlim(np.min(lambda_nm), np.max(lambda_nm))
            ax3.minorticks_on()
            ax3.tick_params(axis="both", which="major", direction="in",
                            length=6, width=2, labelsize=18, top=True, right=True)
            ax3.tick_params(axis="both", which="minor", direction="in",
                            length=3, width=1.5, top=True, right=True)
            fig3.tight_layout()
            if save_fig:
                fp3 = os.path.join(self.save_path,
                                   f"staircase_diff_lambda__{method}{tag}{range_tag}.{file_format}")
                fig3.savefig(fp3, format=file_format, dpi=300, bbox_inches="tight")
                log(f"✓ Staircase diff (λ) saved to: {fp3}")
            show_figure()

    # -------------------------------------------------------------------------
    # Short-range: NNSD
    # -------------------------------------------------------------------------

    def short_range_correlations(self, width=0.1, xlim=None,
                                 plot_wigner=None,
                                 cumulative_spacing=False,
                                 color_list=None,
                                 mode=None,
                                 use_experimental=False,
                                 wavelength_range_stats=None,
                                 yscale="linear",
                                 save_fig=False,
                                 file_format="svg"):
        """
        Plot the nearest-neighbor spacing distribution (NNSD).

        Parameters
        ----------
        width : float
            Histogram bin width.
        xlim : (float, float) or None
            x-axis range.
        plot_wigner : list of str or None
            RMT distributions to overlay, e.g. ``['poissonian', 'goe']``.
        cumulative_spacing : bool
            Also plot the empirical CDF below the histogram.
        color_list : list or None
            Line styles for Wigner curves.
        mode : str or None
            Spectrum selection mode.
        use_experimental : bool
            Use ``self.exp_spectrum``.
        wavelength_range_stats : (float, float) or None
            Restrict the statistics to modes whose vacuum wavelength (µm) falls
            within this range.  ``None`` uses the full spectrum.
        yscale : str
            Y-axis scale for the P(s) panel: ``'linear'`` (default) or ``'log'``.
        save_fig : bool
            Save the figure.
        file_format : str
            Output format.
        """
        if plot_wigner is None:
            plot_wigner = ["poissonian", "goe"]

        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(
            mode=chosen_mode, use_experimental=use_experimental)

        spectrum_used = np.asarray(spectrum_used, dtype=float)
        range_tag = ""
        if wavelength_range_stats is not None:
            lam_min_s, lam_max_s = wavelength_range_stats
            k_lo = 2 * np.pi / lam_max_s
            k_hi = 2 * np.pi / lam_min_s
            mask = (spectrum_used >= k_lo) & (spectrum_used <= k_hi)
            spectrum_used = spectrum_used[mask]
            range_tag = f"_lam{lam_min_s:.3f}-{lam_max_s:.3f}"

        unfolded = self._compute_unfolded_spectrum(
            spectrum=spectrum_used, mode=chosen_mode,
            use_experimental=use_experimental)

        spacings = np.diff(np.sort(np.real(unfolded)))
        if spacings.size == 0:
            raise ValueError("Not enough levels to compute spacings.")

        if xlim is None:
            xlim = (0.0, float(np.max(spacings)))

        bins = np.arange(xlim[0], xlim[1] + width, width)
        if bins.size < 2:
            bins = np.array([xlim[0], xlim[1]])

        hist_values, bin_edges = np.histogram(spacings, bins=bins, density=True)

        if cumulative_spacing:
            fig, (ax, ax_cdf) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [2, 1], "hspace": 0})
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        ax.step(bin_edges[1:], hist_values, where="pre", color="black", linewidth=2)
        if hist_values.size > 0:
            ax.plot([xlim[0], min(xlim[0] + width, xlim[1])],
                    [hist_values[0], hist_values[0]], c="k", lw=2)

        s_vals = np.linspace(xlim[0], xlim[1], 1000)
        if color_list is None:
            color_list = ["r-", "b-.", "g--", "orange", "purple"]

        wigner_labels = {"poissonian": "Poisson", "goe": "GOE",
                         "gue": "GUE", "gse": "GSE"}

        for idx, name in enumerate(plot_wigner):
            style = color_list[idx % len(color_list)]
            ax.plot(s_vals, NNSD_func(s_vals, wigner=name), style, lw=2,
                    label=wigner_labels.get(name, name.upper()))

        tag = "__exp" if use_experimental else ""
        ax.set_ylabel(r"$P(s)$", fontsize=22)
        ax.set_xlabel("Spacing $s$", fontsize=22)
        ax.set_xlim(xlim)
        ax.set_yscale(yscale)
        ymax_data = float(np.max(hist_values)) if hist_values.size else 1.0
        if plot_wigner:
            wigner_max = max(
                float(np.max(NNSD_func(s_vals, wigner=n))) for n in plot_wigner
            )
            ymax_data = max(ymax_data, wigner_max)
        if yscale == "log":
            ymin_log = max(1e-4, float(np.min(hist_values[hist_values > 0])) * 0.5) \
                       if hist_values.size and np.any(hist_values > 0) else 1e-4
            ax.set_ylim(ymin_log, max(ymax_data * 3.0, 1e-9))
        else:
            ymin_lin = -0.04 * ymax_data if cumulative_spacing else 0.0
            ax.set_ylim(ymin_lin, max(ymax_data * 1.05, 1e-9))
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in",
                       length=6, width=2, labelsize=22, top=True, right=True)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=3, width=1.5, top=True, right=True)
        ax.legend(fontsize=20, loc="upper right")

        if cumulative_spacing:
            sorted_s = np.sort(spacings)
            cdf_vals = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
            ax_cdf.scatter(sorted_s, cdf_vals, marker="s",
                           edgecolor="black", facecolor="none",
                           s=30, linewidths=2, zorder=2)
            s_full = np.linspace(xlim[0], xlim[1], 1000)
            ds     = s_full[1] - s_full[0]
            for idx, name in enumerate(plot_wigner):
                style = color_list[idx % len(color_list)]
                pdf   = NNSD_func(s_full, wigner=name)
                ax_cdf.plot(s_full, np.cumsum(pdf) * ds, style, lw=2,
                            label=wigner_labels.get(name, name.upper()))
            ax_cdf.set_ylabel(r"Cumulative $P(s)$", fontsize=22)
            ax_cdf.set_xlabel("Spacing $s$", fontsize=22)
            ax_cdf.minorticks_on()
            ax_cdf.tick_params(axis="both", which="major", direction="in",
                               length=6, width=2, labelsize=22, top=True, right=True)
            ax_cdf.tick_params(axis="both", which="minor", direction="in",
                               length=3, width=1.5, top=True, right=True)

        fig.tight_layout()

        if save_fig:
            wigner_part     = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            cumulative_part = "__cumulative" if cumulative_spacing else ""
            yscale_part     = f"__yscale{yscale}" if yscale != "linear" else ""
            fname = (f"nnsd__{method}{tag}__{wigner_part}__bin{width:.2f}"
                     f"{cumulative_part}{yscale_part}{range_tag}.{file_format}")
            fp = os.path.join(self.save_path, fname)
            fig.savefig(fp, format=file_format, dpi=300, bbox_inches="tight")
            log(f"✓ NNSD plot saved to: {fp}")
        show_figure()

    # -------------------------------------------------------------------------
    # Long-range: Sigma2 and Delta3
    # -------------------------------------------------------------------------

    def long_range_correlations(self, L_max=10, L_step=0.2,
                                plot_wigner=None,
                                correlation_type="both",
                                color_list=None,
                                mode=None, use_experimental=False,
                                wavelength_range_stats=None,
                                save_fig=False, file_format="svg"):
        """
        Plot long-range spectral correlations: number variance Sigma2(L)
        and/or spectral rigidity Delta3(L).

        Parameters
        ----------
        L_max : float
            Maximum interval length (unfolded units).
        L_step : float
            Step size for the L grid.
        plot_wigner : list of str or None
            RMT reference curves to overlay.
        correlation_type : str
            ``'number_variance'``, ``'spectral_rigidity'``, or ``'both'``.
        color_list : list or None
            Line styles for the RMT curves.
        mode : str or None
            Spectrum selection mode.
        use_experimental : bool
            Use ``self.exp_spectrum``.
        wavelength_range_stats : (float, float) or None
            Restrict the statistics to modes whose vacuum wavelength (µm) falls
            within this range.  ``None`` uses the full spectrum.
        save_fig : bool
            Save the figure.
        file_format : str
            Output format.
        """
        if plot_wigner is None:
            plot_wigner = ["poissonian", "goe"]

        sigma2_RMT, delta3_RMT = _get_rmt_funcs()

        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(
            mode=chosen_mode, use_experimental=use_experimental)
        spectrum_used = np.sort(np.real(np.asarray(spectrum_used, dtype=float)))
        if spectrum_used.size == 0:
            raise ValueError("Empty spectrum after selection.")

        range_tag = ""
        if wavelength_range_stats is not None:
            lam_min_s, lam_max_s = wavelength_range_stats
            k_lo = 2 * np.pi / lam_max_s
            k_hi = 2 * np.pi / lam_min_s
            mask = (spectrum_used >= k_lo) & (spectrum_used <= k_hi)
            spectrum_used = spectrum_used[mask]
            range_tag = f"_lam{lam_min_s:.3f}-{lam_max_s:.3f}"
            if spectrum_used.size == 0:
                raise ValueError("Empty spectrum after wavelength_range_stats filtering.")

        log(f"[LongRange:{method}] {note}")

        unfolded = self._compute_unfolded_spectrum(
            spectrum=spectrum_used, mode=chosen_mode,
            use_experimental=use_experimental)

        has_sigma = correlation_type in ("number_variance", "both")
        has_delta = correlation_type in ("spectral_rigidity", "both")

        if has_sigma:
            L_sigma, sigma2_vals = self._compute_sigma2(unfolded, L_step, L_max)
        if has_delta:
            L_delta, delta3_vals = self._compute_delta3(unfolded, L_step, L_max)

        if color_list is None:
            color_list = ["r-", "b-.", "g--", "orange", "purple"]
        wigner_labels = {"poissonian": "Poisson", "goe": "GOE",
                         "gue": "GUE", "gse": "GSE"}

        if correlation_type == "both":
            fig, (ax_sigma, ax_delta) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0})
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        L_plot = np.linspace(0.0, L_max, 1000)

        if has_sigma:
            ax_s = ax_sigma if correlation_type == "both" else ax
            ax_s.scatter(L_sigma, sigma2_vals,
                         edgecolor="black", facecolor="white",
                         s=60, linewidth=1.5, zorder=1)
            ax_s.set_ylabel(r"$\Sigma^2(L)$", fontsize=22)
            ax_s.set_xlim(0.0, L_max)
            ymax_s = float(np.max(sigma2_vals)) if len(sigma2_vals) else 1.0
            for name in (plot_wigner or []):
                if name.lower() != "poissonian":
                    ymax_s = max(ymax_s,
                                 float(np.max(sigma2_RMT(L_plot, name))))
            for idx, name in enumerate(plot_wigner or []):
                ax_s.plot(L_plot, sigma2_RMT(L_plot, name),
                          color_list[idx % len(color_list)], lw=2,
                          label=wigner_labels.get(name, name.upper()))
            ax_s.set_ylim(0.0, ymax_s * 1.05)
            if correlation_type == "number_variance":
                ax_s.set_xlabel("Interval length $L$", fontsize=22)
            ax_s.minorticks_on()
            ax_s.tick_params(axis="both", which="major", direction="in",
                             length=6, width=2, labelsize=22, top=True, right=True)
            ax_s.tick_params(axis="both", which="minor", direction="in",
                             length=3, width=1.5, top=True, right=True)
            if correlation_type != "both":
                ax_s.legend(fontsize=18, loc="lower right")

        if has_delta:
            ax_d = ax_delta if correlation_type == "both" else ax
            ax_d.scatter(L_delta, delta3_vals,
                         edgecolor="black", facecolor="white",
                         s=60, linewidth=1.5, zorder=1)
            ax_d.set_ylabel(r"$\Delta_3(L)$", fontsize=22)
            ax_d.set_xlabel("Interval length $L$", fontsize=22)
            ax_d.set_xlim(0.0, L_max)
            ymax_d = float(np.max(delta3_vals)) if len(delta3_vals) else 1.0
            for name in (plot_wigner or []):
                if name.lower() != "poissonian":
                    ymax_d = max(ymax_d,
                                 float(np.max(delta3_RMT(L_plot, name))))
            for idx, name in enumerate(plot_wigner or []):
                ax_d.plot(L_plot, delta3_RMT(L_plot, name),
                          color_list[idx % len(color_list)], lw=2,
                          label=wigner_labels.get(name, name.upper()))
            ax_d.set_ylim(0.0, ymax_d * 1.05)
            ax_d.minorticks_on()
            ax_d.tick_params(axis="both", which="major", direction="in",
                             length=6, width=2, labelsize=22, top=True, right=True)
            ax_d.tick_params(axis="both", which="minor", direction="in",
                             length=3, width=1.5, top=True, right=True)
            ax_d.legend(fontsize=18, loc="lower right")

        fig.tight_layout()

        if save_fig:
            tag         = "__exp" if use_experimental else ""
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            fname = (f"longrange__{correlation_type}__{wigner_part}_"
                     f"Lmax{L_max}_Lstep{L_step}__{method}{tag}{range_tag}.{file_format}")
            fp = os.path.join(self.save_path, fname)
            fig.savefig(fp, format=file_format, dpi=300, bbox_inches="tight")
            log(f"✓ Long-range correlation plot saved to: {fp}")
        show_figure()

    # -------------------------------------------------------------------------
    # Length spectrum
    # -------------------------------------------------------------------------

    def plot_length_spectrum(self, L_max=5000, n_points=100_000,
                             use_experimental=False,
                             ignore_zero_peak=True, zero_window_um=None,
                             save_fig=False, file_format="pdf"):
        """
        Compute and plot the length spectrum ρ̂(ℓ) = |Σ_j exp(-i k_j n_eff(k_j) ℓ)|.

        Works with complex k values (numerical resonances) and with real k values
        (experimental peaks).  When ``use_experimental=True`` the k values are
        purely real, so the sum reduces to the standard Fourier-based length
        spectrum used in the periodic-orbit theory literature.

        Parameters
        ----------
        L_max : float
            Maximum length to plot (µm).
        n_points : int
            Number of points on the ℓ grid.
        use_experimental : bool
            If True, explicitly use ``self.exp_spectrum`` (real-axis peaks).
            If False (default), use ``self.spectrum`` if it has been computed,
            and fall back to ``self.exp_spectrum`` if not.
        ignore_zero_peak : bool
            If True, exclude a window near ℓ = 0 when setting the y-axis limit,
            so the large DC peak does not dominate the scale.
        zero_window_um : float or None
            Half-width (µm) of the exclusion window near ℓ = 0.
            Defaults to ``max(200, 50 * Δℓ)``.
        save_fig : bool
            Save the figure to ``self.save_path``.
        file_format : str
            Figure format (``"pdf"``, ``"png"``, …).
        """
        if use_experimental:
            if not hasattr(self, "exp_spectrum") or np.asarray(self.exp_spectrum).size == 0:
                raise AttributeError(
                    "use_experimental=True but exp_spectrum is missing or empty. "
                    "Run find_experimental_resonances() first."
                )
            k_vals = np.asarray(self.exp_spectrum, dtype=np.complex128)
            tag = "__exp"
            log(f"[LengthSpectrum] Using experimental spectrum ({k_vals.size} peaks).")
        else:
            # Auto: prefer complex spectrum, fall back to experimental
            if hasattr(self, "spectrum") and np.asarray(self.spectrum).size > 0:
                k_vals = np.asarray(self.spectrum, dtype=np.complex128)
                tag = ""
                log(f"[LengthSpectrum] Using numerical spectrum ({k_vals.size} resonances).")
            elif hasattr(self, "exp_spectrum") and np.asarray(self.exp_spectrum).size > 0:
                k_vals = np.asarray(self.exp_spectrum, dtype=np.complex128)
                tag = "__exp"
                log(f"[LengthSpectrum] Complex spectrum not found; "
                      f"using experimental spectrum ({k_vals.size} peaks).")
            else:
                raise AttributeError(
                    "No spectrum available. Run find_resonances() or "
                    "find_experimental_resonances() first."
                )

        # ℓ grid
        ell   = np.linspace(0.0, float(L_max), int(n_points))
        d_ell = ell[1] - ell[0] if ell.size >= 2 else float(L_max)

        # neff evaluated at Re(k) of each resonance
        if callable(self.neff):
            neff_vec = np.asarray(self.neff(np.real(k_vals)), dtype=float)
            if neff_vec.ndim == 0:
                neff_vec = np.full(k_vals.shape, float(neff_vec))
        else:
            neff_vec = np.full(k_vals.shape, float(self.neff))

        # ρ̂(ℓ) = | Σ_j exp(-i k_j neff_j ℓ) |
        rho_hat = np.zeros(ell.size, dtype=np.complex128)
        for kj, nj in zip(k_vals, neff_vec):
            rho_hat += np.exp(-1j * kj * nj * ell)
        rho_hat = np.abs(rho_hat)

        # y-axis limit: ignore DC spike near ℓ = 0
        if ignore_zero_peak:
            if zero_window_um is None:
                zero_window_um = max(200.0, 50.0 * d_ell)
            mask = ell > zero_window_um
            ymax = float(np.max(rho_hat[mask])) if mask.any() else float(np.max(rho_hat[1:]))
        else:
            ymax = float(rho_hat.max())
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ell, rho_hat, color="black", lw=1.0)
        ax.set_xlim(0.0, float(L_max))
        ax.set_ylim(0.0, 1.05 * ymax)
        ax.set_xlabel(r"Length $\ell$ (µm)", fontsize=22)
        ax.set_ylabel(r"$\hat{\rho}(\ell)$", fontsize=22)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in",
                       length=6, width=2, labelsize=22, top=True, right=True)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=3, width=1.5, top=True, right=True)
        plt.tight_layout()

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            fname = (f"length_spectrum__Lmax{L_max:.0f}_N{n_points}{tag}"
                     f".{file_format}")
            fp = os.path.join(self.save_path, fname)
            fig.savefig(fp, format=file_format, dpi=300, bbox_inches="tight")
            log(f"✓ Length spectrum saved to: {fp}")
        show_figure()
        return fig, ax
