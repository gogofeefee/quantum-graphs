"""
SpectralPlotsMixin — staircase, NNSD, long-range correlations and length
spectrum, shared by closed and open graphs.

Every method takes the selection of :mod:`quantum_graphs.core.spectral_stats`:
``sector`` (None = desymmetrised spectrum, an index, or "all"), an explicit
``spectrum`` (e.g. measured peaks ``graph.exp_spectrum``) and
``wavelength_range`` (µm) to restrict the levels to a sub-window. The default
random-matrix curves are Poisson and the ensemble expected for the selected
sectors (GOE with time-reversal symmetry, GUE otherwise).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from ..core.config import log, show_figure
from ..core.scattering import index_function
from ..stats.rmt import delta3_rmt, nnsd, sigma2_rmt
from ..stats.spectral import number_variance, spectral_rigidity
from .style import MAJOR_TICK_LEN, MAJOR_TICK_W, MINOR_TICK_LEN, MINOR_TICK_W

WIGNER_LABELS = {"poissonian": "Poisson", "goe": "GOE", "gue": "GUE", "gse": "GSE"}
DEFAULT_STYLES = ["r-", "b-.", "g--", "orange", "purple"]


def _tick_params(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in",
                   length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=22, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                   top=True, right=True)


def _file_tag(spectrum, sector, wavelength_range):
    """File-name suffix describing the selection."""
    tag = "" if spectrum is None else "_given"
    if sector is not None:
        tag += "_all" if sector == "all" else f"_sector{int(sector)}"
    if wavelength_range is not None:
        tag += f"_lam{wavelength_range[0]:.3f}-{wavelength_range[1]:.3f}"
    return tag


def _save(fig, graph, fname, file_format, what):
    fpath = os.path.join(graph.save_path, f"{fname}.{file_format}")
    fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
    log(f"✓ {what} saved to: {fpath}")


class SpectralPlotsMixin:

    # =========================================================================
    # Selection
    # =========================================================================

    def _plot_levels(self, spectrum, sector, wavelength_range, minimum, what):
        """Sorted real levels of the selection, with a one-line report."""
        k = self._stats_levels(spectrum, sector, wavelength_range)
        if k.size < minimum:
            raise ValueError(f"{what}: {k.size} level(s) after selection / filtering.")
        if spectrum is not None:
            source = "given spectrum"
        else:
            source = {None: "desymmetrised spectrum", "all": "all levels"}.get(
                sector, f"sector {sector}")
        log(f"[{what}] {k.size} levels ({source}), L_sel = {self.weyl_length(sector):.3f} µm, "
            f"expected ensemble {self.spectral_ensemble(sector)}.")
        return k

    def _default_wigner(self, sector):
        ens = self.spectral_ensemble(sector)
        return ["poissonian", ens.lower()] if ens else ["poissonian", "goe", "gue"]

    # =========================================================================
    # Staircase
    # =========================================================================

    def plot_staircase_function(self, plot_weyl=True, weyl_diff=False,
                                spectrum=None, sector=None, wavelength_range=None,
                                save_fig=False, file_format="pdf"):
        """
        Spectral staircase N(k) of the selected levels with the Weyl law
        N(k) = (L_sel/2π) ∫ n_g dk.

        Parameters
        ----------
        plot_weyl : bool
        weyl_diff : bool
            Also plot N_Weyl(k) − N(k) vs mode index and vs wavelength.
        spectrum, sector, wavelength_range
            Selection (see the module docstring).
        save_fig : bool
        file_format : str
        """
        k = self._plot_levels(spectrum, sector, wavelength_range, 1, "Staircase")
        tag = _file_tag(spectrum, sector, wavelength_range)
        n = k.size

        fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
        ax.step(np.insert(k, 0, k[0]), np.arange(n + 1), where="post",
                label=r"$N(k)$", color="k", lw=2)
        if plot_weyl:
            k_grid = np.linspace(k[0], k[-1], 1000)
            ax.plot(k_grid, self.weyl_staircase(k_grid, sector, k0=k[0]) + 0.5, "r--", lw=1.5,
                    label=r"$N_{\mathrm{Weyl}}(k)$")

        def k_to_lam(k_val):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 2 * np.pi / np.asarray(k_val, dtype=float)
        secax = ax.secondary_xaxis("top", functions=(k_to_lam, k_to_lam))
        secax.set_xlabel(r"Wavelength $\lambda$ (µm)", fontsize=22, labelpad=14)
        secax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4,
                                                  steps=[1, 2, 2.5, 5, 10]))
        secax.xaxis.set_minor_locator(AutoMinorLocator(2))
        secax.tick_params(axis="x", which="major", top=True, bottom=False,
                          direction="in", length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                          labelsize=18, labeltop=True)
        secax.tick_params(axis="x", which="minor", top=True, bottom=False,
                          direction="in", length=MINOR_TICK_LEN, width=MINOR_TICK_W)

        ax.set_title(fr"$L_{{\rm sel}} = {self.weyl_length(sector):.2f}\ \mu m$", fontsize=20)
        ax.set_xlabel(r"$k$ (µm$^{-1}$)", fontsize=22)
        ax.set_ylabel(r"$N(k)$", fontsize=22)
        ax.set_xlim(k[0], k[-1])
        ax.set_ylim(0, n)
        _tick_params(ax)
        if save_fig:
            _save(fig, self, f"staircase_function{tag}", file_format, "Staircase plot")
        show_figure()

        if not weyl_diff:
            return
        delta = self.weyl_staircase(k, sector, k0=k[0]) + 0.5 - np.arange(1, n + 1)
        for x, xlabel, xlim, name in (
                (np.arange(n), "Mode index", (0, n - 1), "index"),
                (2 * np.pi / k * 1e3, r"Wavelength $\lambda$ (nm)",
                 (2e3 * np.pi / k[-1], 2e3 * np.pi / k[0]), "lambda")):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(x, delta, "k-", lw=2)
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel(r"$N_{\mathrm{Weyl}}(k) - N(k)$", fontsize=22)
            ax.set_xlim(*xlim)
            _tick_params(ax)
            fig.tight_layout()
            if save_fig:
                _save(fig, self, f"staircase_diff_{name}{tag}", file_format,
                      f"Staircase difference ({name})")
            show_figure()

    # =========================================================================
    # Short-range correlations (NNSD)
    # =========================================================================

    def short_range_correlations(self, width=0.1, xlim=None, plot_wigner=None,
                                 cumulative_spacing=False, color_list=None,
                                 spectrum=None, sector=None, wavelength_range=None,
                                 yscale="linear", save_fig=False, file_format="pdf"):
        """
        Nearest-neighbour spacing distribution of the unfolded selection.

        Parameters
        ----------
        width : float            Histogram bin width.
        xlim : tuple or None     x-axis limits (the histogram is normalised on it).
        plot_wigner : list       Curves among 'poissonian', 'goe', 'gue', 'gse'
                                 (default: Poisson and the expected ensemble).
        cumulative_spacing : bool  Add the cumulative distribution below.
        color_list : list or None  Matplotlib style strings for the curves.
        spectrum, sector, wavelength_range
            Selection (see the module docstring).
        yscale : str             'linear' or 'log'.
        save_fig : bool
        file_format : str

        Returns
        -------
        spacings : ndarray
        """
        plot_wigner = self._default_wigner(sector) if plot_wigner is None else plot_wigner
        color_list = color_list or DEFAULT_STYLES
        k = self._plot_levels(spectrum, sector, wavelength_range, 2, "NNSD")
        spacings = np.diff(self.unfolded_spectrum(k, sector))

        if xlim is None:
            xlim = (0.0, float(np.max(spacings)))
        bins = np.arange(xlim[0], xlim[1] + width, width)
        if bins.size < 2:
            bins = np.array([xlim[0], xlim[1]])

        if cumulative_spacing:
            fig, (ax, ax_cdf) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [2, 1], "hspace": 0})
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        hist_values, bin_edges = np.histogram(spacings, bins=bins, density=True)
        ax.step(bin_edges[1:], hist_values, where="pre", color="black", linewidth=2)
        if hist_values.size > 0:
            ax.plot([bins[0], min(bins[0] + width, xlim[1])], [hist_values[0]] * 2,
                    "-", lw=2, color="black")

        s_vals = np.linspace(xlim[0], xlim[1], 1000)
        for i, name in enumerate(plot_wigner):
            ax.plot(s_vals, nnsd(s_vals, name), color_list[i % len(color_list)], lw=2,
                    label=WIGNER_LABELS.get(name.lower(), name.upper()))

        ax.set_ylabel(r"$P(s)$", fontsize=22)
        ax.set_xlabel("Spacing $s$", fontsize=22)
        ax.set_xlim(xlim)
        ymax = float(np.max(hist_values)) if hist_values.size else 1.0
        if plot_wigner:
            ymax = max(ymax, max(float(np.max(nnsd(s_vals, n))) for n in plot_wigner))
        ax.set_yscale(yscale)
        if yscale == "log":
            positive = hist_values[hist_values > 0]
            ymin = max(1e-4, float(np.min(positive)) * 0.5) if positive.size else 1e-4
            ax.set_ylim(ymin, max(ymax * 1.8, 1e-9))
        else:
            ax.set_ylim(-0.04 * ymax if cumulative_spacing else 0.0, max(ymax * 1.05, 1e-9))
        ax.legend(fontsize=16, loc="upper right")
        _tick_params(ax)

        if cumulative_spacing:
            s_sorted = np.sort(spacings)
            ax_cdf.scatter(s_sorted, np.arange(1, s_sorted.size + 1) / s_sorted.size,
                           marker="s", edgecolor="black", facecolor="none",
                           s=30, linewidths=2, zorder=2)
            ds = s_vals[1] - s_vals[0]
            for i, name in enumerate(plot_wigner):
                ax_cdf.plot(s_vals, np.cumsum(nnsd(s_vals, name)) * ds,
                            color_list[i % len(color_list)], lw=2,
                            label=WIGNER_LABELS.get(name.lower(), name.upper()))
            ax_cdf.set_ylabel(r"Cumulative $P(s)$", fontsize=22)
            ax_cdf.set_xlabel("Spacing $s$", fontsize=22)
            ax_cdf.set_xlim(xlim)
            ax_cdf.set_ylim(0, 1.1)
            ax_cdf.legend(fontsize=14, loc="lower right")
            _tick_params(ax_cdf)

        fig.tight_layout()
        if save_fig:
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            cumul_part = "_cumulative" if cumulative_spacing else ""
            yscale_part = "_log" if yscale == "log" else ""
            _save(fig, self, f"NNSD_{wigner_part}_bin{width:.2f}{cumul_part}{yscale_part}"
                  f"{_file_tag(spectrum, sector, wavelength_range)}", file_format, "NNSD plot")
        show_figure()
        return spacings

    # =========================================================================
    # Long-range correlations (Σ² and Δ₃)
    # =========================================================================

    def long_range_correlations(self, L_max=10, L_step=0.2, plot_wigner=None,
                                correlation_type="both", color_list=None,
                                spectrum=None, sector=None, wavelength_range=None,
                                save_fig=False, file_format="pdf"):
        """
        Number variance Σ²(L) and / or spectral rigidity Δ₃(L) of the unfolded
        selection, with random-matrix curves.

        Parameters
        ----------
        L_max, L_step : float
        plot_wigner : list       Default: Poisson and the expected ensemble.
        correlation_type : {'number_variance', 'spectral_rigidity', 'both'}
        color_list : list or None
        spectrum, sector, wavelength_range
            Selection (see the module docstring).
        save_fig : bool
        file_format : str

        Returns
        -------
        dict with the computed ``sigma2`` and / or ``delta3`` as (L, values).
        """
        if correlation_type not in ("number_variance", "spectral_rigidity", "both"):
            raise ValueError("correlation_type must be 'number_variance', "
                             "'spectral_rigidity' or 'both'.")
        plot_wigner = self._default_wigner(sector) if plot_wigner is None else plot_wigner
        color_list = color_list or DEFAULT_STYLES
        k = self._plot_levels(spectrum, sector, wavelength_range, 2, "LongRange")
        x = self.unfolded_spectrum(k, sector)

        panels = []
        if correlation_type in ("number_variance", "both"):
            panels.append(("sigma2", number_variance(x, L_step, L_max), sigma2_rmt,
                           r"$\Sigma^2(L)$"))
        if correlation_type in ("spectral_rigidity", "both"):
            panels.append(("delta3", spectral_rigidity(x, L_step, L_max), delta3_rmt,
                           r"$\Delta_3(L)$"))

        if len(panels) == 2:
            fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1], "hspace": 0})
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            axes = [ax]

        L_plot = np.linspace(0.0, L_max, 1000)
        for ax, (_, (L_vals, values), rmt, ylabel) in zip(axes, panels):
            ax.scatter(L_vals, values, edgecolor="black", facecolor="white",
                       s=60, linewidth=1.5, zorder=1)
            ymax = max([float(np.max(values))] +
                       [float(np.max(rmt(L_plot, n))) for n in plot_wigner
                        if n.lower() != "poissonian"]) * 1.05
            for i, name in enumerate(plot_wigner):
                ax.plot(L_plot, rmt(L_plot, name), color_list[i % len(color_list)], lw=2,
                        label=WIGNER_LABELS.get(name.lower(), name.upper()))
            ax.set_ylabel(ylabel, fontsize=22)
            ax.set_xlim(0.0, L_max)
            ax.set_ylim(0.0, max(ymax, 0.1))
            ax.legend(fontsize=18, loc="lower right")
            _tick_params(ax)
        axes[-1].set_xlabel("Interval length $L$", fontsize=22)

        fig.tight_layout()
        if save_fig:
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            _save(fig, self, f"LongRange_{correlation_type}_{wigner_part}_Lmax{L_max}"
                  f"_Lstep{L_step}{_file_tag(spectrum, sector, wavelength_range)}",
                  file_format, "Long-range correlation plot")
        show_figure()
        return {name: data for name, data, _, _ in panels}

    # =========================================================================
    # Length spectrum
    # =========================================================================

    def plot_length_spectrum(self, L_max=5000, n_points=100_000, spectrum=None, sector="all",
                             ignore_zero_peak=True, zero_window_um=None,
                             save_fig=False, file_format="pdf"):
        """
        Length spectrum ρ̂(ℓ) = |Σ_j exp(−i k_j n_eff(Re k_j) ℓ)| (periodic-orbit
        lengths ℓ = n_eff × geometric length).

        Parameters
        ----------
        L_max : float
        n_points : int
        spectrum : array or None
            Levels (real or complex k, e.g. measured peaks); default: the
            computed levels of ``sector``.
        sector : None, int or "all"
            Default "all": every level with its multiplicity.
        ignore_zero_peak : bool
            Exclude the ℓ ≈ 0 peak from the y-axis limit.
        zero_window_um : float or None
            Width of that exclusion (default max(100, 50 Δℓ)).
        save_fig : bool
        file_format : str

        Returns
        -------
        ell, rho_hat : ndarray
        """
        if spectrum is None:
            k_vals = np.asarray(self._selected_levels(sector), dtype=np.complex128)
        else:
            k_vals = np.asarray(spectrum, dtype=np.complex128).ravel()
        if k_vals.size == 0:
            raise ValueError("Length spectrum: no level selected.")

        ell = np.linspace(0.0, float(L_max), int(n_points))
        d_ell = float(ell[1] - ell[0]) if ell.size >= 2 else float(L_max)
        n_eff = index_function(self.neff, np.real(k_vals))
        rho_hat = np.zeros(ell.size, dtype=np.complex128)
        for kj, nj in zip(k_vals, n_eff):
            rho_hat += np.exp(-1j * kj * nj * ell)
        rho_hat = np.abs(rho_hat)

        if ignore_zero_peak:
            if zero_window_um is None:
                zero_window_um = max(100.0, 50 * d_ell)
            mask = ell > float(zero_window_um)
            ymax = float(np.max(rho_hat[mask])) if np.any(mask) else float(np.max(rho_hat[1:]))
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
            tag = _file_tag(spectrum, None if sector == "all" else sector, None)
            _save(fig, self, f"LengthSpectrum_Lmax{L_max}_N{n_points}{tag}", file_format,
                  "Length spectrum")
        show_figure()
        return ell, rho_hat
