#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPlotMixin — plot_graph (shared, see quantum_graphs.plotting), plot_zeta_B.

``plot_zeta_B`` has a secondary x-axis (λ ↔ frequency) and uses
``layout="constrained"``.
"""

from ...core.config import show_figure
import numpy as np
import matplotlib.pyplot as plt

from ...plotting.style import add_secondary_top_axis, box_ticks, save_figure
from ...plotting.topology_plot import TopologyPlotMixin


class GraphPlotMixin(TopologyPlotMixin):
    """plot_graph (shared) and plot_zeta_B of a closed graph."""

    # =========================================================================
    # Secular determinant (zeta_B)
    # =========================================================================

    def plot_zeta_B(self, wavelength_range=None, N_plot=100_000, spectrum=None,
                    save_fig=False, file_format="pdf", xlim=None,
                    ylim_real=None, ylim_imag=None, ylim_mod=None, show=None):
        """
        Real, imaginary, and modulus of the secular determinant f(k).

        Parameters
        ----------
        wavelength_range : (float, float) or None
        N_plot : int
        spectrum : array-like or None
            If provided, mark eigenvalues on each subplot.
        save_fig : bool
        file_format : str
        xlim : (float, float) or None   Zoom on λ axis.
        ylim_real, ylim_imag, ylim_mod : (float, float) or None
        show : bool or None

        Returns
        -------
        fig, axs
        """
        if wavelength_range is None:
            wavelength_range = self.wavelength_range
        lambda_min, lambda_max = wavelength_range

        lambda_vals = np.linspace(lambda_min, lambda_max, N_plot)
        k_vals      = 2 * np.pi / lambda_vals

        det_vals = np.array([self.secular_det(k) for k in k_vals])
        Re_det   = np.real(det_vals)
        Im_det   = np.imag(det_vals)
        Mod_det  = np.abs(det_vals)

        fig, axs = plt.subplots(3, 1, figsize=(14, 10),
                                sharex=True, layout="constrained")

        subplot_data   = [Re_det, Im_det, Mod_det]
        subplot_ylabels = [r"Re[$\zeta_B(\lambda)$]",
                           r"Im[$\zeta_B(\lambda)$]",
                           r"$|\zeta_B(\lambda)|$"]
        subplot_ylims  = [ylim_real, ylim_imag, ylim_mod]

        for ax, data, ylabel, ylim in zip(axs, subplot_data,
                                          subplot_ylabels, subplot_ylims):
            ax.plot(lambda_vals, data, "k-", lw=1.0)
            ax.set_ylabel(ylabel, fontsize=22)
            if ylim:
                ax.set_ylim(ylim)
            if xlim:
                ax.set_xlim(xlim)
            box_ticks(ax)

        axs[-1].set_xlabel(r"Wavelength $\lambda$ (µm)", fontsize=22)

        # Secondary x-axis on the top subplot
        add_secondary_top_axis(axs[0], "wavelength", set_label=True)

        if spectrum is not None:
            spectrum_lambda = 2 * np.pi / np.array(spectrum)
            for ax in axs:
                ax.plot(spectrum_lambda, np.zeros_like(spectrum_lambda),
                        "r*", markersize=10)

        if save_fig:
            fname = (f"zeta_B_plot_lambdamin_{lambda_min:.2f}_"
                     f"lambdamax_{lambda_max:.2f}_Nplot_{N_plot}"
                     f"{'_with_spectrum' if spectrum is not None else ''}.{file_format}")
            save_figure(fig, self.save_path, fname, "Zeta_B plot", bbox_inches=None)
        show_figure(fig, show)
        return fig, axs
