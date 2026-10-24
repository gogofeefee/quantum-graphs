#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPlotMixin — plot_graph (base ClosedGraphs), plot_zeta_B.

``plot_zeta_B`` has a secondary x-axis (λ ↔ frequency) and uses
``layout="constrained"``.
"""

from ...core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .style import (
    MAJOR_TICK_LEN, MINOR_TICK_LEN, MAJOR_TICK_W, MINOR_TICK_W,
    add_secondary_top_axis,
)


def _tick_params(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in",
                   length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=22, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                   top=True, right=True)


class GraphPlotMixin:

    # =========================================================================
    # Topology
    # =========================================================================

    def plot_graph(self, show_lengths=False, save_fig=False, file_format="pdf",
                   figsize=(10, 8), node_size=900, virtual_node_size=400,
                   font_size=20, edge_width=3, length_font_size=22,
                   filename_suffix=None):
        """
        Plot the graph topology: real nodes (large, labeled), virtual nodes
        (small), and edges.

        Parameters
        ----------
        show_lengths : bool
        save_fig : bool
        file_format : str
        figsize : (float, float)
        node_size : float
            Marker size (points^2) for real (labeled) vertices.
        virtual_node_size : float
            Marker size (points^2) for virtual (unlabeled) vertices.
        font_size : float
            Vertex number label size.
        edge_width : float
            Bond line width.
        length_font_size : float
            Edge-length label size (only used if show_lengths=True).
        filename_suffix : str or None
            Extra tag appended to the saved filename, e.g. for A/B testing
            different sizing combos without overwriting previous renders.
        """
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.get_node_attributes(self.G, "pos")

        real_nodes  = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virt_nodes  = self.virtual_vertices

        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=node_size, node_color="black", ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virt_nodes,
                               node_size=virtual_node_size, node_color="black", ax=ax)

        node_labels = {i: f"{i + 1}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color="white", font_weight="bold",
                                font_size=font_size, ax=ax)

        nx.draw_networkx_edges(self.G, pos, ax=ax, width=edge_width, edge_color="black")

        if show_lengths:
            edge_labels = {k: f"{v:.2f}"
                           for k, v in nx.get_edge_attributes(self.G, "length").items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=length_font_size, ax=ax)

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=1.0)

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            suffix   = "with_lengths" if show_lengths else "no_lengths"
            if filename_suffix:
                suffix = f"{suffix}_{filename_suffix}"
            filename = f"{self.name}_graph_{suffix}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            log(f"Graph plot saved to {filepath}")
        show_figure()

    # =========================================================================
    # Secular determinant (zeta_B)
    # =========================================================================

    def plot_zeta_B(self, wavelength_range=None, N_plot=100_000, spectrum=None,
                    save_fig=False, file_format="pdf", xlim=None,
                    ylim_real=None, ylim_imag=None, ylim_mod=None):
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
            _tick_params(ax)

        axs[-1].set_xlabel(r"Wavelength $\lambda$ (µm)", fontsize=22)

        # Secondary x-axis on the top subplot
        add_secondary_top_axis(axs[0], "wavelength", set_label=True)

        if spectrum is not None:
            spectrum_lambda = 2 * np.pi / np.array(spectrum)
            for ax in axs:
                ax.plot(spectrum_lambda, np.zeros_like(spectrum_lambda),
                        "r*", markersize=10)

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            fname = (f"zeta_B_plot_lambdamin_{lambda_min:.2f}_"
                     f"lambdamax_{lambda_max:.2f}_Nplot_{N_plot}"
                     f"{'_with_spectrum' if spectrum is not None else ''}.{file_format}")
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300)
            log(f"✓ Zeta_B plot saved to: {fpath}")
        show_figure()
