#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting/graph_plot.py — GraphPlotMixin for classical_graphs.

plot_graph: draws the graph topology (vertices, edges, optional lengths).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .._utils import _is_number


class GraphPlotMixin:
    """Base graph topology plot."""

    def plot_graph(
        self,
        ax=None,
        show_lengths: bool = False,
        vertex_radius_frac: float = 0.12,
        save_fig: bool = False,
        file_format: str = "pdf",
        title=None,
    ):
        """
        Plot the graph topology.

        - Real vertices: black disks with white 1-indexed labels.
        - Virtual vertices: smaller black disks (no label).
        - Edges: thick black lines.
        - Optional edge-length labels when show_lengths=True.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
        show_lengths : bool
        vertex_radius_frac : float
            Passed through to the directional child class for compatibility.
        save_fig : bool
        file_format : str
        title : str or None

        Returns
        -------
        fig, ax
        """
        if not hasattr(self, "virtual_lengths") or len(self.virtual_lengths) == 0:
            raise AttributeError(
                "self.virtual_lengths must be a non-empty dict."
            )

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            created_fig = True
        else:
            fig = ax.figure

        pos = nx.get_node_attributes(self.G, "pos")
        if not pos:
            raise AttributeError("Graph nodes must have 'pos' attribute.")

        real_nodes = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = list(self.virtual_vertices)

        nx.draw_networkx_nodes(
            self.G, pos, nodelist=real_nodes,
            node_size=1000, node_color="black", ax=ax,
        )
        nx.draw_networkx_nodes(
            self.G, pos, nodelist=virtual_nodes,
            node_size=400, node_color="black", ax=ax,
        )
        nx.draw_networkx_labels(
            self.G, pos,
            labels={i: f"{i + 1}" for i in real_nodes},
            font_color="white", font_weight="bold", font_size=20, ax=ax,
        )
        nx.draw_networkx_edges(
            self.G, pos, ax=ax, width=3, edge_color="black",
        )

        if show_lengths:
            edge_labels = {
                k: f"{v:.1f}"
                for k, v in nx.get_edge_attributes(self.G, "length").items()
            }
            nx.draw_networkx_edge_labels(
                self.G, pos, edge_labels=edge_labels, font_size=12, ax=ax,
            )

        if title is not None:
            ax.set_title(title)

        ax.set_aspect("equal")
        ax.axis("off")

        if created_fig:
            fig.tight_layout()

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            tags = ["with_lengths" if show_lengths else "no_lengths"]
            fname = f"{self.name}_graph_{'_'.join(tags)}.{file_format}"
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
            print(f"✓ Graph plot saved to {fpath}")

        return fig, ax
