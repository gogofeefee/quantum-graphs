#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directional.py — DirectionalClassicalGraphs.

Specialisation for 4-port directional-coupler graphs. Overrides plot_graph
to add colored disk port markers (blue for ports 0/1, orange for ports 2/3),
matching the ClosedGraphs / OpenGraphs convention.
"""

from ..core.config import log
import os
import numpy as np

from ..core.vertices import couplings_for_vertices, directional_coupler_matrix
import matplotlib.pyplot as plt
import networkx as nx

from .core      import CoreMixin
from .dynamics  import DynamicsMixin
from .plotting  import GraphPlotMixin, RhoPlotMixin, AnimationMixin
from ._utils import _normalize, _perpendicular


class DirectionalClassicalGraphs(
    CoreMixin, DynamicsMixin,
    GraphPlotMixin, RhoPlotMixin, AnimationMixin,
):
    """
    Classical directional graph: inherits all dynamics and plotting from
    ClassicalGraphs and adds port-marker visualisation.

    Port-marker color convention
    ----------------------------
    - BLUE   (#1f77b4) for ports 0, 1
    - ORANGE (#ff7f0e) for ports 2, 3
    """

    def __init__(
        self,
        vertex_positions,
        port_map,
        lengths=None,
        vertex_matrices=None,
        virtual_vertex_spacing=None,
        name="MyDirectionalClassicalGraph",
        save_path=None,
        base_dir=None,
        vertex_couplings=None,
    ):
        """``vertex_couplings`` (float, list or dict of C) builds the directional-
        coupler matrices, as in the quantum DirectionalGraphs classes; it is an
        alternative to ``vertex_matrices``."""
        if vertex_couplings is not None:
            if vertex_matrices is not None:
                raise ValueError("Give vertex_couplings or vertex_matrices, not both.")
            couplings = couplings_for_vertices(vertex_couplings, len(port_map))
            vertex_matrices = {v: directional_coupler_matrix(C) for v, C in enumerate(couplings)}
        super().__init__(
            vertex_positions=vertex_positions,
            port_map=port_map,
            lengths=lengths,
            vertex_matrices=vertex_matrices,
            virtual_vertex_spacing=virtual_vertex_spacing,
            name=name,
            save_path=save_path,
            base_dir=base_dir,
        )

    def plot_graph(
        self,
        ax=None,
        show_lengths: bool = False,
        show_port_markers: bool = False,
        vertex_radius_frac: float = 0.12,
        marker_radius_frac: float = 0.04,
        marker_offset_frac: float = 0.02,
        save_fig: bool = False,
        file_format: str = "pdf",
        title=None,
    ):
        """
        Plot the directional graph with optional colored port-marker disks.

        Parameters
        ----------
        show_port_markers : bool
            If True, draw blue/orange disk markers on each port of 4-port vertices.
        vertex_radius_frac : float
            Real-vertex disk radius as a fraction of Lmin.
        marker_radius_frac : float
            Port-marker disk radius as a fraction of Lmin.
        marker_offset_frac : float
            Offset of marker center from vertex center, as a fraction of Lmin.

        Returns
        -------
        fig, ax
        """
        BLUE   = "#1f77b4"
        ORANGE = "#ff7f0e"

        if not hasattr(self, "virtual_lengths") or not self.virtual_lengths:
            raise AttributeError("self.virtual_lengths must be non-empty.")

        Lmin = float(min(self.virtual_lengths.values()))
        Rv   = float(vertex_radius_frac)  * Lmin
        Rm   = float(marker_radius_frac)  * Lmin
        delta = float(marker_offset_frac) * Lmin

        if Rv <= 0:
            raise ValueError("vertex_radius_frac must be > 0.")
        if Rm <= 0:
            raise ValueError("marker_radius_frac must be > 0.")

        # Draw base graph (calls GraphPlotMixin.plot_graph via MRO)
        fig, ax = super(DirectionalClassicalGraphs, self).plot_graph(
            ax=ax,
            show_lengths=show_lengths,
            vertex_radius_frac=vertex_radius_frac,
            save_fig=False,
            file_format=file_format,
            title=title,
        )

        pos = nx.get_node_attributes(self.G, "pos")

        def _draw_marker_disk(center_xy, radius, color):
            x, y = float(center_xy[0]), float(center_xy[1])
            circ = plt.Circle(
                (x, y), radius=float(radius),
                facecolor=color, edgecolor="black",
                linewidth=0.8, zorder=10_000, clip_on=False,
            )
            ax.add_patch(circ)

        def _marker_center(vertex_xy, neighbor_xy):
            v = np.asarray(vertex_xy, dtype=float)
            n = np.asarray(neighbor_xy, dtype=float)
            d = n - v
            L = float(np.hypot(d[0], d[1]))
            if L == 0.0:
                return None
            u = d / L
            s_des = Rv + delta
            s_min = max(0.0, Rv - 5.0 * Rm)
            s_max = max(Rv, L - 1.1 * Rm)
            s = min(max(s_des, s_min), s_max)
            return v + s * u

        if show_port_markers:
            if not hasattr(self, "virtual_port_map"):
                raise AttributeError("Needs self.virtual_port_map to place port markers.")

            for v in self.G.nodes:
                if v in self.virtual_vertices:
                    continue
                try:
                    if self.valencies[v] != 4:
                        continue
                except Exception:
                    pass

                vxy = pos[v]
                for p in range(4):
                    conn = self.virtual_port_map[v][p]
                    if conn is None:
                        continue
                    n, q = conn
                    nxy = pos[n]
                    center = _marker_center(vxy, nxy)
                    if center is None:
                        continue
                    color = BLUE if p in (0, 1) else ORANGE
                    _draw_marker_disk(center, radius=Rm, color=color)

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            tags = ["with_lengths" if show_lengths else "no_lengths"]
            if show_port_markers:
                tags.append("port_markers")
            fname = f"{self.name}_graph_{'_'.join(tags)}.{file_format}"
            fpath = os.path.join(self.save_path, fname)
            fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
            log(f"✓ Graph plot saved to {fpath}")

        return fig, ax
