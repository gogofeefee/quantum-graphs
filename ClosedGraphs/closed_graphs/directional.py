#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectionalGraphs — closed quantum graph made of 4-port directional couplers.

Inherits all ClosedGraphs functionality via the same mixin chain.
Scattering matrices are built automatically from coupling constants
(constant float or callable C(k)).

The ``plot_graph`` method uses disk markers ("pastilles") for port indicators,
matching the open_graphs convention (blue for ports 0,1; orange for ports 2,3).
"""

import numpy as np

from .core           import CoreMixin
from .spectrum       import SpectrumMixin
from .spectral_stats import SpectralStatsMixin
from .plotting.graph_plot        import GraphPlotMixin
from .plotting.spectral_plots    import SpectralPlotsMixin
from .plotting.wavefunction_plots import WavefunctionPlotsMixin


class DirectionalGraphs(CoreMixin, SpectrumMixin, SpectralStatsMixin,
                        GraphPlotMixin, SpectralPlotsMixin, WavefunctionPlotsMixin):
    """
    Closed quantum graph where every vertex is a lossless 4-port directional
    coupler.

    Parameters
    ----------
    vertex_positions : list of (float, float)
    port_map : list of lists
    lengths : dict or None
    vertex_couplings : dict, list, float, or callable
        Coupling constants.  float/callable → same for all vertices.
    neff : float or callable
    ng : float, callable, or None
    wavelength_range : (float, float)
    virtual_vertex_spacing : float, sequence, or None
    name : str
    save_path : str, Path, or None
    base_dir : str, Path, or None
    """

    def __init__(self,
                 vertex_positions,
                 port_map,
                 lengths=None,
                 vertex_couplings=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 name="MyClosedDirectionalGraph",
                 save_path=None,
                 base_dir=None):

        self.vertex_couplings = vertex_couplings

        vertex_matrices   = self._build_vertex_matrices(vertex_positions, port_map)
        self.coupling_info = self._get_coupling_description()

        super().__init__(
            vertex_positions       = vertex_positions,
            port_map               = port_map,
            lengths                = lengths,
            vertex_matrices        = vertex_matrices,
            neff                   = neff,
            ng                     = ng,
            wavelength_range       = wavelength_range,
            virtual_vertex_spacing = virtual_vertex_spacing,
            name                   = name,
            save_path              = save_path,
            base_dir               = base_dir,
        )

    # =========================================================================
    # Coupling helpers
    # =========================================================================

    def _build_vertex_matrices(self, vertex_positions, port_map):
        n         = len(vertex_positions)
        valencies = [len(ports) for ports in port_map]
        matrices  = {}

        for v in range(n):
            val = valencies[v]
            if val != 4:
                raise ValueError(
                    f"Only 4-port directional couplers supported; "
                    f"vertex {v} has valency {val}."
                )
            if isinstance(self.vertex_couplings, (dict, list)):
                if v not in self.vertex_couplings:
                    raise ValueError(f"Missing coupling constant for vertex {v}.")
                C = self.vertex_couplings[v]
            else:
                C = self.vertex_couplings

            matrices[v] = self._directional_coupler_matrix(C)
        return matrices

    def _directional_coupler_matrix(self, C):
        """
        4×4 directional coupler scattering matrix.

        σ = [[0,    0,    √(1-C),  i√C ],
             [0,    0,    i√C,   √(1-C)],
             [√(1-C), i√C, 0,    0    ],
             [i√C, √(1-C), 0,    0    ]]
        """
        def matrix_fn(k=None):
            c_val = C(k) if callable(C) else C
            sC    = np.sqrt(c_val)
            s1C   = np.sqrt(1 - c_val)
            return np.array([
                [0,      0,      s1C,     1j * sC],
                [0,      0,      1j * sC, s1C    ],
                [s1C,  1j * sC,  0,       0      ],
                [1j * sC, s1C,   0,       0      ],
            ], dtype=complex)

        return matrix_fn if callable(C) else matrix_fn()

    def _get_coupling_description(self):
        if isinstance(self.vertex_couplings, (dict, list)):
            return "custom_vertex_couplings"
        C = self.vertex_couplings
        if callable(C):
            return getattr(C, "__name__", "Cfunc")
        return f"C={C:.2f}"

    # =========================================================================
    # Plot graph (disk-marker style, matching open_graphs)
    # =========================================================================

    def plot_graph(
        self,
        show_lengths: bool = False,
        show_port_markers: bool = False,
        vertex_radius_frac: float = 0.12,
        marker_radius_frac: float = 0.06,
        marker_offset_frac: float = 0.00,
        save_fig: bool = False,
        file_format: str = "pdf",
    ):
        """
        Plot the graph with labeled nodes, optional edge lengths, and optional
        colored disk port markers on 4-port vertices.

        Port coloring:
          - BLUE   (#1f77b4) for ports (0, 1)
          - ORANGE (#ff7f0e) for ports (2, 3)

        Parameters
        ----------
        show_lengths : bool
        show_port_markers : bool
        vertex_radius_frac : float
            Vertex disk radius as a fraction of Lmin (data coords).
        marker_radius_frac : float
            Port marker disk radius as a fraction of Lmin.
        marker_offset_frac : float
            Offset from vertex border along bond direction (0 → on border).
        save_fig : bool
        file_format : str
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import networkx as nx

        BLUE       = "#1f77b4"
        ORANGE     = "#ff7f0e"
        EDGE_COLOR = "black"

        if not self.virtual_lengths:
            raise AttributeError("virtual_lengths is empty.")
        Lmin = float(min(self.virtual_lengths.values()))
        Rv   = float(vertex_radius_frac) * Lmin
        Rm   = float(marker_radius_frac) * Lmin
        delta = float(marker_offset_frac) * Lmin

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.get_node_attributes(self.G, "pos")

        real_nodes    = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = list(self.virtual_vertices)

        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=1000, node_color="black", ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400,  node_color="black", ax=ax)

        node_labels = {i: f"{i + 1}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color="white", font_weight="bold",
                                font_size=20, ax=ax)

        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color=EDGE_COLOR)

        if show_lengths:
            edge_labels = {k: f"{v:.1f}"
                           for k, v in nx.get_edge_attributes(self.G, "length").items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=20, ax=ax)

        # --- Disk marker helpers ---
        def draw_marker_disk(center_xy, radius, color, z=10_000):
            x, y = float(center_xy[0]), float(center_xy[1])
            circ = plt.Circle(
                (x, y), radius=float(radius),
                facecolor=color, edgecolor="black",
                linewidth=0.8, zorder=z, clip_on=False,
            )
            ax.add_patch(circ)

        def marker_center_on_border(vertex_xy, neighbor_xy):
            v  = np.asarray(vertex_xy,  dtype=float)
            n  = np.asarray(neighbor_xy, dtype=float)
            d  = n - v
            Ls = float(np.hypot(d[0], d[1]))
            if Ls == 0.0:
                return None
            u     = d / Ls
            s_des = Rv + delta
            margin = 1.1 * Rm
            s_min  = max(0.0, Rv - 5.0 * Rm)
            s_max  = max(Rv, Ls - margin)
            s      = min(max(s_des, s_min), s_max)
            return v + s * u

        if show_port_markers:
            for v in self.G.nodes:
                if v in self.virtual_vertices:
                    continue
                if hasattr(self, "valencies") and self.valencies[v] != 4:
                    continue
                vxy = pos[v]
                for p in range(4):
                    conn = self.virtual_port_map[v][p]
                    if conn is None:
                        continue
                    n, _ = conn
                    nxy  = pos[n]
                    center = marker_center_on_border(vxy, nxy)
                    if center is None:
                        continue
                    color = BLUE if p in (0, 1) else ORANGE
                    draw_marker_disk(center, radius=Rm, color=color)

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=1.0)

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            tags = ["with_lengths" if show_lengths else "no_lengths"]
            if show_port_markers:
                tags.append("port_markers")
            filename  = f"{self.name}_graph_{'_'.join(tags)}.{file_format}"
            filepath  = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            print(f"Graph plot saved to {filepath}")

        plt.show()
