#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectionalGraphs — open quantum graph where every vertex is a 4-port
directional coupler.  Inherits all OpenGraphs functionality; scattering
matrices are built automatically from coupling constants.
"""

from ..core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .core           import CoreMixin
from .scattering     import ScatteringMixin
from .resonances     import ResonancesMixin
from .experimental   import ExperimentalMixin
from .spectral_stats import SpectralStatsMixin
from .plotting.graph_plot        import GraphPlotMixin
from .plotting.s_matrix_plots    import SMatrixPlotsMixin
from .plotting.spectral_plots    import SpectralPlotsMixin
from .plotting.wavefunction_plots import WavefunctionPlotsMixin


class DirectionalGraphs(CoreMixin, ScatteringMixin, ResonancesMixin,
                        ExperimentalMixin, SpectralStatsMixin,
                        GraphPlotMixin, SMatrixPlotsMixin, SpectralPlotsMixin,
                        WavefunctionPlotsMixin):
    """
    Open quantum graph where every real vertex is a lossless 4-port
    directional coupler.

    Coupling constants determine the scattering matrices automatically.
    Optionally, per-vertex couplings can be specified via ``vertex_couplings``.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of each vertex (µm).
    port_map : list of lists
        Connectivity.  ``port_map[i][p] = (j, q)`` or ``None`` for a lead.
    lengths : dict or None
        Bond lengths ``{((i,p),(j,q)): L_um}``.  ``None`` → Euclidean.
    vertex_couplings : dict, optional
        Per-vertex coupling constants ``{v: C}`` (float or callable).
        When provided, overrides ``internal_coupling`` / ``external_coupling``
        for the specified vertices.
    internal_coupling : float, callable, or None
        Default coupling for internal (non-lead) vertices.
    external_coupling : float, callable, or None
        Default coupling for external (lead-connected) vertices.
    neff : float or callable
        Effective refractive index.
    ng : float, callable, or None
        Group index.  Derived from ``neff`` if ``None``.
    wavelength_range : (float, float)
        Vacuum wavelength range (µm).
    virtual_vertex_spacing : float, sequence, or None
        Spacing when splitting multiple bonds / self-loops.
    lead_orientations : dict or None
        ``{(vertex, port): angle_deg}`` for plotting.
    name : str
        Graph label (used in save paths and plot titles).
    save_path : str, Path, or None
        Override the default save directory.
    """

    def __init__(self,
                 vertex_positions,
                 port_map,
                 lengths=None,
                 vertex_couplings=None,
                 internal_coupling=None,
                 external_coupling=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 lead_orientations=None,
                 name="MyOpenDirectionalGraph",
                 save_path=None,
                 base_dir=None):

        self.vertex_couplings   = vertex_couplings
        self.internal_coupling  = internal_coupling
        self.external_coupling  = external_coupling

        vertex_matrices    = self._build_vertex_matrices(vertex_positions, port_map)
        self.coupling_info = self._get_coupling_description()

        super().__init__(
            vertex_positions=vertex_positions,
            port_map=port_map,
            lengths=lengths,
            vertex_matrices=vertex_matrices,
            internal_matrix=None,
            external_matrix=None,
            neff=neff,
            ng=ng,
            wavelength_range=wavelength_range,
            virtual_vertex_spacing=virtual_vertex_spacing,
            lead_orientations=lead_orientations,
            name=name,
            save_path=save_path,
            base_dir=base_dir,
        )

    # -------------------------------------------------------------------------
    # Coupling helpers
    # -------------------------------------------------------------------------

    def _build_vertex_matrices(self, vertex_positions, port_map):
        """Build per-vertex scattering matrices from coupling constants."""
        n         = len(vertex_positions)
        valencies = [len(ports) for ports in port_map]
        matrices  = {}

        for v in range(n):
            if valencies[v] != 4:
                raise ValueError(
                    f"DirectionalGraphs requires 4-port vertices, "
                    f"but vertex {v} has valency {valencies[v]}."
                )

            if self.vertex_couplings is not None and v in self.vertex_couplings:
                C = self.vertex_couplings[v]
            else:
                is_external = any(conn is None for conn in port_map[v])
                C = self.external_coupling if is_external else self.internal_coupling
                if C is None:
                    kind = "external" if is_external else "internal"
                    raise ValueError(
                        f"No coupling constant for vertex {v} "
                        f"({kind}). Provide ``{'external' if is_external else 'internal'}"
                        f"_coupling`` or ``vertex_couplings``."
                    )

            matrices[v] = self._directional_coupler_matrix(C)

        return matrices

    @staticmethod
    def _directional_coupler_matrix(C):
        """
        Return a 4×4 unitary directional-coupler scattering matrix.

        .. code-block::

            σ = [[  0,    0,   √(1-C),  i√C  ],
                 [  0,    0,   i√C,   √(1-C) ],
                 [√(1-C), i√C,  0,      0    ],
                 [i√C,  √(1-C), 0,      0    ]]

        ``C`` may be a float or a callable ``C(k)``.
        """
        def _matrix(k=None):
            c_val = C(k) if callable(C) else C
            sC    = np.sqrt(c_val)
            s1C   = np.sqrt(1.0 - c_val)
            return np.array([
                [0,    0,    s1C,        1j * sC],
                [0,    0,    1j * sC,    s1C    ],
                [s1C,  1j * sC, 0,       0      ],
                [1j * sC, s1C,  0,       0      ],
            ], dtype=complex)

        return _matrix if callable(C) else _matrix()

    def _get_coupling_description(self):
        """Build a compact coupling string for the deterministic save-path."""
        def _str(C):
            if C is None:
                return None
            if callable(C):
                return getattr(C, "__name__", "Cfunc")
            return f"{C:.2f}"

        if self.vertex_couplings is not None:
            return "custom_vertex_couplings"

        parts = []
        cin  = _str(self.internal_coupling)
        cext = _str(self.external_coupling)
        if cin  is not None:
            parts.append(f"Cin={cin}")
        if cext is not None:
            parts.append(f"Cext={cext}")
        return "_".join(parts) if parts else "no_coupling"

    # -------------------------------------------------------------------------
    # Ring extraction (directional-graph specific)
    # -------------------------------------------------------------------------

    def get_ring(self,
                 start=(0, 3),
                 end=(0, 1),
                 max_steps=None,
                 strict=True):
        """
        Trace the deterministic ring in the C=1 limit.

        Starting from port ``start=(v, p)``, propagates along that port to
        the next vertex, applies the paired-port scattering rule, and
        continues until ``end=(v, p)`` is reached.

        Parameters
        ----------
        start, end : (int, int)
            ``(vertex, port)`` states.
        max_steps : int or None
            Safety limit on iteration count.
        strict : bool
            Raise on cycles or lead hits; otherwise return partial path.

        Returns
        -------
        ring : list of (int, int)
        """
        if not hasattr(self, "virtual_port_map"):
            raise AttributeError("virtual_port_map missing — did __init__ complete?")

        nV      = len(self.vertex_positions)
        v0, p0  = start
        ve, pe  = end

        if not (0 <= v0 < nV and 0 <= ve < nV):
            raise ValueError("start/end vertex index out of range.")

        if max_steps is None:
            max_steps = 10 * sum(len(ports) for ports in self.virtual_port_map)

        def _paired(v, p):
            val  = len(self.virtual_port_map[v])
            if val == 4:
                pair = {0: 3, 3: 0, 1: 2, 2: 1}
                if p not in pair:
                    raise ValueError(f"Invalid port {p} for 4-port vertex {v}.")
                return pair[p]
            if val == 2:
                if p not in (0, 1):
                    raise ValueError(f"Invalid port {p} for 2-port vertex {v}.")
                return 1 - p
            raise ValueError(f"Vertex {v} has valency {val}; only 2 and 4 supported.")

        ring    = [(v0, p0)]
        visited = {(v0, p0)}

        if (v0, p0) == (ve, pe):
            return ring

        cur_v, cur_p = v0, p0

        for _ in range(max_steps):
            conn = self.virtual_port_map[cur_v][cur_p]
            if conn is None:
                if strict:
                    raise RuntimeError(
                        f"Ring trajectory hit a lead from {(cur_v, cur_p)}.")
                return ring

            nxt_v, nxt_p = conn
            ring.append((nxt_v, nxt_p))
            if (nxt_v, nxt_p) == (ve, pe):
                return ring

            out_p = _paired(nxt_v, nxt_p)
            state = (nxt_v, out_p)

            if state in visited:
                if strict:
                    raise RuntimeError(
                        f"Cycle detected at {state} before reaching end={end}.")
                return ring

            visited.add(state)
            cur_v, cur_p = state
            ring.append(state)

        if strict:
            raise RuntimeError(
                f"max_steps={max_steps} exceeded before reaching end={end}.")
        return ring

    def path_length(self, path):
        """
        Total physical length (µm) of a port-level path on the virtual graph.

        Same path format as returned by ``get_ring()``.  Steps that stay on
        the same vertex (scattering events) contribute zero length.

        Parameters
        ----------
        path : list of (int, int)

        Returns
        -------
        float
        """
        if path is None or len(path) < 2:
            return 0.0

        Ltot = 0.0
        for (v, p), (vn, pn) in zip(path[:-1], path[1:]):
            if vn == v:
                continue

            a   = (v, p)
            b   = (vn, pn)
            key = tuple(sorted([a, b]))

            if hasattr(self, "virtual_lengths") and key in self.virtual_lengths:
                Ltot += float(self.virtual_lengths[key])
                continue

            if hasattr(self, "directed_lengths"):
                if (a, b) in self.directed_lengths:
                    Ltot += float(self.directed_lengths[(a, b)])
                    continue
                if (b, a) in self.directed_lengths:
                    Ltot += float(self.directed_lengths[(b, a)])
                    continue

            # Last resort: infer from virtual_port_map
            if hasattr(self, "virtual_port_map"):
                conn = self.virtual_port_map[v][p]
                if conn is not None and conn[0] == vn:
                    b2   = (vn, conn[1])
                    key2 = tuple(sorted([a, b2]))
                    if hasattr(self, "virtual_lengths") and key2 in self.virtual_lengths:
                        Ltot += float(self.virtual_lengths[key2])
                        continue
                    if hasattr(self, "directed_lengths") and (a, b2) in self.directed_lengths:
                        Ltot += float(self.directed_lengths[(a, b2)])
                        continue

            raise KeyError(
                f"Cannot find length for step {(v,p)} → {(vn,pn)}."
            )

        return Ltot

    # -------------------------------------------------------------------------
    # Directional plot_graph  (overrides GraphPlotMixin.plot_graph)
    # -------------------------------------------------------------------------

    def plot_graph(self,
                   show_lengths=False,
                   show_port_markers=True,
                   vertex_radius_frac=0.12,
                   marker_radius_frac=0.06,
                   marker_offset_frac=0.00,
                   save_fig=False,
                   file_format="pdf"):
        """
        Plot the directional open quantum graph with colour-coded port markers.

        Colored circular disks ("pastilles") are drawn at the vertex border
        for each port:

        * **Blue**   (``#1f77b4``) → ports 0 and 1.
        * **Orange** (``#ff7f0e``) → ports 2 and 3.

        Leads are drawn as dashed black stubs and also receive disk markers.

        Parameters
        ----------
        show_lengths : bool
            Annotate edges with bond lengths (µm).
        show_port_markers : bool
            Draw the blue/orange port disks.
        vertex_radius_frac : float
            Vertex disk radius in data coordinates as a fraction of the
            shortest bond length.  Used to position markers at the node
            border.  Tune until markers sit visually on the node perimeter.
        marker_radius_frac : float
            Port marker (disk) radius as a fraction of the shortest bond length.
        marker_offset_frac : float
            Extra offset from the vertex border along the bond direction,
            as a fraction of the shortest bond length.  0 → marker centre
            sits exactly on the border; positive → moves outward.
        save_fig : bool
            Save the figure to ``self.save_path``.
        file_format : str
            Output format (``'pdf'``, ``'svg'``, ``'png'``, …).
        """
        BLUE       = "#1f77b4"   # ports 0, 1
        ORANGE     = "#ff7f0e"   # ports 2, 3
        EDGE_COLOR = "black"

        if not hasattr(self, "virtual_lengths") or not self.virtual_lengths:
            raise AttributeError("self.virtual_lengths must be a non-empty dict.")
        Lmin  = float(min(self.virtual_lengths.values()))
        Rv    = float(vertex_radius_frac) * Lmin
        Rm    = float(marker_radius_frac) * Lmin
        delta = float(marker_offset_frac) * Lmin

        fig, ax = plt.subplots(figsize=(10, 8))
        pos           = nx.get_node_attributes(self.G, "pos")
        real_nodes    = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = list(self.virtual_vertices)

        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=1000, node_color="black", ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color="black", ax=ax)
        node_labels = {i: self.vertex_labels[i] for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color="white", font_weight="bold",
                                font_size=18, ax=ax)
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color=EDGE_COLOR)

        if show_lengths:
            elabels = {k: f"{v:.1f}"
                       for k, v in nx.get_edge_attributes(self.G, "length").items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=elabels,
                                         font_size=18, ax=ax)

        def _draw_disk(center_xy, color, z=10_000):
            x, y = float(center_xy[0]), float(center_xy[1])
            circ = plt.Circle(
                (x, y), radius=Rm,
                facecolor=color, edgecolor="black",
                linewidth=0.8, zorder=z, clip_on=False,
            )
            ax.add_patch(circ)

        def _marker_center(vertex_xy, neighbor_xy):
            """Place marker so its inner edge is tangent to the vertex border.

            The disk center sits at Rv + Rm + delta from the vertex center,
            so the closest point of the disk is exactly at Rv (on the border).
            """
            v = np.asarray(vertex_xy, dtype=float)
            n = np.asarray(neighbor_xy, dtype=float)
            d = n - v
            Lseg = float(np.hypot(d[0], d[1]))
            if Lseg == 0.0:
                return None
            u     = d / Lseg
            s_des = Rv + Rm + delta
            s_min = Rv
            s_max = max(Rv + Rm, Lseg - 1.1 * Rm)
            s     = min(max(s_des, s_min), s_max)
            return v + s * u

        # Internal port markers
        if show_port_markers:
            for v in self.G.nodes:
                if v in self.virtual_vertices:
                    continue
                vxy = pos[v]
                for p in range(4):
                    conn = self.virtual_port_map[v][p]
                    if conn is None:
                        continue
                    n_nb, _ = conn
                    center = _marker_center(vxy, pos[n_nb])
                    if center is None:
                        continue
                    color = BLUE if p in (0, 1) else ORANGE
                    _draw_disk(center, color)

        # Lead stubs and markers
        lead_length = 80.0
        for v in self.G.nodes:
            if v in self.virtual_vertices:
                continue
            xv, yv = pos[v]
            origin = np.array([xv, yv])
            for p in range(4):
                if self.virtual_port_map[v][p] is not None:
                    continue
                if (not hasattr(self, "lead_orientations")
                        or (v, p) not in self.lead_orientations):
                    continue
                theta = np.deg2rad(self.lead_orientations[(v, p)])
                u     = np.array([np.cos(theta), np.sin(theta)])
                tip   = origin + lead_length * u
                ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                        linestyle="--", linewidth=2, color=EDGE_COLOR, zorder=2)
                if show_port_markers:
                    s_des   = Rv + Rm + delta
                    s_max   = max(Rv + Rm, lead_length - 1.1 * Rm)
                    s       = min(max(s_des, Rv), s_max)
                    center  = origin + s * u
                    color   = BLUE if p in (0, 1) else ORANGE
                    _draw_disk(center, color)

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=1.0)

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            tags = ["with_lengths" if show_lengths else "no_lengths"]
            if show_port_markers:
                tags.append("port_markers")
            filename = f"{self.name}_graph_{'_'.join(tags)}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            log(f"Graph plot saved to {filepath}")

        show_figure()
