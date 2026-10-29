"""
Drawing of a graph: bond paths and the elements shared by every figure.

:class:`BondPaths` gives each bond a curve in the plane, from its start vertex
(``i`` of ``(i, p, j, q, L)``) to its end vertex:

* a single bond between two vertices: a straight segment;
* parallel bonds between the same two vertices: quadratic Bézier arcs bowed
  symmetrically on both sides of the segment (an odd number keeps one straight);
* a self-loop: a teardrop (cubic Bézier) whose drawn length equals the bond
  length, pointing away from the other bonds of the vertex.

Curves are parametrised by their arc-length fraction ``s`` in [0, 1], so a
quantity known along the bond (wavefunction, classical density) is drawn at
the right place even on a curved bond. The geometry is for drawing only; the
physics uses the bond lengths.

The functions ``draw_*`` add the usual elements to a matplotlib axes, in data
units (µm); sizes are fractions of the shortest bond length, so figures look
the same at any graph scale. :func:`draw_graph` combines them.
"""

import numpy as np
from matplotlib.collections import LineCollection

PORT_COLORS = ("#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e")   # coupler ports (0, 1) | (2, 3)
_N_CURVE = 400


def _bezier(points, t):
    """Bézier curve of control ``points`` (shape (m, 2)) at parameters ``t``."""
    pts = np.asarray(points, dtype=float)
    m = len(pts) - 1
    t = np.asarray(t, dtype=float)[:, None]
    from math import comb
    return sum(comb(m, r) * (1 - t) ** (m - r) * t ** r * pts[r] for r in range(m + 1))


def _rotate(u, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


class BondPaths:
    """Drawing curve of every bond of a :class:`~quantum_graphs.core.topology.Topology`.

    Parameters
    ----------
    topology : Topology
    bond_spacing : float, sequence or None
        Distance (µm) between neighbouring parallel arcs at their middle. A
        sequence is used cyclically, one value per group of parallel bonds;
        None means 0.3 × the distance between the two vertices.
    lead_orientations : dict or None
        {(v, p): angle in degrees} of the drawn leads; loops avoid them.
    loop_angle : float
        Half opening angle (degrees) of a teardrop.
    """

    def __init__(self, topology, bond_spacing=None, lead_orientations=None, loop_angle=35.0):
        self.topology = topology
        t = topology
        pos = t.positions
        controls = [None] * t.n_bonds
        spacing_seq = (None if bond_spacing is None or np.isscalar(bond_spacing)
                       else [float(x) for x in bond_spacing])
        n_group = 0
        for (i, j), group in t.parallel_groups().items():
            if i == j:
                continue
            d = pos[j] - pos[i]
            dist = float(np.hypot(*d))
            u = d / dist if dist > 0 else np.array([1.0, 0.0])
            normal = np.array([-u[1], u[0]])
            if len(group) > 1 and spacing_seq:
                spacing = spacing_seq[n_group % len(spacing_seq)]
                n_group += 1
            elif bond_spacing is not None and np.isscalar(bond_spacing):
                spacing = float(bond_spacing)
            else:
                spacing = 0.3 * dist if dist > 0 else 1.0
            for k, b in enumerate(group):
                offset = (k - (len(group) - 1) / 2.0) * spacing
                if (i, j) != (t.bonds[b][0], t.bonds[b][2]):
                    offset = -offset                   # same side whatever the bond orientation
                a, z = pos[t.bonds[b][0]], pos[t.bonds[b][2]]
                if offset == 0.0:
                    controls[b] = np.array([a, z])
                else:
                    controls[b] = np.array([a, 0.5 * (a + z) + 2.0 * offset * normal, z])

        lead_dirs = {}
        for (v, p), angle in (lead_orientations or {}).items():
            lead_dirs.setdefault(v, []).append(np.deg2rad(angle))
        theta = np.deg2rad(loop_angle)
        unit = _bezier([[0, 0], _rotate([1, 0], theta), _rotate([1, 0], -theta), [0, 0]],
                       np.linspace(0, 1, _N_CURVE))
        unit_length = float(np.sum(np.hypot(*np.diff(unit, axis=0).T)))
        for v in range(t.n_vertices):
            loops = [b for b in t.vertex_bonds(v) if t.bonds[b][0] == t.bonds[b][2]]
            if not loops:
                continue
            c = pos[v]
            away = np.zeros(2)
            for b in t.vertex_bonds(v):
                i, _, j, _, _ = t.bonds[b]
                if i != j:
                    d = pos[j if i == v else i] - c
                    away -= d / max(np.hypot(*d), 1e-12)
            for a in lead_dirs.get(v, []):
                away -= np.array([np.cos(a), np.sin(a)])
            base = np.arctan2(away[1], away[0]) if np.hypot(*away) > 1e-9 else np.pi / 2
            busy = np.hypot(*away) > 1e-9
            spread = (min(np.deg2rad(70.0), np.pi / len(loops)) if busy
                      else 2 * np.pi / len(loops))
            for k, b in enumerate(loops):
                ang = base + (k - (len(loops) - 1) / 2.0) * spread
                u = np.array([np.cos(ang), np.sin(ang)])
                h = t.length[b] / unit_length
                controls[b] = np.array([c, c + h * _rotate(u, theta), c + h * _rotate(u, -theta), c])

        self._curves = []
        self._cum = []
        self._drawn = []
        for ctrl in controls:
            pts = _bezier(ctrl, np.linspace(0, 1, 2 if len(ctrl) == 2 else _N_CURVE))
            seg = np.hypot(*np.diff(pts, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            self._curves.append(pts)
            self._cum.append(cum / cum[-1])
            self._drawn.append(float(cum[-1]))

    # -- geometry ---------------------------------------------------------------

    def curve(self, b):
        """Points of bond ``b`` from its start vertex to its end vertex, shape (n, 2)."""
        return self._curves[b]

    def point(self, b, s):
        """Points at arc-length fractions ``s`` (array) of bond ``b``, from its start."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        pts, cum = self._curves[b], self._cum[b]
        return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=-1)

    def tangent(self, b, s):
        """Unit tangents (direction start -> end) at arc-length fractions ``s``."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        eps = 1e-3
        d = self.point(b, np.minimum(s + eps, 1.0)) - self.point(b, np.maximum(s - eps, 0.0))
        return d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-300)[:, None]

    def drawn_length(self, b):
        """Length of the drawn curve of bond ``b`` (µm in the figure; equal to the
        bond length for a self-loop only)."""
        return self._drawn[b]

    def directed_point(self, d, s):
        """Points at fractions ``s`` along directed bond ``d`` (from its source vertex)."""
        t = self.topology
        s = np.asarray(s, dtype=float)
        return self.point(t.bond_of(d), s if t.is_forward(d) else 1.0 - s)

    def directed_tangent(self, d, s):
        """Unit tangents along directed bond ``d`` (direction of propagation)."""
        t = self.topology
        s = np.asarray(s, dtype=float)
        b = t.bond_of(d)
        return self.tangent(b, s) if t.is_forward(d) else -self.tangent(b, 1.0 - s)

    def bounds(self):
        """(xmin, xmax, ymin, ymax) of all curves and vertices."""
        pts = np.vstack(self._curves + [self.topology.positions])
        return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def size_scale(topology):
    """Reference length of the figure elements: the shortest bond (µm)."""
    return float(np.min(topology.length))


def draw_bonds(ax, paths, bonds=None, color="black", linewidth=3.0, values=None, cmap=None,
               vmin=None, vmax=None, zorder=1, **kwargs):
    """Draw bond curves. ``values`` (one per drawn bond) colours them with ``cmap``;
    ``linewidth`` may be one value per drawn bond. Returns the LineCollection."""
    bonds = range(paths.topology.n_bonds) if bonds is None else list(bonds)
    lc = LineCollection([paths.curve(b) for b in bonds], linewidths=linewidth,
                        capstyle="round", joinstyle="round", zorder=zorder, **kwargs)
    if values is not None:
        lc.set_array(np.asarray(values, dtype=float))
        lc.set_cmap(cmap)
        lc.set_clim(vmin, vmax)
    else:
        lc.set_color(color)
    ax.add_collection(lc)
    return lc


def draw_vertices(ax, topology, radius, labels=None, fontsize=20, color="black",
                  text_color="white", vertices=None, zorder=5):
    """Vertex disks of radius ``radius`` (µm) with their labels (``topology.vertex_labels``
    by default; ``labels=False`` for none)."""
    from matplotlib.patches import Circle
    if labels is None:
        labels = topology.vertex_labels
    for v in (range(topology.n_vertices) if vertices is None else vertices):
        x, y = topology.positions[v]
        ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=color, zorder=zorder))
        if labels:
            ax.text(x, y, labels[v], color=text_color, fontsize=fontsize, fontweight="bold",
                    ha="center", va="center", zorder=zorder + 1)


def label_angle(direction):
    """Text rotation (degrees) following ``direction``, never upside down."""
    ang = np.degrees(np.arctan2(direction[1], direction[0]))
    return ((ang + 90.0) % 180.0) - 90.0


def draw_lengths(ax, paths, fontsize=16, fmt="{:.1f}", rotate=True, zorder=8):
    """Write each bond length on its curve, at mid-length."""
    for b, bond in enumerate(paths.topology.bonds):
        xy = paths.point(b, 0.5)
        rot = label_angle(paths.tangent(b, 0.5)[0]) if rotate else 0.0
        ax.text(xy[0], xy[1], fmt.format(bond[4]), fontsize=fontsize, ha="center", va="center",
                rotation=rot, rotation_mode="anchor", zorder=zorder,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


def draw_port_markers(ax, paths, vertex_radius, marker_radius, colors=PORT_COLORS,
                      lead_orientations=None, zorder=10):
    """Coloured disks at both ends of every bond and on the drawn leads, tangent
    to the vertex disk, on 4-port vertices (blue: ports 0, 1; orange: ports 2, 3)."""
    from matplotlib.patches import Circle
    t = paths.topology
    for v, port in t.leads:
        u = lead_direction(lead_orientations, (v, port))
        if u is not None and t.valency[v] == len(colors):
            center = t.positions[v] + (vertex_radius + marker_radius) * u
            ax.add_patch(Circle(center, marker_radius, facecolor=colors[port],
                                edgecolor="black", linewidth=0.8, zorder=zorder, clip_on=False))
    for b, (i, p, j, q, L) in enumerate(t.bonds):
        for v, port, s, sign in ((i, p, 0.0, 1.0), (j, q, 1.0, -1.0)):
            if t.valency[v] != len(colors):
                continue
            # point of the curve at distance vertex_radius + marker_radius from the vertex
            target = vertex_radius + marker_radius
            ss = np.linspace(0, 1, 2000)
            pts = paths.point(b, ss if s == 0.0 else 1.0 - ss)
            dist = np.hypot(*(pts - t.positions[v]).T)
            idx = np.argmax(dist >= target) if np.any(dist >= target) else len(ss) - 1
            ax.add_patch(Circle(pts[idx], marker_radius, facecolor=colors[port],
                                edgecolor="black", linewidth=0.8, zorder=zorder, clip_on=False))


def lead_direction(lead_orientations, lead):
    """Unit vector of a drawn lead (None if it has no orientation)."""
    if not lead_orientations or tuple(lead) not in lead_orientations:
        return None
    a = np.deg2rad(lead_orientations[tuple(lead)])
    return np.array([np.cos(a), np.sin(a)])


def default_lead_length(topology):
    """Drawn length of a lead (µm): 0.45 × the median bond length."""
    return 0.45 * float(np.median(topology.length))


def draw_leads(ax, topology, lead_orientations, length=None, linewidth=2.0, zorder=1):
    """Dashed segments for the leads that have an orientation; returns their end points."""
    length = default_lead_length(topology) if length is None else length
    ends = []
    for lead in topology.leads:
        u = lead_direction(lead_orientations, lead)
        if u is None:
            continue
        start = topology.positions[lead[0]]
        end = start + length * u
        ax.plot([start[0], end[0]], [start[1], end[1]], linestyle="--", linewidth=linewidth,
                color="black", zorder=zorder)
        ends.append(end)
    return ends


def intensity_cmap(cmap=None):
    """Colormap of the bond-intensity figures (default: Reds truncated at 0.75)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if cmap is None:
        base = plt.get_cmap("Reds")
        return LinearSegmentedColormap.from_list(
            "Reds_truncated", [base(x) for x in np.linspace(0.0, 0.75, 256)])
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def draw_bond_values(ax, paths, values, cmap=None, width_min=6.0, width_gain=8.0, zorder=1):
    """Bonds coloured and thickened by ``values`` in [0, 1] (one per bond), with a
    black outline. Returns the coloured LineCollection (for a colorbar)."""
    values = np.asarray(values, dtype=float)
    widths = width_min + width_gain * values
    draw_bonds(ax, paths, linewidth=widths + 2.0, color="black", zorder=zorder)
    return draw_bonds(ax, paths, linewidth=widths, values=values, cmap=intensity_cmap(cmap),
                      vmin=0.0, vmax=1.0, zorder=zorder + 1)


def add_colorbar(ax, mappable, label, fontsize=22):
    """Colorbar in an inset on the right of ``ax`` (core matplotlib only: a stale
    system mpl_toolkits can shadow the pip one and break append_axes)."""
    import matplotlib.pyplot as plt
    cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    return cbar


def finish_axes(ax, paths, margin=0.08, extra_points=None):
    """Equal aspect, no axis frame, limits fitting every curve."""
    xmin, xmax, ymin, ymax = paths.bounds()
    if extra_points is not None and len(extra_points):
        pts = np.asarray(extra_points, dtype=float).reshape(-1, 2)
        xmin, xmax = min(xmin, pts[:, 0].min()), max(xmax, pts[:, 0].max())
        ymin, ymax = min(ymin, pts[:, 1].min()), max(ymax, pts[:, 1].max())
    pad = margin * max(xmax - xmin, ymax - ymin, 1e-9)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_graph(ax, paths, show_lengths=False, show_port_markers=False, show_labels=True,
               lead_orientations=None, vertex_radius_frac=0.065, marker_radius_frac=0.03,
               vertex_fontsize=20, length_fontsize=16, bond_linewidth=3.0, lead_length=None,
               bond_color="black"):
    """Topology figure: bonds, vertices with labels, and optionally bond lengths,
    port markers and leads. Sizes are fractions of the shortest bond length."""
    t = paths.topology
    scale = size_scale(t)
    rv, rm = vertex_radius_frac * scale, marker_radius_frac * scale
    draw_bonds(ax, paths, linewidth=bond_linewidth, color=bond_color)
    ends = draw_leads(ax, t, lead_orientations, lead_length) if lead_orientations else []
    draw_vertices(ax, t, rv, labels=None if show_labels else False, fontsize=vertex_fontsize)
    if show_lengths:
        draw_lengths(ax, paths, fontsize=length_fontsize)
    if show_port_markers:
        draw_port_markers(ax, paths, rv, rm, lead_orientations=lead_orientations)
    finish_axes(ax, paths, extra_points=ends)
