"""
Graph topology: validation, bond lengths, virtualisation and directed bonds.

This is the single implementation shared by every graph class of the package.

Conventions
-----------
* ``port_map[i][p] = (j, q)``: port ``p`` of vertex ``i`` is connected to port
  ``q`` of vertex ``j``; both directions must be given. ``None`` marks an
  external lead (only allowed when ``allow_leads=True``).
* ``lengths``: ``{((i, p), (j, q)): L}`` in µm, one entry per bond, in either
  orientation. If omitted, Euclidean distances between vertex positions.
* Self-loops and multiple bonds are replaced by chains of *virtual* 2-port
  vertices with the pass-through matrix [[0, 1], [1, 0]]: a multiple bond is
  split in two halves, a self-loop in four quarters. The physics is unchanged
  (same total length, pure transmission); positions of virtual vertices only
  matter for drawing.
* Directed bonds: with B virtualised bonds ``(a, b)`` sorted canonically,
  index ``i < B`` is ``a -> b`` and index ``i + B`` is ``b -> a``.
"""

from collections import defaultdict

import numpy as np
import networkx as nx


def canonical_edge(a, b):
    """Canonical (sorted) key of the bond joining ports ``a`` and ``b``."""
    return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_vertex_positions(vertex_positions):
    if len(vertex_positions) == 0:
        raise ValueError("vertex_positions must contain at least one vertex.")
    out = []
    for i, pos in enumerate(vertex_positions):
        try:
            x, y = pos
            out.append((float(x), float(y)))
        except (TypeError, ValueError):
            raise ValueError(f"vertex_positions[{i}] = {pos!r} is not an (x, y) pair.") from None
    return out


def validate_port_map(port_map, n_vertices, allow_leads=False):
    """Check that ``port_map`` describes a consistent, symmetric graph."""
    if len(port_map) != n_vertices:
        raise ValueError(f"port_map has {len(port_map)} entries but there are "
                         f"{n_vertices} vertex positions.")
    for i, ports in enumerate(port_map):
        if not isinstance(ports, (list, tuple)):
            raise ValueError(f"port_map[{i}] must be a list of connections.")
        for p, conn in enumerate(ports):
            if conn is None:
                if not allow_leads:
                    raise ValueError(f"port_map[{i}][{p}] is None (external lead), "
                                     "but this graph class only supports closed graphs.")
                continue
            if not (isinstance(conn, (list, tuple)) and len(conn) == 2):
                raise ValueError(f"port_map[{i}][{p}] = {conn!r} must be a (vertex, port) pair.")
            j, q = int(conn[0]), int(conn[1])
            if not (0 <= j < n_vertices) or not (0 <= q < len(port_map[j])):
                raise ValueError(f"port_map[{i}][{p}] = {conn!r} points to a "
                                 "non-existent vertex or port.")
            if (j, q) == (i, p):
                raise ValueError(f"port_map[{i}][{p}] is connected to itself.")
            back = port_map[j][q]
            if back is None or (int(back[0]), int(back[1])) != (i, p):
                raise ValueError(f"port_map is not symmetric: port_map[{i}][{p}] = ({j}, {q}) "
                                 f"but port_map[{j}][{q}] = {back!r} (expected ({i}, {p})).")


def list_bonds(port_map):
    """Bonds as (i, pi, j, pj), in order of first appearance in ``port_map``."""
    bonds, seen = [], set()
    for i, ports in enumerate(port_map):
        for pi, conn in enumerate(ports):
            if conn is None:
                continue
            j, pj = int(conn[0]), int(conn[1])
            key = canonical_edge((i, pi), (j, pj))
            if key not in seen:
                seen.add(key)
                bonds.append((i, pi, j, pj))
    return bonds


def canonical_lengths(lengths, bonds, vertex_positions):
    """Return ``{canonical_edge: L}`` with exactly one positive length per bond.

    Raises a ValueError for a missing, unknown, duplicated or non-positive
    length, instead of silently falling back on a default.
    """
    bond_keys = [canonical_edge((i, pi), (j, pj)) for (i, pi, j, pj) in bonds]
    if lengths is None:
        out = {}
        for (a, b) in bond_keys:
            L = float(np.hypot(*(np.subtract(vertex_positions[b[0]], vertex_positions[a[0]]))))
            if L <= 0:
                raise ValueError(f"Bond {a}-{b} has zero Euclidean length (self-loop or "
                                 "coincident vertices): pass its length explicitly.")
            out[(a, b)] = L
        return out
    if not isinstance(lengths, dict):
        raise ValueError("lengths must be a dict {((i, p), (j, q)): L} or None.")
    out = {}
    for key, val in lengths.items():
        try:
            (i, p), (j, q) = key
            ckey = canonical_edge((int(i), int(p)), (int(j), int(q)))
            L = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid lengths entry {key!r}: {val!r}.") from None
        if not np.isfinite(L) or L <= 0:
            raise ValueError(f"Bond length must be finite and > 0: lengths[{key}] = {val}.")
        if ckey in out:
            raise ValueError(f"Bond {ckey} is given twice in lengths.")
        out[ckey] = L
    known = set(bond_keys)
    unknown = [k for k in out if k not in known]
    missing = [k for k in bond_keys if k not in out]
    if unknown:
        raise ValueError(f"lengths contains bonds that are not in port_map: {unknown}.")
    if missing:
        raise ValueError(f"Missing bond lengths for {missing}.")
    return {k: out[k] for k in bond_keys}


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

class Topology:
    """Validated, virtualised graph topology.

    Attributes keep the names used throughout the package (``virtual_port_map``,
    ``updated_bonds``, ``directed_bond_index``...). Build it with
    :func:`build_topology`.
    """

    #: attributes copied onto the graph objects by :meth:`apply_to`
    PUBLIC = ("vertex_positions", "n_vertices", "valencies", "bonds", "n_bonds", "leads",
              "n_leads", "virtual_port_map", "virtual_vertices", "updated_bonds",
              "virtual_lengths", "total_length", "G", "n_vertices_total",
              "n_virtual_vertices", "n_internal_bonds", "n_directed_bonds",
              "directed_bonds", "directed_bond_index", "directed_lengths",
              "directed_positions", "reverse_directed_bond", "canonical_lengths")

    def apply_to(self, obj):
        for name in self.PUBLIC:
            setattr(obj, name, getattr(self, name))
        return obj

    def directed_length_array(self):
        """Lengths of the directed bonds in index order."""
        out = np.empty(self.n_directed_bonds)
        for db, idx in self.directed_bond_index.items():
            out[idx] = self.directed_lengths[db]
        return out


def build_topology(vertex_positions, port_map, lengths=None, virtual_vertex_spacing=None,
                   allow_leads=False):
    """Validate the inputs and build a :class:`Topology`.

    Parameters
    ----------
    vertex_positions : sequence of (x, y)
    port_map : list of lists
    lengths : dict or None
    virtual_vertex_spacing : float, sequence or None
        Drawing only: perpendicular offset of the virtual vertices of a
        multiple bond. A sequence is used cyclically, one value per group of
        parallel bonds; None means 0.3 × the vertex distance.
    allow_leads : bool
        Accept ``None`` entries (external leads) in ``port_map``.
    """
    t = Topology()
    positions = validate_vertex_positions(vertex_positions)
    n = len(positions)
    validate_port_map(port_map, n, allow_leads=allow_leads)

    t.n_vertices = n
    t.valencies = [len(ports) for ports in port_map]
    t.bonds = list_bonds(port_map)
    t.n_bonds = len(t.bonds)
    t.leads = [(i, p) for i, ports in enumerate(port_map) for p, c in enumerate(ports) if c is None]
    t.n_leads = len(t.leads)
    t.canonical_lengths = canonical_lengths(lengths, t.bonds, positions)
    t.total_length = float(sum(t.canonical_lengths.values()))

    t.vertex_positions = list(positions)
    t.virtual_port_map = [[None if c is None else (int(c[0]), int(c[1])) for c in ports]
                          for ports in port_map]
    t.virtual_vertices = []
    vlengths = dict(t.canonical_lengths)
    updated = []

    spacing_seq = None
    if virtual_vertex_spacing is not None and not np.isscalar(virtual_vertex_spacing):
        spacing_seq = [float(x) for x in virtual_vertex_spacing] or None
    group_counter = [0]

    def next_group_spacing():
        if spacing_seq:
            val = spacing_seq[group_counter[0] % len(spacing_seq)]
            group_counter[0] += 1
            return val
        return virtual_vertex_spacing

    def new_virtual(pos):
        v = len(t.vertex_positions)
        t.vertex_positions.append((float(pos[0]), float(pos[1])))
        t.valencies.append(2)
        t.virtual_vertices.append(v)
        t.virtual_port_map.append([None, None])
        return v

    def split_loop(i, pi, pj):
        L = vlengths.pop(canonical_edge((i, pi), (i, pj)))
        c = np.asarray(t.vertex_positions[i], dtype=float)
        off = L / (4.0 * np.sqrt(2.0))
        v1, v2, v3 = (new_virtual(c + d) for d in ((-off, off), (0.0, 2 * off), (off, off)))
        chain = [((i, pi), (v1, 0)), ((v1, 1), (v2, 0)), ((v2, 1), (v3, 0)), ((v3, 1), (i, pj))]
        for a, b in chain:
            t.virtual_port_map[a[0]][a[1]] = b
            t.virtual_port_map[b[0]][b[1]] = a
            updated.append((a, b))
            vlengths[canonical_edge(a, b)] = L / 4.0

    def split_multiple(i, pi, j, pj, index, n_total, group_spacing):
        L = vlengths.pop(canonical_edge((i, pi), (j, pj)))
        p_i = np.asarray(t.vertex_positions[i], dtype=float)
        p_j = np.asarray(t.vertex_positions[j], dtype=float)
        d = p_j - p_i
        dist = float(np.hypot(*d))
        u = d / dist if dist > 0 else np.array([1.0, 0.0])
        normal = np.array([-u[1], u[0]])
        if group_spacing is not None:
            spacing = float(group_spacing)
        else:
            spacing = 0.3 * dist if dist > 0 else 1.0
        v = new_virtual(0.5 * (p_i + p_j) + (index - (n_total - 1) / 2.0) * spacing * normal)
        for a, b in (((i, pi), (v, 0)), ((v, 1), (j, pj))):
            t.virtual_port_map[a[0]][a[1]] = b
            t.virtual_port_map[b[0]][b[1]] = a
            updated.append((a, b))
            vlengths[canonical_edge(a, b)] = L / 2.0

    groups = defaultdict(list)
    for (i, pi, j, pj) in t.bonds:
        groups[tuple(sorted((i, j)))].append((i, pi, j, pj))
    for (i, j), group in groups.items():
        if i == j:
            for (_, pi, _, pj) in group:
                split_loop(i, pi, pj)
        elif len(group) == 1:
            i0, pi, j0, pj = group[0]
            updated.append(((i0, pi), (j0, pj)))
        else:
            spacing = next_group_spacing()
            for k, (i0, pi, j0, pj) in enumerate(group):
                split_multiple(i0, pi, j0, pj, k, len(group), spacing)

    t.updated_bonds = sorted((canonical_edge(a, b) for a, b in updated),
                             key=lambda ab: (ab[0][0], ab[0][1], ab[1][0], ab[1][1]))
    t.virtual_lengths = {key: vlengths[key] for key in t.updated_bonds}

    t.n_vertices_total = len(t.vertex_positions)
    t.n_virtual_vertices = len(t.virtual_vertices)
    t.n_internal_bonds = len(t.updated_bonds)
    t.n_directed_bonds = 2 * t.n_internal_bonds

    virtual = set(t.virtual_vertices)
    t.G = nx.Graph()
    for i, pos in enumerate(t.vertex_positions):
        t.G.add_node(i, pos=pos, is_virtual=(i in virtual))
    t.directed_bonds, t.directed_bond_index, t.directed_lengths = [], {}, {}
    t.directed_positions, t.reverse_directed_bond = {}, {}
    B = t.n_internal_bonds
    for idx, (a, b) in enumerate(t.updated_bonds):
        L = t.virtual_lengths[(a, b)]
        p_a = np.asarray(t.vertex_positions[a[0]], dtype=float)
        p_b = np.asarray(t.vertex_positions[b[0]], dtype=float)
        t.G.add_edge(a[0], b[0], length=L, port_edge=(a, b), p0=p_a, p1=p_b)
        fwd, bwd = (a, b), (b, a)
        t.directed_bonds += [fwd, bwd]
        t.directed_bond_index[fwd], t.directed_bond_index[bwd] = idx, idx + B
        t.directed_lengths[fwd] = t.directed_lengths[bwd] = L
        t.directed_positions[fwd], t.directed_positions[bwd] = (p_a, p_b), (p_b, p_a)
        t.reverse_directed_bond[fwd], t.reverse_directed_bond[bwd] = bwd, fwd
    return t
