"""
Graph topology: vertices, bonds, leads and directed bonds.

This is the single implementation shared by every graph class of the package.

Input
-----
* ``vertex_positions``: one (x, y) per vertex (µm). Used for drawing, and for
  the length of a bond given without one.
* ``bonds``: one entry per bond, ``(i, p, j, q)`` or ``(i, p, j, q, L)``: port
  ``p`` of vertex ``i`` is joined to port ``q`` of vertex ``j`` by a bond of
  length ``L`` (µm). Without ``L`` the Euclidean distance between the two
  vertices is used. Self-loops (``i == j``) and multiple bonds between the
  same vertices are ordinary bonds.
* ``leads``: ``(v, p)`` pairs, the external leads of an open graph, numbered in
  the order given.

Every port of a vertex carries exactly one bond end or one lead, and the ports
of a vertex are numbered 0, 1, ..., valency − 1.

Directed bonds
--------------
Bond ``b = (i, p, j, q)`` gives two directed bonds: ``d = b`` goes from port
``(i, p)`` to port ``(j, q)``, ``d = b + B`` the other way (B bonds, 2B directed
bonds). Arrays ``src_vertex``, ``src_port``, ``dst_vertex``, ``dst_port``,
and ``directed_length`` are indexed by ``d``; ``bond_of(d)``, ``is_forward(d)``
and ``reverse(d)`` hold the numbering convention.

Vertex labels
-------------
``vertex_labels`` (figures): the vertices without a lead are numbered 1, 2, ...
in index order; a vertex carrying leads (access coupler) is labelled "A"
(A1, A2, ... if there are several). For an open layout whose access coupler is
vertex 0, the labels are the closed-graph labels after
``layouts.remove_access_coupler``.
"""

from collections import defaultdict

import numpy as np


def vertex_labels(n_vertices, leads):
    """Figure labels of the vertices: {vertex: label}.

    Vertices without a lead are numbered "1", "2", ... in index order; the
    vertices carrying leads are "A", or "A1", "A2", ... if there are several.
    """
    access = sorted({v for v, _ in leads})
    labels = {v: ("A" if len(access) == 1 else f"A{n + 1}") for n, v in enumerate(access)}
    others = [v for v in range(n_vertices) if v not in labels]
    labels.update({v: str(n + 1) for n, v in enumerate(others)})
    return dict(sorted(labels.items()))


def _readonly(a):
    a = np.asarray(a)
    a.setflags(write=False)
    return a


class Topology:
    """Validated graph topology (see the module docstring). Do not modify it:
    build a new graph instead."""

    def __init__(self, vertex_positions, bonds, leads=(), allow_leads=True):
        positions = _validate_positions(vertex_positions)
        n = len(positions)
        bond_list = _validate_bonds(bonds, positions)
        lead_list = [_vertex_port(lead, n, f"leads[{m}]") for m, lead in enumerate(leads)]
        if lead_list and not allow_leads:
            raise ValueError("This graph class only supports closed graphs (no leads).")

        # every port carries exactly one bond end or lead; ports numbered 0..valency-1
        owner = {}
        for b, (i, p, j, q, _) in enumerate(bond_list):
            for end in ((i, p), (j, q)):
                if end in owner:
                    raise ValueError(f"Port {end} is used twice ({owner[end]} and bond {b}).")
                owner[end] = f"bond {b}"
        for m, lead in enumerate(lead_list):
            if lead in owner:
                raise ValueError(f"Port {lead} is used twice ({owner[lead]} and lead {m}).")
            owner[lead] = f"lead {m}"
        ports = defaultdict(set)
        for v, p in owner:
            ports[v].add(p)
        valency = np.zeros(n, dtype=int)
        for v in range(n):
            if not ports[v]:
                raise ValueError(f"Vertex {v} has no bond and no lead.")
            valency[v] = max(ports[v]) + 1
            missing = sorted(set(range(valency[v])) - ports[v])
            if missing:
                raise ValueError(f"Vertex {v}: ports {missing} are not connected (ports must "
                                 f"be numbered 0..{valency[v] - 1}).")

        B = len(bond_list)
        self.n_vertices = n
        self.n_bonds = B
        self.n_leads = len(lead_list)
        self.n_directed = 2 * B
        self.positions = _readonly(np.array(positions, dtype=float).reshape(n, 2))
        self.valency = _readonly(valency)
        self.bonds = tuple(bond_list)                    # (i, p, j, q, L), L as float
        self.leads = tuple(lead_list)
        self.length = _readonly(np.array([bd[4] for bd in bond_list], dtype=float))
        self.total_length = float(self.length.sum())
        self.vertex_labels = vertex_labels(n, lead_list)

        ij = np.array([(bd[0], bd[1], bd[2], bd[3]) for bd in bond_list], dtype=int).reshape(B, 4)
        self.src_vertex = _readonly(np.concatenate([ij[:, 0], ij[:, 2]]))
        self.src_port = _readonly(np.concatenate([ij[:, 1], ij[:, 3]]))
        self.dst_vertex = _readonly(np.concatenate([ij[:, 2], ij[:, 0]]))
        self.dst_port = _readonly(np.concatenate([ij[:, 3], ij[:, 1]]))
        self.directed_length = _readonly(np.concatenate([self.length, self.length]))

        # (v, p) -> directed bond leaving / arriving through that port; leads absent
        self._leaving = {(int(v), int(p)): d
                         for d, (v, p) in enumerate(zip(self.src_vertex, self.src_port))}
        self._lead_index = {lead: m for m, lead in enumerate(lead_list)}

    # -- lookups ----------------------------------------------------------------

    def reverse(self, d):
        """Directed bond(s) opposite to ``d`` (int or array)."""
        return (np.asarray(d) + self.n_bonds) % self.n_directed if np.ndim(d) else \
            (int(d) + self.n_bonds) % self.n_directed

    def bond_of(self, d):
        """Bond(s) of directed bond(s) ``d`` (int or array)."""
        return np.asarray(d) % self.n_bonds if np.ndim(d) else int(d) % self.n_bonds

    def is_forward(self, d):
        """True if directed bond ``d`` goes from the start (i, p) of its bond to its end (j, q)."""
        return np.asarray(d) < self.n_bonds if np.ndim(d) else int(d) < self.n_bonds

    def leaving(self, v, p):
        """Directed bond leaving vertex ``v`` through port ``p`` (None for a lead)."""
        return self._leaving.get((int(v), int(p)))

    def arriving(self, v, p):
        """Directed bond arriving at vertex ``v`` through port ``p`` (None for a lead)."""
        d = self.leaving(v, p)
        return None if d is None else self.reverse(d)

    def lead_index(self, v, p):
        """Index of the lead on port ``p`` of vertex ``v`` (None if it is a bond end)."""
        return self._lead_index.get((int(v), int(p)))

    def directed(self, i, p, j, q):
        """Directed bond going from port ``(i, p)`` to port ``(j, q)``."""
        d = self.leaving(i, p)
        if d is None or (self.dst_vertex[d], self.dst_port[d]) != (j, q):
            raise KeyError(f"No bond from {(i, p)} to {(j, q)}.")
        return d

    def bond_id(self, i, p, j, q):
        """Index of the bond joining ports ``(i, p)`` and ``(j, q)`` (either order)."""
        return self.bond_of(self.directed(i, p, j, q))

    def vertex_bonds(self, v):
        """Indices of the bonds touching vertex ``v`` (a self-loop once)."""
        return sorted({self.bond_of(d) for (u, _), d in self._leaving.items() if u == v})

    def parallel_groups(self):
        """{(i, j) with i <= j: [bonds joining i and j]}, in bond order (drawing)."""
        groups = defaultdict(list)
        for b, (i, _, j, _, _) in enumerate(self.bonds):
            groups[(min(i, j), max(i, j))].append(b)
        return dict(groups)

    # -- export -----------------------------------------------------------------

    def as_dict(self):
        """``dict(vertex_positions, bonds, leads)``: the input that rebuilds this topology."""
        return dict(vertex_positions=[tuple(map(float, p)) for p in self.positions],
                    bonds=[tuple(b) for b in self.bonds], leads=list(self.leads))

    def __repr__(self):
        return (f"Topology({self.n_vertices} vertices, {self.n_bonds} bonds, "
                f"{self.n_leads} leads, L = {self.total_length:.3f} µm)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_positions(vertex_positions):
    if vertex_positions is None or len(vertex_positions) == 0:
        raise ValueError("vertex_positions must contain at least one vertex.")
    out = []
    for i, pos in enumerate(vertex_positions):
        try:
            x, y = pos
            out.append((float(x), float(y)))
        except (TypeError, ValueError):
            raise ValueError(f"vertex_positions[{i}] = {pos!r} is not an (x, y) pair.") from None
    return out


def _vertex_port(pair, n, where):
    try:
        v, p = pair
        v, p = int(v), int(p)
    except (TypeError, ValueError):
        raise ValueError(f"{where} = {pair!r} must be a (vertex, port) pair.") from None
    if not 0 <= v < n or p < 0:
        raise ValueError(f"{where} = {pair!r}: no such vertex or port ({n} vertices).")
    return v, p


def _validate_bonds(bonds, positions):
    if bonds is None or len(bonds) == 0:
        raise ValueError("bonds must contain at least one bond (i, p, j, q[, L]).")
    n = len(positions)
    out = []
    for b, entry in enumerate(bonds):
        if not isinstance(entry, (list, tuple)) or len(entry) not in (4, 5):
            raise ValueError(f"bonds[{b}] = {entry!r} must be (i, p, j, q) or (i, p, j, q, L).")
        i, p = _vertex_port(entry[:2], n, f"bonds[{b}] start")
        j, q = _vertex_port(entry[2:4], n, f"bonds[{b}] end")
        if (i, p) == (j, q):
            raise ValueError(f"bonds[{b}] joins port {(i, p)} to itself.")
        if len(entry) == 5:
            try:
                L = float(entry[4])
            except (TypeError, ValueError):
                raise ValueError(f"bonds[{b}]: invalid length {entry[4]!r}.") from None
        else:
            L = float(np.hypot(*np.subtract(positions[j], positions[i])))
            if L == 0:
                raise ValueError(f"bonds[{b}] = {entry!r} has zero Euclidean length "
                                 "(self-loop or coincident vertices): give its length.")
        if not np.isfinite(L) or L <= 0:
            raise ValueError(f"bonds[{b}]: length must be finite and > 0, got {L}.")
        out.append((i, p, j, q, L))
    return out


def build_topology(vertex_positions, bonds, leads=(), allow_leads=True):
    """Validate the inputs and build a :class:`Topology`."""
    return Topology(vertex_positions, bonds, leads, allow_leads=allow_leads)
