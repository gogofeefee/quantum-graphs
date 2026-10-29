"""
Ready-made graph geometries and graph transformations.

Every function returns a dict that can be passed to any graph class with
``**``::

    import quantum_graphs as qg
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), couplings=0.5,
                                   neff=3.97, wavelength_range=(1.50, 1.53))

The dict contains ``vertex_positions`` and ``bonds`` (``(i, p, j, q, L)``, L in
µm), plus ``leads`` and ``lead_orientations`` (degrees, drawing only) for open
layouts. Use :func:`closed` (drop the leads) or :func:`remove_access_coupler`
(delete the access coupler) to pass an open layout to a closed-graph class.

The FlowerGraph and BowTieGraph geometries are those of two fabricated
silicon devices (typical length 65 / 85 µm, coupler length 3.5 / 2.5 µm),
extracted from the lithography layout.
"""

import copy

import numpy as np


def single_ring(radius=100.0):
    """One 2-port vertex closed on itself (use a Neumann / pass-through matrix)."""
    return dict(vertex_positions=[(0.0, 0.0)], bonds=[(0, 0, 0, 1, 2 * np.pi * radius)])


def double_ring(r1=100.0, r2=100.0 / np.sqrt(2.0)):
    """One directional coupler whose two arms are closed into rings of radii r1, r2."""
    return dict(vertex_positions=[(0.0, 0.0)],
                bonds=[(0, 0, 0, 2, 2 * np.pi * r1), (0, 1, 0, 3, 2 * np.pi * r2)])


def ring_with_access(radius=100.0, n_access=1):
    """Open ring: one access coupler (self-loop) or two (add-drop, two half rings).

    Leads are on ports 0 and 2 of each access coupler.
    """
    if n_access == 1:
        return dict(vertex_positions=[(0.0, 0.0)], bonds=[(0, 1, 0, 3, 2 * np.pi * radius)],
                    leads=[(0, 0), (0, 2)], lead_orientations={(0, 0): -150.0, (0, 2): -30.0})
    if n_access == 2:
        return dict(vertex_positions=[(0.0, -radius), (0.0, radius)],
                    bonds=[(0, 1, 1, 1, np.pi * radius), (0, 3, 1, 3, np.pi * radius)],
                    leads=[(0, 0), (0, 2), (1, 0), (1, 2)],
                    lead_orientations={(0, 0): -150.0, (0, 2): -30.0,
                                       (1, 0): 150.0, (1, 2): 30.0})
    raise ValueError("n_access must be 1 or 2.")


def tetrahedron():
    """Complete graph K4 with 3-port vertices (use e.g. Neumann matrices)."""
    lA = 60.0
    return dict(
        vertex_positions=[(0.0, 0.0), (0.0, lA), (-np.sqrt(3) / 2 * np.sqrt(2) * lA, -0.5 * np.sqrt(2) * lA),
                          (np.sqrt(3) / 2 * np.sqrt(3) * lA, -0.5 * np.sqrt(3) * lA)],
        bonds=[(0, 0, 1, 0, 60.000), (0, 1, 2, 0, 84.853), (0, 2, 3, 0, 103.923),
               (1, 1, 2, 1, 202.050), (1, 2, 3, 1, 227.993), (2, 2, 3, 2, 261.801)])


def flower_graph(open=False):
    """Five directional couplers on a pentagon, each neighbouring pair joined by two
    bonds (L_tot = 1599.62 µm). ``open=True`` adds the access coupler (vertex 0)
    of the fabricated device, with leads on its ports 0 and 2.

    Silicon device of H. Girin et al., "Graphs on chip: a silicon photonics
    platform", arXiv:2605.12538 (2026).
    """
    if open:
        return dict(
            vertex_positions=[(0.000, -93.744), (0.000, 73.539), (-64.244, 20.874),
                              (-33.058, -45.500), (35.379, -48.695), (78.195, 25.407)],
            bonds=[(0, 1, 3, 3, 92.324), (0, 3, 4, 2, 91.162),
                   (1, 0, 5, 1, 123.830), (1, 1, 2, 0, 112.068),
                   (1, 2, 5, 3, 224.798), (1, 3, 2, 2, 215.516),
                   (2, 1, 3, 0, 100.260), (2, 3, 3, 2, 207.531),
                   (3, 1, 4, 0, 92.901), (4, 1, 5, 0, 116.961),
                   (4, 3, 5, 2, 222.271)],
            leads=[(0, 0), (0, 2)],
            lead_orientations={(0, 0): -150, (0, 2): -30})
    # written out rather than derived with remove_access_coupler: the merged
    # bond is 183.485 um in the layout (92.324 + 91.162 after rounding)
    return dict(
        vertex_positions=[(0.000, 73.539), (-64.244, 20.874), (-33.058, -45.500),
                          (35.379, -48.695), (78.195, 25.407)],
        bonds=[(0, 0, 4, 1, 123.830), (0, 1, 1, 0, 112.068),
               (0, 2, 4, 3, 224.798), (0, 3, 1, 2, 215.516),
               (1, 1, 2, 0, 100.260), (1, 3, 2, 2, 207.531),
               (2, 1, 3, 0, 92.901), (2, 3, 3, 2, 183.485),
               (3, 1, 4, 0, 116.961), (3, 3, 4, 2, 222.271)])


def bow_tie_graph(open=False):
    """Five directional couplers, 10 bonds of which two are doubled
    (L_tot = 1597.07 µm). ``open=True`` adds the access coupler (vertex 0) of
    the fabricated device, with leads on its ports 0 and 2.

    Silicon device of H. Girin et al., "Graphs on chip: a silicon photonics
    platform", arXiv:2605.12538 (2026).
    """
    if open:
        return dict(
            vertex_positions=[(0.000, -91.942), (0.000, 0.000), (83.283, 48.083),
                              (-76.500, 44.167), (-63.693, -36.773), (68.166, -39.356)],
            bonds=[(0, 1, 4, 0, 124.697), (0, 3, 5, 2, 126.560),
                   (1, 0, 4, 2, 79.153), (1, 1, 3, 3, 95.085),
                   (1, 2, 5, 0, 84.717), (1, 3, 2, 1, 103.523),
                   (2, 0, 5, 1, 96.144), (2, 2, 5, 3, 262.852),
                   (2, 3, 3, 1, 279.690), (3, 0, 4, 1, 255.817),
                   (3, 2, 4, 3, 88.833)],
            leads=[(0, 0), (0, 2)],
            lead_orientations={(0, 0): -150, (0, 2): -30})
    return remove_access_coupler(bow_tie_graph(open=True))


def remove_access_coupler(layout, vertex=0):
    """Close an open layout: delete ``vertex`` (which must have exactly two
    bonds), join its two bonds into one bond of the summed length (appended
    last), and renumber the remaining vertices (vertex v > ``vertex`` becomes
    v − 1).

    Leads of other vertices are kept; ``leads`` and ``lead_orientations`` are
    dropped if no lead remains.
    """
    def renum(v):
        return v - 1 if v > vertex else v

    ends, kept = [], []
    for (i, p, j, q, L) in layout["bonds"]:
        if i == vertex and j == vertex:
            raise ValueError(f"Vertex {vertex} carries a self-loop.")
        if i == vertex:
            ends.append(((j, q), L))
        elif j == vertex:
            ends.append(((i, p), L))
        else:
            kept.append((renum(i), p, renum(j), q, L))
    if len(ends) != 2:
        raise ValueError(f"Vertex {vertex} has {len(ends)} bonds; expected 2.")
    ((a, pa), La), ((b, pb), Lb) = ends
    bonds = kept + [(renum(a), pa, renum(b), pb, La + Lb)]
    out = dict(vertex_positions=[pos for v, pos in enumerate(layout["vertex_positions"])
                                 if v != vertex], bonds=bonds)
    leads = [(renum(v), p) for v, p in layout.get("leads", ()) if v != vertex]
    if leads:
        out["leads"] = leads
        out["lead_orientations"] = {(renum(v), p): ang for (v, p), ang
                                    in layout.get("lead_orientations", {}).items() if v != vertex}
    return out


def from_port_map(vertex_positions, port_map, lengths=None):
    """Convert the description of 1.0 (``port_map[i][p] = (j, q)`` or ``None``
    for a lead, ``lengths = {((i, p), (j, q)): L}``) into a layout dict with
    ``bonds`` and ``leads``, for tools that still produce port maps (e.g. GDS
    layout builders). Bonds are listed in order of first appearance; a bond
    without a length gets the Euclidean distance."""
    lengths = {} if lengths is None else lengths
    canon = {tuple(sorted(((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))))): float(L)
             for (a, b), L in lengths.items()}
    bonds, leads, seen = [], [], set()
    for i, ports in enumerate(port_map):
        for p, conn in enumerate(ports):
            if conn is None:
                leads.append((i, p))
                continue
            j, q = int(conn[0]), int(conn[1])
            key = tuple(sorted(((i, p), (j, q))))
            if key in seen:
                continue
            seen.add(key)
            bonds.append((i, p, j, q, canon[key]) if key in canon else (i, p, j, q))
    out = dict(vertex_positions=[tuple(map(float, xy)) for xy in vertex_positions], bonds=bonds)
    if leads:
        out["leads"] = leads
    return out


def closed(layout):
    """Copy of a layout without its leads (and their drawing orientations)."""
    out = copy.deepcopy(layout)
    out.pop("leads", None)
    out.pop("lead_orientations", None)
    return out
