"""Literal geometries used by the reference (non-regression) data set.

Written out on purpose (not taken from ``quantum_graphs.layouts``), so that a
change of a layout cannot go unnoticed. Bonds ``(i, p, j, q, L)``, L in µm;
leads ``(v, p)``. Converted on 2026-10-29 from the port_map / lengths form
used up to reference_v3 (same bonds, same order of the leads); floats are
written with full precision.
"""

import numpy as np

FG_CLOSED = dict(
    vertex_positions=[(0.0, 73.539), (-64.244, 20.874), (-33.058, -45.5), (35.379, -48.695), (78.195, 25.407)],
    bonds=[
        (0, 0, 4, 1, 123.83),
        (0, 1, 1, 0, 112.068),
        (0, 2, 4, 3, 224.798),
        (0, 3, 1, 2, 215.516),
        (1, 1, 2, 0, 100.26),
        (1, 3, 2, 2, 207.531),
        (2, 1, 3, 0, 92.901),
        (2, 3, 3, 2, 183.485),
        (3, 1, 4, 0, 116.961),
        (3, 3, 4, 2, 222.271),
    ],
)

BTG_CLOSED = dict(
    vertex_positions=[(0.0, 0.0), (83.283, 48.083), (-76.5, 44.167), (-63.693, -36.773), (68.166, -39.356)],
    bonds=[
        (0, 0, 3, 2, 79.153),
        (0, 1, 2, 3, 95.085),
        (0, 2, 4, 0, 84.717),
        (0, 3, 1, 1, 103.523),
        (1, 0, 4, 1, 96.144),
        (1, 2, 4, 3, 262.852),
        (1, 3, 2, 1, 279.69),
        (2, 0, 3, 1, 255.817),
        (2, 2, 3, 3, 88.833),
        (3, 0, 4, 2, 251.257),
    ],
)

FG_OPEN = dict(
    vertex_positions=[(0.0, -93.744), (0.0, 73.539), (-64.244, 20.874), (-33.058, -45.5), (35.379, -48.695), (78.195, 25.407)],
    bonds=[
        (0, 1, 3, 3, 92.324),
        (0, 3, 4, 2, 91.162),
        (1, 0, 5, 1, 123.83),
        (1, 1, 2, 0, 112.068),
        (1, 2, 5, 3, 224.798),
        (1, 3, 2, 2, 215.516),
        (2, 1, 3, 0, 100.26),
        (2, 3, 3, 2, 207.531),
        (3, 1, 4, 0, 92.901),
        (4, 1, 5, 0, 116.961),
        (4, 3, 5, 2, 222.271),
    ],
    leads=[(0, 0), (0, 2)],
    lead_orientations={(0, 0): -150, (0, 2): -30},
)

BTG_OPEN = dict(
    vertex_positions=[(0.0, -91.942), (0.0, 0.0), (83.283, 48.083), (-76.5, 44.167), (-63.693, -36.773), (68.166, -39.356)],
    bonds=[
        (0, 1, 4, 0, 124.697),
        (0, 3, 5, 2, 126.56),
        (1, 0, 4, 2, 79.153),
        (1, 1, 3, 3, 95.085),
        (1, 2, 5, 0, 84.717),
        (1, 3, 2, 1, 103.523),
        (2, 0, 5, 1, 96.144),
        (2, 2, 5, 3, 262.852),
        (2, 3, 3, 1, 279.69),
        (3, 0, 4, 1, 255.817),
        (3, 2, 4, 3, 88.833),
    ],
    leads=[(0, 0), (0, 2)],
    lead_orientations={(0, 0): -150, (0, 2): -30},
)

DOUBLE_RING = dict(
    vertex_positions=[(0.0, 0.0)],
    bonds=[
        (0, 0, 0, 2, 628.3185307179587),
        (0, 1, 0, 3, 444.28829381583654),
    ],
)

SINGLE_RING = dict(
    vertex_positions=[(0.0, 0.0)],
    bonds=[
        (0, 0, 0, 1, 628.3185307179587),
    ],
)

TETRAHEDRON = dict(
    vertex_positions=[(0.0, 0.0), (60.0, 0.0), (0.0, 84.8528137423857), (-103.92304845413263, 0.0)],
    bonds=[
        (0, 0, 1, 0, 60.0),
        (0, 1, 2, 0, 84.853),
        (0, 2, 3, 0, 103.923),
        (1, 1, 2, 1, 202.05),
        (1, 2, 3, 1, 227.993),
        (2, 2, 3, 2, 261.801),
    ],
)

RING_2ACCESS = dict(
    vertex_positions=[(0.0, -100.0), (0.0, 100.0)],
    bonds=[
        (0, 1, 1, 1, 314.1592653589793),
        (0, 3, 1, 3, 314.1592653589793),
    ],
    leads=[(0, 0), (0, 2), (1, 0), (1, 2)],
)

RING_1ACCESS = dict(
    vertex_positions=[(0.0, 0.0)],
    bonds=[
        (0, 1, 0, 3, 628.3185307179587),
    ],
    leads=[(0, 0), (0, 2)],
)

def neumann(n):
    """Neumann-Kirchhoff vertex matrix 2/n J - I."""
    return 2.0 / n * np.ones((n, n)) - np.eye(n)


def directional_coupler(C):
    t, c = np.sqrt(1 - C), 1j * np.sqrt(C)
    return np.array([[0, 0, t, c], [0, 0, c, t], [t, c, 0, 0], [c, t, 0, 0]], dtype=complex)


def neff_dispersive(k):
    """Smooth dispersive index used by the reference data set."""
    return 2.40 + 0.08 * (np.asarray(k) - 4.0) - 0.01 * (np.asarray(k) - 4.0) ** 2


def coupling_dispersive(k):
    return 0.5 + 0.1 * np.sin(3.0 * np.asarray(k))
