#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_utils.py — Module-level helper functions shared across classical_graphs modules.
"""

import hashlib
import numpy as np
from pathlib import Path


def _as_path(path_like):
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def _normalize(vec, eps=1e-15):
    vec = np.asarray(vec, dtype=float)
    nrm = np.linalg.norm(vec)
    if nrm < eps:
        return np.array([1.0, 0.0], dtype=float)
    return vec / nrm


def _perpendicular(vec):
    vec = np.asarray(vec, dtype=float)
    return np.array([-vec[1], vec[0]], dtype=float)


def _canonical_port_edge(a, b):
    return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _safe_logical_allclose_nonnegative(arr, atol=1e-14):
    arr = np.asarray(arr, dtype=float)
    return bool(np.all(arr >= -atol))


def _format_float_for_name(x, ndigits=5):
    return f"{float(x):.{ndigits}f}"


def _euclidean_distance(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.hypot(*(p2 - p1)))


def _linear_interpolate(p0, p1, t):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return (1.0 - float(t)) * p0 + float(t) * p1


def _compute_parallel_offset_points(p0, p1, offset):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = _normalize(p1 - p0)
    normal = _perpendicular(direction)
    return p0 + offset * normal, p1 + offset * normal


def _validate_closed_port_map(port_map):
    for i, ports in enumerate(port_map):
        if not isinstance(ports, (list, tuple)):
            raise ValueError(f"port_map[{i}] must be a list or tuple.")
        for p, conn in enumerate(ports):
            if conn is None:
                raise ValueError(
                    f"Found None at port_map[{i}][{p}]. "
                    "This module only supports closed graphs."
                )
            if not (isinstance(conn, (list, tuple)) and len(conn) == 2):
                raise ValueError(
                    f"Invalid connection at port_map[{i}][{p}] = {conn}. "
                    "Each connection must be a tuple (j, q)."
                )


def _validate_vertex_positions(vertex_positions):
    if len(vertex_positions) == 0:
        raise ValueError("vertex_positions must contain at least one vertex.")
    for i, pos in enumerate(vertex_positions):
        if not (isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 2):
            raise ValueError(
                f"vertex_positions[{i}] = {pos} is invalid. "
                "Each position must be a 2D point."
            )


def _validate_lengths_dict(lengths):
    if lengths is None:
        return
    if not isinstance(lengths, dict):
        raise ValueError("lengths must be a dict or None.")
    for key, val in lengths.items():
        if not (
            isinstance(key, tuple) and len(key) == 2
            and isinstance(key[0], tuple) and isinstance(key[1], tuple)
            and len(key[0]) == 2 and len(key[1]) == 2
        ):
            raise ValueError(f"Invalid lengths key: {key}. Expected ((i, p), (j, q)).")
        if not _is_number(val):
            raise ValueError(f"Invalid length value for {key}: {val}.")
        if float(val) <= 0:
            raise ValueError(f"Bond lengths must be > 0. Invalid value for {key}: {val}.")


def _hash_graph_signature(port_map, lengths):
    payload = str(port_map) + str(lengths)
    return hashlib.md5(payload.encode()).hexdigest()[:8]
