#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_utils.py — Module-level helper functions shared across classical_graphs modules.
"""

import numpy as np


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


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _format_float_for_name(x, ndigits=5):
    return f"{float(x):.{ndigits}f}"
