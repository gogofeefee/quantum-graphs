#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classical_graphs — Classical dynamics on closed directed graphs.

Usage
-----
from classical_graphs import ClassicalGraphs, DirectionalClassicalGraphs

Classes
-------
ClassicalGraphs
    Base class. Accepts arbitrary per-vertex scattering matrices
    (constant arrays or callables σ(k)).

DirectionalClassicalGraphs
    Specialisation for 4-port directional-coupler graphs. Builds scattering
    matrices from coupling constants and adds port-marker graph visualisation.
"""

from .core       import CoreMixin
from .dynamics   import DynamicsMixin
from .plotting   import GraphPlotMixin, RhoPlotMixin, AnimationMixin


class ClassicalGraphs(
    CoreMixin, DynamicsMixin,
    GraphPlotMixin, RhoPlotMixin, AnimationMixin,
):
    """
    Classical Perron-Frobenius dynamics on a closed graph.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of each vertex (µm).
    port_map : list of lists
        Connectivity. port_map[i][p] = (j, q) connects port p of vertex i
        to port q of vertex j. No None entries (closed graph).
    lengths : dict or None
        Bond lengths {((i,p),(j,q)): L_um}. None → Euclidean distances.
    vertex_matrices : dict, list, ndarray, or callable
        Scattering matrices. dict/list: one per vertex. ndarray: same for all.
        callable: same function for all.
    virtual_vertex_spacing : float, sequence, or None
        Spacing when splitting self-loops / multiple bonds into virtual chains.
    name : str
        Graph label used in save-path construction.
    save_path : str, Path, or None
        Override the default deterministic save directory.
    base_dir : str, Path, or None
        Base directory for the save path. Priority: save_path > base_dir
        > sys.argv[0].parent > cwd.
    """
    pass


from .directional import DirectionalClassicalGraphs  # noqa: E402 (after ClassicalGraphs)


__all__ = [
    "ClassicalGraphs",
    "DirectionalClassicalGraphs",
]
