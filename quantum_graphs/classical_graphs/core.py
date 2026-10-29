#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — CoreMixin for classical_graphs: graph construction, save path and
parameter file (topology handling is shared through GraphBase).
"""

from ..core.base import GraphBase
from ..core.persistence import topology_hash
from ..core.vertices import resolve_vertex_matrices
from ._utils import _format_float_for_name, _is_number


class CoreMixin(GraphBase):
    """
    Graph construction and save-path logic.

    Parameters
    ----------
    vertex_positions : list of (float, float)
    bonds : list of (i, p, j, q) or (i, p, j, q, L)
    vertex_matrices : dict, list, ndarray, or callable
    bond_spacing : float, sequence, or None
        Drawing only: distance between parallel bonds (see BondPaths).
    name : str
    save_path : str | Path | None
        Override the automatic save directory.
    base_dir : str | Path | None
        Base directory for the save path. Priority: save_path > base_dir >
        sys.argv[0].parent > cwd.
    """

    _parameters_title = "Classical Graph Parameters"

    def __init__(
        self,
        vertex_positions,
        bonds,
        *,
        vertex_matrices,
        bond_spacing=None,
        name="MyClassicalGraph",
        save_path=None,
        base_dir=None,
    ):
        self.name = str(name)
        self._setup_topology(vertex_positions, bonds)
        self.bond_spacing = bond_spacing
        self._setup_vertex_matrices(resolve_vertex_matrices(vertex_matrices, self.n_vertices))
        self._setup_save_path(self.name, self._build_save_dir_name(), save_path, base_dir)

    def _build_save_dir_name(self):
        info_str = f"Nv={self.n_vertices}_Nb={self.n_bonds}"
        bs = self.bond_spacing
        if bs is None:
            spacing_str = "bs=None"
        elif _is_number(bs):
            spacing_str = f"bs={_format_float_for_name(bs, ndigits=3)}"
        else:
            spacing_str = "bs=seq"
        return f"{info_str}_{spacing_str}_{topology_hash(self.topology)}"
