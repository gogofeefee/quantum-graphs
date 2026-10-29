#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — CoreMixin for classical_graphs: graph construction, save path and
parameter file (topology handling is shared through GraphBase).
"""

import inspect
import datetime
import numpy as np


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

    def _save_parameters(self):
        filepath = self.save_path / "graph_parameters.txt"
        t = self.topology
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("### Classical Graph Parameters ###\n")
            f.write(f"Graph name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n\n")
            f.write(f"Vertices: {self.n_vertices}\n")
            f.write(f"Bonds: {self.n_bonds}\n")
            f.write(f"Directed bonds: {t.n_directed}\n")
            f.write(f"Total length: {self.total_length:.6f}\n\n")
            f.write("### Vertex positions ###\n")
            for i, pos in enumerate(t.positions):
                f.write(f"{i}: ({pos[0]:.6f}, {pos[1]:.6f})\n")
            f.write("\n### Bonds (i, p, j, q, L) ###\n")
            for b, bond in enumerate(t.bonds):
                f.write(f"{b}: {bond}\n")
            f.write("\n### Vertex scattering matrices ###\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v}: callable\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency={t.valency[v]}):\n")
                    np.savetxt(f, np.asarray(sigma), fmt="%-10.6f")
                    f.write("\n")
