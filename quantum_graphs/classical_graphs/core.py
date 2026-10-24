#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — CoreMixin for classical_graphs.

Handles graph construction: validation, length inference, virtualisation of
self-loops and multiple bonds, directed-bond indexing, save-path logic, and
parameter saving.
"""

import copy
import inspect
import datetime
import numpy as np


from ..core.base import GraphBase
from ..core.vertices import resolve_vertex_matrices
from ._utils import (_compute_parallel_offset_points, _format_float_for_name,
                     _hash_graph_signature, _is_number)


class CoreMixin(GraphBase):
    """
    Graph construction, virtualisation, directed-bond indexing, and save-path logic.

    Parameters
    ----------
    vertex_positions : list of (float, float)
    port_map : list of lists
    lengths : dict or None
    vertex_matrices : dict, list, ndarray, or callable
    virtual_vertex_spacing : float, sequence, or None
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
        port_map,
        lengths=None,
        vertex_matrices=None,
        virtual_vertex_spacing=None,
        name="MyClassicalGraph",
        save_path=None,
        base_dir=None,
    ):
        self.name = str(name)
        self.vertex_matrices_input = vertex_matrices
        self._setup_topology(vertex_positions, copy.deepcopy(port_map), lengths,
                             virtual_vertex_spacing)
        if vertex_matrices is None:
            raise ValueError("vertex_matrices is required (dict, list, array or sigma(k)).")
        self._setup_vertex_matrices(resolve_vertex_matrices(vertex_matrices, self.n_vertices))
        self._setup_save_path(self.name, self._build_save_dir_name(), save_path, base_dir)

    def _build_save_dir_name(self):
        info_str = f"Nv={len(self.vertex_positions)}_Nb={len(self.updated_bonds)}"
        vvs = self.virtual_vertex_spacing
        if vvs is None:
            spacing_str = "vvs=None"
        elif _is_number(vvs):
            spacing_str = f"vvs={_format_float_for_name(vvs, ndigits=3)}"
        else:
            spacing_str = "vvs=seq"
        digest = _hash_graph_signature(self.port_map, self.lengths)
        return f"{info_str}_{spacing_str}_{digest}"

    def _save_parameters(self):
        filepath = self.save_path / "graph_parameters.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("### Classical Graph Parameters ###\n")
            f.write(f"Graph name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n\n")
            f.write(f"Original vertices: {self.n_vertices}\n")
            f.write(f"Total vertices after virtualisation: {len(self.vertex_positions)}\n")
            f.write(f"Virtual vertices: {self.virtual_vertices}\n")
            f.write(f"Internal undirected bonds: {len(self.updated_bonds)}\n")
            f.write(f"Directed bonds: {2 * len(self.updated_bonds)}\n")
            f.write(f"Total raw length: {self.total_length:.6f}\n\n")
            f.write("### Raw vertex positions ###\n")
            for i, pos in enumerate(self.vertex_positions[:self.n_vertices]):
                f.write(f"{i}: {pos}\n")
            f.write("\n### Raw port_map ###\n")
            for i, ports in enumerate(self.port_map):
                f.write(f"Vertex {i}: {ports}\n")
            f.write("\n### Raw lengths ###\n")
            for key, val in self.lengths.items():
                f.write(f"{key}: {float(val):.6f}\n")
            f.write("\n### Virtualised lengths ###\n")
            for key, val in self.virtual_lengths.items():
                f.write(f"{key}: {float(val):.6f}\n")
            f.write("\n### Updated bonds ###\n")
            for bond in self.updated_bonds:
                f.write(f"{bond}\n")
            f.write("\n### Vertex scattering matrices ###\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v}: callable\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency={self.valencies[v]}):\n")
                    np.savetxt(f, np.asarray(sigma), fmt="%-10.6f")
                    f.write("\n")

    # ==================================================================
    # Directed-bond geometry accessors
    # ==================================================================

    def get_directed_bond_coordinates(self, dbond):
        return self.directed_positions[dbond]

    def get_directed_bond_offset_coordinates(self, dbond, offset):
        p0, p1 = self.get_directed_bond_coordinates(dbond)
        return _compute_parallel_offset_points(p0, p1, offset)
