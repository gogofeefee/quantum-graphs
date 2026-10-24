#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — CoreMixin for classical_graphs.

Handles graph construction: validation, length inference, virtualisation of
self-loops and multiple bonds, directed-bond indexing, save-path logic, and
parameter saving.
"""

import os
import sys
import copy
import inspect
import datetime
import numpy as np
import networkx as nx

from collections import defaultdict, OrderedDict
from pathlib import Path

from ._utils import (
    _as_path, _is_number, _normalize, _perpendicular,
    _canonical_port_edge, _format_float_for_name,
    _euclidean_distance, _compute_parallel_offset_points,
    _validate_closed_port_map, _validate_vertex_positions,
    _validate_lengths_dict, _hash_graph_signature,
)


class CoreMixin:
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
        # ------------------------------------------------------------------
        # 1) Validate inputs
        # ------------------------------------------------------------------
        _validate_vertex_positions(vertex_positions)
        _validate_closed_port_map(port_map)
        _validate_lengths_dict(lengths)

        self.vertex_positions = [tuple(map(float, pos)) for pos in vertex_positions]
        self.port_map = copy.deepcopy(port_map)
        self.virtual_port_map = copy.deepcopy(port_map)
        self.lengths = copy.deepcopy(lengths)
        self.vertex_matrices_input = vertex_matrices
        self.name = str(name)

        self._init_spacing_sequence(virtual_vertex_spacing)
        self.virtual_vertex_spacing = virtual_vertex_spacing

        self.n_vertices = len(self.vertex_positions)
        self.valencies = [len(ports) for ports in self.port_map]

        # ------------------------------------------------------------------
        # 2) Compute raw lengths from Euclidean distances if not provided
        # ------------------------------------------------------------------
        if self.lengths is None:
            self.lengths = {}
            for i, ports in enumerate(self.port_map):
                for pi, conn in enumerate(ports):
                    j, pj = conn
                    key = _canonical_port_edge((i, pi), (j, pj))
                    if key not in self.lengths:
                        self.lengths[key] = _euclidean_distance(
                            self.vertex_positions[i], self.vertex_positions[j]
                        )

        self.total_length = float(np.sum(list(self.lengths.values()))) if self.lengths else 0.0
        self.virtual_lengths = copy.deepcopy(self.lengths)

        # ------------------------------------------------------------------
        # 3) Build raw bond list from ORIGINAL port_map
        # ------------------------------------------------------------------
        self.bonds = []
        for i, ports in enumerate(self.port_map):
            for pi, conn in enumerate(ports):
                j, pj = conn
                candidate = (i, pi, j, pj)
                if (j, pj, i, pi) not in self.bonds:
                    self.bonds.append(candidate)
        self.n_bonds = len(self.bonds)

        # ------------------------------------------------------------------
        # 4) Group bonds by unordered vertex pair
        # ------------------------------------------------------------------
        vertex_pair_bonds = defaultdict(list)
        for (i, pi, j, pj) in self.bonds:
            pair = tuple(sorted([i, j]))
            vertex_pair_bonds[pair].append(((i, pi), (j, pj)))

        # ------------------------------------------------------------------
        # 5) Virtualise self-loops and multiple bonds
        # ------------------------------------------------------------------
        self.virtual_vertices = []
        self.updated_bonds = []

        for (i, j), bond_list in vertex_pair_bonds.items():
            if i == j:
                for bond in bond_list:
                    self._break_complex_bond(bond=bond, loop=True)
            elif len(bond_list) == 1:
                self.updated_bonds.append(bond_list[0])
            else:
                group_spacing = self._next_group_spacing()
                for k, bond in enumerate(bond_list):
                    self._break_complex_bond(
                        bond=bond, loop=False,
                        index=k, n_total=len(bond_list),
                        group_spacing=group_spacing,
                    )

        self._order_bonds_and_lengths()

        # ------------------------------------------------------------------
        # 6) Build NetworkX graph container
        # ------------------------------------------------------------------
        self.G = nx.Graph()
        for i, pos in enumerate(self.vertex_positions):
            self.G.add_node(i, pos=pos, is_virtual=(i in self.virtual_vertices))

        for a, b in self.updated_bonds:
            key = _canonical_port_edge(a, b)
            self.G.add_edge(
                a[0], b[0],
                length=self.virtual_lengths[key],
                port_edge=key,
                p0=np.asarray(self.vertex_positions[a[0]], dtype=float),
                p1=np.asarray(self.vertex_positions[b[0]], dtype=float),
            )

        # ------------------------------------------------------------------
        # 7) Assign vertex scattering matrices
        # ------------------------------------------------------------------
        self.vertex_scattering_matrix = {}
        self._assign_vertex_matrices(vertex_matrices)

        # ------------------------------------------------------------------
        # 8) Post-virtualisation counts
        # ------------------------------------------------------------------
        self.n_vertices_total = len(self.vertex_positions)
        self.n_virtual_vertices = len(self.virtual_vertices)
        self.n_internal_bonds = len(self.updated_bonds)
        self.n_directed_bonds = 2 * self.n_internal_bonds

        # ------------------------------------------------------------------
        # 9) Directed bonds and indexing
        # ------------------------------------------------------------------
        self.directed_bonds = []
        self.directed_bond_index = {}
        self.directed_lengths = {}
        self.directed_positions = {}
        self.reverse_directed_bond = {}

        for idx, (a, b) in enumerate(self.updated_bonds):
            fwd, bwd = (a, b), (b, a)
            self.directed_bonds += [fwd, bwd]
            self.directed_bond_index[fwd] = idx
            self.directed_bond_index[bwd] = idx + self.n_internal_bonds

            key = _canonical_port_edge(a, b)
            L = float(self.virtual_lengths[key])
            self.directed_lengths[fwd] = L
            self.directed_lengths[bwd] = L

            p_a = np.asarray(self.vertex_positions[a[0]], dtype=float)
            p_b = np.asarray(self.vertex_positions[b[0]], dtype=float)
            self.directed_positions[fwd] = (p_a, p_b)
            self.directed_positions[bwd] = (p_b, p_a)

            self.reverse_directed_bond[fwd] = bwd
            self.reverse_directed_bond[bwd] = fwd

        # ------------------------------------------------------------------
        # 10) Save-path logic — base_dir > sys.argv[0].parent > cwd
        # ------------------------------------------------------------------
        if save_path is not None:
            self.save_path = _as_path(save_path)
        else:
            if base_dir is not None:
                _base = Path(base_dir)
            else:
                try:
                    _argv0 = sys.argv[0] if sys.argv else ""
                    if _argv0 and os.path.isfile(_argv0):
                        _base = Path(_argv0).parent
                    else:
                        _base = Path.cwd()
                except Exception:
                    _base = Path.cwd()
            self.save_path = _base / self.name / self._build_save_dir_name()

        self.save_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 11) Save parameters to text file
        # ------------------------------------------------------------------
        self._save_parameters()

    # ==================================================================
    # Spacing helpers
    # ==================================================================

    def _init_spacing_sequence(self, spacing):
        self._spacing_sequence = None
        self._spacing_ptr = 0
        if spacing is None or _is_number(spacing):
            return
        seq = list(spacing)
        if seq:
            self._spacing_sequence = [float(x) for x in seq]

    def _next_group_spacing(self):
        if self._spacing_sequence:
            val = self._spacing_sequence[self._spacing_ptr % len(self._spacing_sequence)]
            self._spacing_ptr += 1
            return val
        return self.virtual_vertex_spacing

    # ==================================================================
    # Vertex matrices
    # ==================================================================

    def _assign_vertex_matrices(self, vertex_matrices):
        neumann_2x2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        for v in range(len(self.vertex_positions)):
            if v in self.virtual_vertices:
                self.vertex_scattering_matrix[v] = neumann_2x2
                continue
            if isinstance(vertex_matrices, dict):
                self.vertex_scattering_matrix[v] = vertex_matrices[v]
            elif isinstance(vertex_matrices, list):
                self.vertex_scattering_matrix[v] = vertex_matrices[v]
            elif isinstance(vertex_matrices, np.ndarray):
                self.vertex_scattering_matrix[v] = vertex_matrices
            elif callable(vertex_matrices):
                self.vertex_scattering_matrix[v] = vertex_matrices
            else:
                raise ValueError(
                    "vertex_matrices must be a dict, list, ndarray, or callable."
                )

    # ==================================================================
    # Virtualisation
    # ==================================================================

    def _break_complex_bond(self, bond, loop=False, index=0, n_total=1, group_spacing=None):
        (i, pi), (j, pj) = bond
        xi, yi = self.vertex_positions[i]
        xj, yj = self.vertex_positions[j]
        p_i = np.array([xi, yi], dtype=float)
        p_j = np.array([xj, yj], dtype=float)
        diff = p_j - p_i
        distance = float(np.linalg.norm(diff))
        direction = _normalize(diff if distance > 0 else np.array([1.0, 0.0]))
        normal = _perpendicular(direction)

        if group_spacing is not None:
            spacing = float(group_spacing)
        elif _is_number(self.virtual_vertex_spacing):
            spacing = float(self.virtual_vertex_spacing)
        else:
            spacing = 0.3 * distance if distance > 0 else 1.0

        if loop:
            old_key = _canonical_port_edge((i, pi), (j, pj))
            L_total = float(self.virtual_lengths.pop(old_key, 40.0))
            self.virtual_port_map[i][pi] = None
            self.virtual_port_map[i][pj] = None

            center = p_i
            offset = L_total / (4.0 * np.sqrt(2.0))
            loop_positions = [(-offset, offset), (0.0, 2.0 * offset), (offset, offset)]

            new_verts = []
            for dx, dy in loop_positions:
                v_idx = len(self.vertex_positions)
                self.vertex_positions.append((center[0] + dx, center[1] + dy))
                self.valencies.append(2)
                self.virtual_vertices.append(v_idx)
                self.virtual_port_map.append([None, None])
                new_verts.append(v_idx)

            v1, v2, v3 = new_verts
            self.virtual_port_map[i][pi] = (v1, 0)
            self.virtual_port_map[i][pj] = (v3, 1)
            self.virtual_port_map[v1][0] = (i, pi)
            self.virtual_port_map[v1][1] = (v2, 0)
            self.virtual_port_map[v2][0] = (v1, 1)
            self.virtual_port_map[v2][1] = (v3, 0)
            self.virtual_port_map[v3][0] = (v2, 1)
            self.virtual_port_map[v3][1] = (i, pj)

            seg_len = L_total / 4.0
            for a, b in [((i, pi), (v1, 0)), ((v1, 1), (v2, 0)),
                         ((v2, 1), (v3, 0)), ((v3, 1), (i, pj))]:
                self.updated_bonds.append((a, b))
                self.virtual_lengths[_canonical_port_edge(a, b)] = seg_len
            return

        # Multiple bond
        old_key = _canonical_port_edge((i, pi), (j, pj))
        L_total = float(self.virtual_lengths.pop(old_key, 40.0))

        midpoint = 0.5 * (p_i + p_j)
        offset_signed = (index - (n_total - 1) / 2.0) * spacing
        p_virtual = midpoint + offset_signed * normal

        v_idx = len(self.vertex_positions)
        self.vertex_positions.append((float(p_virtual[0]), float(p_virtual[1])))
        self.valencies.append(2)
        self.virtual_vertices.append(v_idx)
        self.virtual_port_map.append([None, None])
        self.virtual_port_map[i][pi] = (v_idx, 0)
        self.virtual_port_map[v_idx][0] = (i, pi)
        self.virtual_port_map[j][pj] = (v_idx, 1)
        self.virtual_port_map[v_idx][1] = (j, pj)

        seg_len = L_total / 2.0
        b1 = ((i, pi), (v_idx, 0))
        b2 = ((v_idx, 1), (j, pj))
        self.updated_bonds += [b1, b2]
        self.virtual_lengths[_canonical_port_edge(*b1)] = seg_len
        self.virtual_lengths[_canonical_port_edge(*b2)] = seg_len

    def _order_bonds_and_lengths(self):
        canon = [_canonical_port_edge(a, b) for (a, b) in self.updated_bonds]
        canon.sort(key=lambda ab: (ab[0][0], ab[0][1], ab[1][0], ab[1][1]))
        ordered = OrderedDict()
        for a, b in canon:
            key = _canonical_port_edge(a, b)
            if key not in self.virtual_lengths:
                raise KeyError(f"Missing virtual length for bond {key}.")
            if key not in ordered:
                ordered[key] = self.virtual_lengths[key]
        for k, v in self.virtual_lengths.items():
            if k not in ordered:
                ordered[k] = v
        self.updated_bonds = canon
        self.virtual_lengths = dict(ordered)

    # ==================================================================
    # Save helpers
    # ==================================================================

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
