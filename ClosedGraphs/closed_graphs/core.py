#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreMixin — __init__, virtualisation helpers, geometry, save-path logic.

Notes
-----
- ``_compute_group_index`` uses ``ng(k) = neff(k) + k * d(neff)/dk``.
- ``save_path`` resolution order: ``save_path`` > ``base_dir`` >
  ``sys.argv[0]`` > ``cwd``, so Spyder / IPython runs always save next to the
  calling script when ``base_dir`` is passed.
"""

import os
import sys
import copy
import datetime
import inspect
import hashlib
import warnings
import numpy as np
import networkx as nx
from collections import defaultdict, OrderedDict
from pathlib import Path
import numdifftools as nd


class CoreMixin:
    """
    Mixin providing graph construction, virtualisation, and save-path logic.
    """

    def __init__(self, vertex_positions, port_map,
                 lengths=None,
                 vertex_matrices=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 name="MyClosedGraph",
                 save_path=None,
                 base_dir=None):
        """
        Initialize a closed quantum graph (no external leads).
        Sets geometry, bond lengths, and vertex scattering matrices,
        while resolving loops and multiple bonds with virtual vertices.
        """
        # --- Basic attributes ---
        self.vertex_positions = list(vertex_positions)
        self.port_map         = port_map
        self.virtual_port_map = copy.deepcopy(port_map)
        self.lengths          = lengths
        self.n_vertices       = len(vertex_positions)
        self.wavelength_range = wavelength_range
        self.neff             = neff
        self.ng               = ng if ng is not None else self._compute_group_index()

        self._init_spacing_sequence(virtual_vertex_spacing)
        self.virtual_vertex_spacing = virtual_vertex_spacing

        # --- Compute ORIGINAL lengths if not provided ---
        if self.lengths is None:
            self.lengths = {}
            for v_idx, ports in enumerate(self.port_map):
                for p_idx, bond in enumerate(ports):
                    j, p2 = bond
                    if v_idx < j:
                        x1, y1 = self.vertex_positions[v_idx]
                        x2, y2 = self.vertex_positions[j]
                        dist = np.hypot(x2 - x1, y2 - y1)
                        self.lengths[((v_idx, p_idx), (j, p2))] = dist

        self.total_length    = sum(self.lengths.values())
        self.virtual_lengths = copy.deepcopy(self.lengths)

        # STEP 1: Build bond list
        self.bonds     = []
        self.valencies = []

        for i, ports in enumerate(self.port_map):
            self.valencies.append(len(ports))
            for pi, conn in enumerate(ports):
                j, pj = conn
                if (j, pj, i, pi) not in self.bonds:
                    self.bonds.append((i, pi, j, pj))

        self.n_bonds = len(self.bonds)

        # STEP 2: Group by vertex pair
        vertex_pair_bonds = defaultdict(list)
        for (i, pi, j, pj) in self.bonds:
            vpair = tuple(sorted([i, j]))
            vertex_pair_bonds[vpair].append(((i, pi), (j, pj)))

        # STEP 3: Break multiple bonds / loops
        self.virtual_vertices = []
        self.updated_bonds    = []

        for (i, j), bond_list in vertex_pair_bonds.items():
            if i == j:
                for bond in bond_list:
                    self._break_complex_bond(bond, loop=True, index=0, n_total=len(bond_list))
            elif len(bond_list) == 1:
                self.updated_bonds.append(bond_list[0])
            else:
                group_spacing = self._next_group_spacing()
                for k, bond in enumerate(bond_list):
                    self._break_complex_bond(
                        bond, loop=False, index=k,
                        n_total=len(bond_list), group_spacing=group_spacing
                    )

        self._order_bonds_and_lengths()

        # STEP 4: Build networkx graph
        self.G = nx.Graph()
        self.vertex_scattering_matrix = {}

        for i, pos in enumerate(self.vertex_positions):
            self.G.add_node(i, pos=pos)

        for (a, pa), (b, pb) in self.updated_bonds:
            key = tuple(sorted([(a, pa), (b, pb)]))
            self.G.add_edge(a, b, length=self.virtual_lengths[key])

        # STEP 5: Assign vertex scattering matrices
        for v in range(len(self.vertex_positions)):
            if v in self.virtual_vertices:
                self.vertex_scattering_matrix[v] = np.array([[0, 1], [1, 0]])
            else:
                if isinstance(vertex_matrices, (np.ndarray, list)):
                    if isinstance(vertex_matrices, np.ndarray):
                        self.vertex_scattering_matrix[v] = vertex_matrices
                    else:
                        self.vertex_scattering_matrix[v] = vertex_matrices[v]
                elif callable(vertex_matrices):
                    self.vertex_scattering_matrix[v] = vertex_matrices
                elif isinstance(vertex_matrices, dict):
                    self.vertex_scattering_matrix[v] = vertex_matrices[v]
                else:
                    raise ValueError(
                        "vertex_matrices must be a dict, list, array, or callable."
                    )

        # STEP 6: Metadata
        self.n_vertices_total    = len(self.vertex_positions)
        self.n_virtual_vertices  = len(self.virtual_vertices)
        self.n_internal_bonds    = len(self.updated_bonds)
        self.n_directed_bonds    = 2 * self.n_internal_bonds

        # STEP 7: Directed bonds and lengths
        self.directed_bonds      = []
        self.directed_bond_index = {}
        self.directed_lengths    = {}

        for idx, (a, b) in enumerate(self.updated_bonds):
            self.directed_bonds.append((a, b))
            self.directed_bond_index[(a, b)] = idx
            self.directed_bonds.append((b, a))
            self.directed_bond_index[(b, a)] = idx + self.n_internal_bonds

            key = tuple(sorted([a, b]))
            L = self.virtual_lengths[key]
            self.directed_lengths[(a, b)] = L
            self.directed_lengths[(b, a)] = L

        # STEP 8: Name & save path
        self.name = name

        if save_path is not None:
            self.save_path = Path(save_path)
        else:
            if base_dir is not None:
                _base = Path(base_dir).resolve()
            else:
                argv0 = Path(sys.argv[0]).resolve()
                if argv0.suffix == ".py" and argv0.exists():
                    _base = argv0.parent
                else:
                    _base = Path.cwd()
            graph_dir = _base / self.name
            coupling_info = getattr(self, "coupling_info", None)
            leaf = self._build_save_dir_name(coupling_info)
            self.save_path = graph_dir / leaf

        self.save_path.mkdir(parents=True, exist_ok=True)

        # STEP 9: Save parameters
        self._save_parameters()

    # =========================================================================
    # Spacing helpers
    # =========================================================================

    def _init_spacing_sequence(self, spacing):
        self._spacing_sequence = None
        self._spacing_ptr      = 0

        if spacing is None:
            return
        if isinstance(spacing, (int, float)):
            return

        seq = list(spacing)
        if len(seq) == 0:
            return
        self._spacing_sequence = [float(x) for x in seq]

    def _next_group_spacing(self):
        if self._spacing_sequence:
            val = self._spacing_sequence[self._spacing_ptr % len(self._spacing_sequence)]
            self._spacing_ptr += 1
            return val
        return self.virtual_vertex_spacing

    def _break_complex_bond(self, bond, loop=False, index=0, n_total=1, group_spacing=None):
        """
        Replace a multiple bond or self-loop with virtual vertices.
        Mutates virtual_lengths, virtual_port_map, vertex_positions,
        virtual_vertices, updated_bonds.
        """
        (i, pi), (j, pj) = bond
        xi, yi = self.vertex_positions[i]
        xj, yj = self.vertex_positions[j]

        dx, dy   = xj - xi, yj - yi
        distance = np.hypot(dx, dy)
        norm_dir = np.array([dx, dy]) / distance if distance > 0 else np.array([1.0, 0.0])
        perp     = np.array([-norm_dir[1], norm_dir[0]])

        if group_spacing is not None:
            spacing = float(group_spacing)
        elif isinstance(self.virtual_vertex_spacing, (int, float)):
            spacing = float(self.virtual_vertex_spacing)
        else:
            spacing = 0.3 * distance

        # --- LOOP (i == j) ---
        if loop:
            L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)

            self.virtual_port_map[i][pi] = None
            self.virtual_port_map[i][pj] = None

            center = np.array([xi, yi])
            offset = L_total / 4.0 / np.sqrt(2)

            v_positions = [
                (-offset,  offset),
                ( 0.0,     offset * 2),
                ( offset,  offset),
            ]

            v_idx_list = []
            for (dx_v, dy_v) in v_positions:
                v_idx = len(self.vertex_positions)
                self.vertex_positions.append((center[0] + dx_v, center[1] + dy_v))
                self.valencies.append(2)
                self.virtual_vertices.append(v_idx)
                self.virtual_port_map.append([None, None])
                v_idx_list.append(v_idx)

            v1, v2, v3 = v_idx_list

            self.virtual_port_map[i][pi]  = (v1, 0)
            self.virtual_port_map[i][pj]  = (v3, 1)
            self.virtual_port_map[v1][0]  = (i,  pi)
            self.virtual_port_map[v1][1]  = (v2, 0)
            self.virtual_port_map[v2][0]  = (v1, 1)
            self.virtual_port_map[v2][1]  = (v3, 0)
            self.virtual_port_map[v3][0]  = (v2, 1)
            self.virtual_port_map[v3][1]  = (i,  pj)

            segment_length = L_total / 4.0
            new_bonds = [
                ((i,  pi), (v1, 0)),
                ((v1, 1),  (v2, 0)),
                ((v2, 1),  (v3, 0)),
                ((v3, 1),  (i,  pj)),
            ]
            for (a, b) in new_bonds:
                self.updated_bonds.append((a, b))
                self.virtual_lengths[tuple(sorted([a, b]))] = segment_length
            return

        # --- MULTIPLE BONDS (i ≠ j) ---
        L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)

        midpoint = (np.array([xi, yi]) + np.array([xj, yj])) / 2
        offset   = (index - (n_total - 1) / 2) * spacing
        pt       = midpoint + offset * perp

        v_idx = len(self.vertex_positions)
        self.vertex_positions.append(tuple(pt))
        self.valencies.append(2)
        self.virtual_vertices.append(v_idx)
        self.virtual_port_map.append([None, None])

        self.virtual_port_map[i][pi] = None
        self.virtual_port_map[j][pj] = None

        self.virtual_port_map[i][pi]    = (v_idx, 0)
        self.virtual_port_map[v_idx][0] = (i,  pi)
        self.virtual_port_map[j][pj]    = (v_idx, 1)
        self.virtual_port_map[v_idx][1] = (j,  pj)

        segment_length = L_total / 2.0
        self.updated_bonds.append(((i, pi), (v_idx, 0)))
        self.updated_bonds.append(((v_idx, 1), (j, pj)))
        self.virtual_lengths[tuple(sorted([(i, pi), (v_idx, 0)]))]  = segment_length
        self.virtual_lengths[tuple(sorted([(v_idx, 1), (j, pj)]))]  = segment_length

    def _order_bonds_and_lengths(self):
        """Canonicalize and globally sort updated_bonds and virtual_lengths."""

        def canon_edge(a, b):
            return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)

        canon_bonds = [canon_edge(a, b) for (a, b) in self.updated_bonds]
        canon_bonds.sort(key=lambda ab: (ab[0][0], ab[0][1], ab[1][0], ab[1][1]))

        ordered_lengths = OrderedDict()
        for a, b in canon_bonds:
            key = tuple(sorted([a, b]))
            if key not in self.virtual_lengths:
                raise KeyError(f"Missing virtual length for bond {key}.")
            if key not in ordered_lengths:
                ordered_lengths[key] = self.virtual_lengths[key]

        for k, v in self.virtual_lengths.items():
            if k not in ordered_lengths:
                ordered_lengths[k] = v

        self.updated_bonds   = canon_bonds
        self.virtual_lengths = dict(ordered_lengths)

    # =========================================================================
    # Group index
    # =========================================================================

    def _compute_group_index(self):
        """
        Compute ng(k) from neff(k).

        Returns
        -------
        float or callable
            Constant if neff is constant; callable ng(k) otherwise.

        Notes
        -----
        ng(k) = neff(k) + k * d(neff)/d(k)
        """
        if not callable(self.neff):
            return float(self.neff)

        def ng_func(k):
            d_neff = nd.Derivative(self.neff)(k)
            return self.neff(k) + k * d_neff   # fixed sign

        return ng_func

    # =========================================================================
    # Save-path helpers
    # =========================================================================

    def _build_save_dir_name(self, coupling_info=None):
        """Build the deterministic save directory leaf name."""
        info_str = f"Nv={self.n_vertices_total}_Nb={self.n_internal_bonds}"

        if callable(self.neff):
            neff_str = self.neff.__name__ if hasattr(self.neff, "__name__") else "neff_func"
        else:
            neff_str = f"neff={float(self.neff):.5f}"

        lamb_min, lamb_max = self.wavelength_range
        lamb_str = f"lambda={lamb_min:.3f}-{lamb_max:.3f}"

        hash_input  = str(self.port_map) + str(self.lengths)
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        if coupling_info:
            return f"{info_str}_{neff_str}_{lamb_str}_{coupling_info}_{hash_digest}"
        return f"{info_str}_{neff_str}_{lamb_str}_{hash_digest}"

    def _save_parameters(self):
        """Save a human-readable description of the graph to graph_parameters.txt."""
        filepath = os.path.join(self.save_path, "graph_parameters.txt")
        with open(filepath, "w") as f:
            f.write("### Closed Quantum Graph Parameters ###\n")
            f.write(f"Graph Name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n")
            f.write(f"Total vertices: {self.n_vertices_total} "
                    f"(virtual: {self.n_virtual_vertices})\n")
            f.write(f"Internal bonds: {self.n_internal_bonds}\n")
            f.write(f"Wavelength range (vacuum): {self.wavelength_range[0]:.3f} – "
                    f"{self.wavelength_range[1]:.3f} µm\n\n")

            if callable(self.neff):
                f.write("Effective index: function of k\n")
                try:
                    f.write(inspect.getsource(self.neff) + "\n")
                except Exception:
                    f.write("(source code unavailable)\n")
            else:
                f.write(f"Effective index: {float(self.neff):.4f}\n")

            if callable(self.ng):
                f.write("\nGroup index: function of k (derived from neff)\n")
            else:
                f.write(f"\nGroup index: {float(self.ng):.4f}\n")

            f.write("\n### Vertex Scattering Matrices ###\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v}: function of k\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source code unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency {self.valencies[v]}):\n")
                    np.savetxt(f, sigma, fmt="%-8.4f", delimiter=" ", newline="\n")
                    f.write("\n")

            f.write("\n### Port Map (raw, before virtualization) ###\n")
            for i, ports in enumerate(self.port_map):
                f.write(f"Vertex {i} ports: {ports}\n")

            f.write("\n### Bond Lengths (raw, before virtualization) ###\n")
            for (a, b), L in self.lengths.items():
                f.write(f"{a} — {b} : {L:.6f} µm\n")

            f.write("\n### Virtualization Info ###\n")
            f.write(f"Virtual vertices: {self.virtual_vertices}\n")
            f.write(f"Updated bonds (after virtualization): {self.updated_bonds}\n")

        print(f"Graph parameters saved to {filepath}")
