#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreMixin — __init__, virtualisation helpers, geometry, save-path logic.
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
                 internal_matrix=None, external_matrix=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 lead_orientations=None,
                 name="MyOpenGraph",
                 save_path=None,
                 base_dir=None):
        """
        Initialize an open quantum graph (with possible external leads).
        Sets geometry, bond lengths, and vertex scattering matrices,
        while resolving loops and multiple bonds with virtual vertices.
        """
        # --- Basic attributes ---
        self.vertex_positions = list(vertex_positions)
        self.port_map         = port_map
        self.virtual_port_map = copy.deepcopy(self.port_map)
        self.lengths          = lengths
        self.n_vertices       = len(vertex_positions)
        self.wavelength_range = wavelength_range
        self.neff             = neff
        self.ng               = ng if ng is not None else self._compute_group_index()
        self.lead_orientations = lead_orientations

        self._init_spacing_sequence(virtual_vertex_spacing)
        self.virtual_vertex_spacing = virtual_vertex_spacing

        # --- Compute ORIGINAL lengths if not provided ---
        if self.lengths is None:
            self.lengths = {}
            for v_idx, ports in enumerate(self.port_map):
                for p_idx, bond in enumerate(ports):
                    if bond is not None:
                        v2, p2 = bond
                        if v_idx < v2:
                            x1, y1 = self.vertex_positions[v_idx]
                            x2, y2 = self.vertex_positions[v2]
                            dist = np.hypot(x2 - x1, y2 - y1)
                            self.lengths[((v_idx, p_idx), (v2, p2))] = dist

        self.total_length  = sum(self.lengths.values())
        self.virtual_lengths = copy.deepcopy(self.lengths)

        # STEP 1: Build bond list and lead list
        self.bonds    = []
        self.leads    = []
        self.valencies = []

        for i, ports in enumerate(self.port_map):
            self.valencies.append(len(ports))
            for pi, conn in enumerate(ports):
                if conn is None:
                    self.leads.append((i, pi))
                else:
                    j, pj = conn
                    if i == j and pi == pj:
                        continue
                    if (j, pj, i, pi) not in self.bonds:
                        self.bonds.append((i, pi, j, pj))

        self.n_bonds = len(self.bonds)
        self.n_leads = len(self.leads)

        # STEP 2: Group by vertex pair (i, j)
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
                        n_total=len(bond_list), group_spacing=group_spacing,
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
                continue
            if vertex_matrices is not None and v in vertex_matrices:
                self.vertex_scattering_matrix[v] = vertex_matrices[v]
            else:
                is_external = any((v, p) in self.leads for p in range(self.valencies[v]))
                if is_external:
                    if external_matrix is None:
                        raise ValueError(f"No external_matrix provided for external vertex {v}")
                    self.vertex_scattering_matrix[v] = external_matrix
                else:
                    if internal_matrix is None:
                        raise ValueError(f"No internal_matrix provided for internal vertex {v}")
                    self.vertex_scattering_matrix[v] = internal_matrix

        # STEP 6: Metadata
        self.n_vertices_total    = len(self.vertex_positions)
        self.n_virtual_vertices  = len(self.virtual_vertices)
        self.n_internal_bonds    = len(self.updated_bonds)
        self.n_directed_bonds    = 2 * self.n_internal_bonds

        # STEP 7: Build directed bonds
        self.directed_bonds      = []
        self.directed_bond_index = {}
        self.directed_lengths    = {}

        for idx, (a, b) in enumerate(self.updated_bonds):
            self.directed_bonds.append((a, b))
            self.directed_bond_index[(a, b)] = idx
            self.directed_bonds.append((b, a))
            self.directed_bond_index[(b, a)] = idx + self.n_internal_bonds
            key = tuple(sorted([a, b]))
            if key not in self.virtual_lengths:
                raise ValueError(f"No virtual length provided for bond {key}")
            L = self.virtual_lengths[key]
            self.directed_lengths[(a, b)] = L
            self.directed_lengths[(b, a)] = L

        # STEP 8: Lead orientations
        if hasattr(self, "_generate_lead_orientations"):
            self._generate_lead_orientations()

        # STEP 9: Name & save directory
        # Priority: explicit save_path > explicit base_dir > sys.argv[0] > cwd
        self.name = name

        if save_path is None:
            if base_dir is not None:
                _base = Path(base_dir).resolve()
            else:
                argv0 = Path(sys.argv[0]).resolve()
                if argv0.suffix == '.py' and argv0.exists():
                    _base = argv0.parent
                else:
                    _base = Path.cwd()
            graph_dir    = _base / self.name
            coupling_info = getattr(self, "coupling_info", None)
            leaf = self._build_save_dir_name(coupling_info)
            self.save_path = graph_dir / leaf
        else:
            self.save_path = Path(save_path)

        self.save_path.mkdir(parents=True, exist_ok=True)

        # STEP 10: Save key parameters to text
        self._save_parameters()

    # -------------------------------------------------------------------------
    # Spacing helpers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Virtual vertex construction
    # -------------------------------------------------------------------------

    def _break_complex_bond(self, bond, loop=False, index=0, n_total=1, group_spacing=None):
        """Replace a multiple bond or self-loop with virtual vertices."""
        (i, pi), (j, pj) = bond
        xi, yi = self.vertex_positions[i]
        xj, yj = self.vertex_positions[j]
        dx, dy = xj - xi, yj - yi
        distance = np.hypot(dx, dy)
        norm_dir = np.array([dx, dy]) / distance if distance > 0 else np.array([1.0, 0.0])
        perp = np.array([-norm_dir[1], norm_dir[0]])

        if group_spacing is not None:
            spacing = float(group_spacing)
        elif isinstance(self.virtual_vertex_spacing, (int, float)):
            spacing = float(self.virtual_vertex_spacing)
        else:
            spacing = 0.3 * distance

        if loop:
            square_side = 100.0
            self.virtual_port_map[i][pi] = None
            self.virtual_port_map[i][pj] = None
            L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)
            offset = square_side / np.sqrt(2.0)
            v_positions = [
                (xi + offset, yi + offset),
                (xi,          yi + 2.0 * offset),
                (xi - offset, yi + offset),
            ]
            idxs = []
            for pt in v_positions:
                v_idx = len(self.vertex_positions)
                self.vertex_positions.append(tuple(pt))
                self.valencies.append(2)
                self.virtual_vertices.append(v_idx)
                self.virtual_port_map.append([None, None])
                idxs.append(v_idx)
            v1, v2, v3 = idxs
            self.virtual_port_map[i][pj]  = (v1, 0)
            self.virtual_port_map[i][pi]  = (v3, 1)
            self.virtual_port_map[v1][0]  = (i, pj)
            self.virtual_port_map[v1][1]  = (v2, 0)
            self.virtual_port_map[v2][0]  = (v1, 1)
            self.virtual_port_map[v2][1]  = (v3, 0)
            self.virtual_port_map[v3][0]  = (v2, 1)
            self.virtual_port_map[v3][1]  = (i, pi)
            segment_length = L_total / 4.0
            new_bonds = [
                ((i, pj), (v1, 0)),
                ((v1, 1), (v2, 0)),
                ((v2, 1), (v3, 0)),
                ((v3, 1), (i, pi)),
            ]
            for a, b in new_bonds:
                self.updated_bonds.append((a, b))
                self.virtual_lengths[tuple(sorted([a, b]))] = segment_length
            return

        midpoint = (np.array([xi, yi]) + np.array([xj, yj])) / 2
        offset_v = (index - (n_total - 1) / 2) * spacing
        pt = midpoint + offset_v * perp
        v_idx = len(self.vertex_positions)
        self.vertex_positions.append(tuple(pt))
        self.valencies.append(2)
        self.virtual_vertices.append(v_idx)
        self.virtual_port_map.append([None, None])
        self.virtual_port_map[i][pi]      = None
        self.virtual_port_map[j][pj]      = None
        self.virtual_port_map[i][pi]      = (v_idx, 0)
        self.virtual_port_map[v_idx][0]   = (i, pi)
        self.virtual_port_map[j][pj]      = (v_idx, 1)
        self.virtual_port_map[v_idx][1]   = (j, pj)
        L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)
        segment_length = L_total / 2.0
        self.updated_bonds.append(((i, pi), (v_idx, 0)))
        self.updated_bonds.append(((v_idx, 1), (j, pj)))
        self.virtual_lengths[tuple(sorted([(i, pi), (v_idx, 0)]))] = segment_length
        self.virtual_lengths[tuple(sorted([(v_idx, 1), (j, pj)]))] = segment_length

    def _order_bonds_and_lengths(self):
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

    # -------------------------------------------------------------------------
    # Lead orientations
    # -------------------------------------------------------------------------

    def _generate_lead_orientations(self):
        self.lead_orientations_rad = {}
        if self.lead_orientations is not None:
            for (v, p) in self.leads:
                if (v, p) not in self.lead_orientations:
                    raise ValueError(f"Missing orientation for lead (vertex {v}, port {p})")
                self.lead_orientations_rad[(v, p)] = np.deg2rad(self.lead_orientations[(v, p)])
            return
        self.lead_orientations = {}
        leads_by_vertex = defaultdict(list)
        for (v, p) in self.leads:
            leads_by_vertex[v].append(p)
        for v, ports in leads_by_vertex.items():
            connected_vecs = []
            for port_idx, conn in enumerate(self.virtual_port_map[v]):
                if conn is not None:
                    neighbor, n_port = conn
                    vec = np.array(self.vertex_positions[neighbor]) - np.array(self.vertex_positions[v])
                    connected_vecs.append(vec)
            if len(connected_vecs) == 0:
                base_angle = 0.0
            else:
                avg_vec    = np.mean(connected_vecs, axis=0)
                base_vec   = -avg_vec
                base_angle = np.degrees(np.arctan2(base_vec[1], base_vec[0]))
            n_leads = len(ports)
            if n_leads == 1:
                self.lead_orientations[(v, ports[0])]     = base_angle
                self.lead_orientations_rad[(v, ports[0])] = np.deg2rad(base_angle)
            else:
                spread      = 120 / (n_leads - 1)
                start_angle = base_angle - spread * (n_leads - 1) / 2
                for i, port in enumerate(ports):
                    angle = start_angle + i * spread
                    self.lead_orientations[(v, port)]     = angle
                    self.lead_orientations_rad[(v, port)] = np.deg2rad(angle)

    # -------------------------------------------------------------------------
    # Group index
    # -------------------------------------------------------------------------

    def _compute_group_index(self):
        if not callable(self.neff):
            return float(self.neff)
        def ng_func(k):
            d_neff = nd.Derivative(self.neff)(k)
            return self.neff(k) + k * d_neff
        return ng_func

    # -------------------------------------------------------------------------
    # Save-directory naming
    # -------------------------------------------------------------------------

    def _build_save_dir_name(self, coupling_info=None):
        info_str = f"Nv={self.n_vertices_total}_Nb={self.n_internal_bonds}_Nl={self.n_leads}"
        if callable(self.neff):
            neff_str = self.neff.__name__ if hasattr(self.neff, "__name__") else "neff_func"
        else:
            neff_str = f"neff={float(self.neff):.3f}"
        lamb_min, lamb_max = self.wavelength_range
        lamb_str   = f"lambda={lamb_min:.3f}-{lamb_max:.3f}"
        hash_input = str(self.port_map) + str(self.lengths)
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        if coupling_info:
            return f"{info_str}_{neff_str}_{lamb_str}_{coupling_info}_{hash_digest}"
        else:
            return f"{info_str}_{neff_str}_{lamb_str}_{hash_digest}"

    # -------------------------------------------------------------------------
    # Parameter file
    # -------------------------------------------------------------------------

    def _save_parameters(self):
        filepath = os.path.join(self.save_path, "graph_parameters.txt")
        with open(filepath, "w") as f:
            f.write("### Open Quantum Graph Parameters ###\n")
            f.write(f"Graph Name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n")
            f.write(f"Save path: {self.save_path}\n")
            f.write(f"Wavelength range (vacuum): {self.wavelength_range[0]:.3f} – "
                    f"{self.wavelength_range[1]:.3f} µm\n\n")
            f.write(f"Total vertices: {self.n_vertices_total} "
                    f"(virtual: {self.n_virtual_vertices})\n")
            f.write(f"Internal bonds: {self.n_internal_bonds}\n")
            f.write(f"Directed bonds: {self.n_directed_bonds}\n")
            f.write(f"Leads: {getattr(self, 'n_leads', len(getattr(self, 'leads', [])))}\n\n")
            hash_input  = str(self.port_map) + str(self.lengths)
            hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            f.write(f"Config hash (port_map + lengths): {hash_digest}\n\n")

            f.write("### Optical Parameters ###\n")
            if callable(self.neff):
                f.write("Effective index neff(k): function of k\n")
                try:
                    f.write(inspect.getsource(self.neff) + "\n")
                except Exception:
                    f.write("(source code unavailable)\n")
            else:
                f.write(f"Effective index neff: {float(self.neff):.6f}\n")
            if callable(self.ng):
                f.write("Group index ng(k): function of k\n")
                try:
                    f.write(inspect.getsource(self.ng) + "\n")
                except Exception:
                    f.write("(source code unavailable)\n")
            else:
                f.write(f"Group index ng: {float(self.ng):.6f}\n")
            f.write("\n")

            f.write("### Vertex Scattering Matrices ###\n")
            if hasattr(self, "internal_matrix_default"):
                internal_matrix = getattr(self, "internal_matrix_default")
                f.write("Internal (bulk) vertex matrix:\n")
                try:
                    if callable(internal_matrix):
                        f.write("function of k\n")
                        f.write(inspect.getsource(internal_matrix) + "\n")
                    else:
                        np.savetxt(f, np.asarray(internal_matrix), fmt="%-10.6f")
                        f.write("\n")
                except Exception:
                    f.write("(unavailable)\n")
            if hasattr(self, "external_matrix_default"):
                external_matrix = getattr(self, "external_matrix_default")
                f.write("External (lead-connected) vertex matrix:\n")
                try:
                    if callable(external_matrix):
                        f.write("function of k\n")
                        f.write(inspect.getsource(external_matrix) + "\n")
                    else:
                        np.savetxt(f, np.asarray(external_matrix), fmt="%-10.6f")
                        f.write("\n")
                except Exception:
                    f.write("(unavailable)\n")
            f.write("\nPer-vertex matrices:\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v} (valency {self.valencies[v]}): function of k\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source code unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency {self.valencies[v]}):\n")
                    try:
                        np.savetxt(f, np.asarray(sigma), fmt="%-10.6f")
                    except Exception:
                        f.write(str(sigma))
                    f.write("\n")
            f.write("\n")

            f.write("### Port Map (raw, as provided) ###\n")
            for i, ports in enumerate(self.port_map):
                f.write(f"Vertex {i} ports: {ports}\n")
            f.write("\n")

            f.write("### Leads ###\n")
            if hasattr(self, "leads") and self.leads:
                for (i, p) in self.leads:
                    f.write(f"Lead at (vertex {i}, port {p})\n")
            else:
                f.write("(none)\n")
            f.write("\n")

            f.write("### Lead Orientations (deg) ###\n")
            if isinstance(getattr(self, "lead_orientations", None), dict) and self.lead_orientations:
                for key, ang in self.lead_orientations.items():
                    f.write(f"{key}: {ang}\n")
            else:
                f.write("(none)\n")
            f.write("\n")

            f.write("### Bond Lengths (raw, before virtualization) ###\n")
            if self.lengths:
                for (a, b), L in self.lengths.items():
                    f.write(f"{a} — {b} : {L:.6f} µm\n")
            else:
                f.write("(none)\n")
            f.write("\n")

            f.write("### Virtualization Info ###\n")
            f.write(f"Virtual vertices: {getattr(self, 'virtual_vertices', [])}\n")
            f.write("Updated bonds (after virtualization):\n")
            for bond in getattr(self, "updated_bonds", []):
                f.write(f"  {bond}\n")
            f.write("\nVirtual lengths (after virtualization):\n")
            if getattr(self, "virtual_lengths", None):
                for key, L in self.virtual_lengths.items():
                    f.write(f"{key} : {L:.6f} µm\n")
            else:
                f.write("(none)\n")
            f.write("\n")

            f.write("### Directed Bonds (indexing) ###\n")
            try:
                for db in self.directed_bonds:
                    idx = self.directed_bond_index.get(db, None)
                    L   = self.directed_lengths.get(db, None)
                    if L is not None:
                        f.write(f"{db} -> idx {idx}, L={L:.6f} µm\n")
                    else:
                        f.write(f"{db} -> idx {idx}, L=(missing)\n")
            except Exception:
                f.write("(unavailable)\n")

        print(f"Graph parameters saved to {filepath}")
