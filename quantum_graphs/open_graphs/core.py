#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreMixin — __init__, virtualisation helpers, geometry, save-path logic.
"""

import os
import datetime
import inspect
import numpy as np
from collections import defaultdict

from ..core.base import GraphBase
from ..core.optics import group_index_function
from ..core.persistence import legacy_config_hash


class CoreMixin(GraphBase):
    _allow_leads = True

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
        self.wavelength_range = wavelength_range
        self.neff             = neff
        self.ng               = ng if ng is not None else self._compute_group_index()
        self.lead_orientations = lead_orientations

        self._setup_topology(vertex_positions, port_map, lengths, virtual_vertex_spacing)

        matrices = {}
        for v in range(self.n_vertices):
            if vertex_matrices is not None and v in vertex_matrices:
                matrices[v] = vertex_matrices[v]
            elif any(conn is None for conn in port_map[v]):
                if external_matrix is None:
                    raise ValueError(f"No external_matrix provided for external vertex {v}")
                matrices[v] = external_matrix
            else:
                if internal_matrix is None:
                    raise ValueError(f"No internal_matrix provided for internal vertex {v}")
                matrices[v] = internal_matrix
        self._setup_vertex_matrices(matrices, self._k_samples())

        # STEP 8: Lead orientations
        if hasattr(self, "_generate_lead_orientations"):
            self._generate_lead_orientations()

        # STEP 9: Name, save directory and parameter file
        leaf = self._build_save_dir_name(getattr(self, "coupling_info", None))
        self._setup_save_path(name, leaf, save_path, base_dir)

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
        """ng(k) = neff(k) + k dneff/dk (float if neff is constant)."""
        return group_index_function(self.neff)

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
        hash_digest = legacy_config_hash(self.port_map, self.lengths)
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
            f.write(f"Config hash (port_map + lengths): "
                    f"{legacy_config_hash(self.port_map, self.lengths)}\n")
            f.write(f"Physics fingerprint: {self.fingerprint}\n\n")

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

        self._log(f"Graph parameters saved to {filepath}")
