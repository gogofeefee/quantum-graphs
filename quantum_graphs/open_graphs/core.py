"""
CoreMixin — __init__, lead orientations, save-path logic.
"""

import os
import datetime
import inspect
import numpy as np
from collections import defaultdict

from ..core.base import GraphBase
from ..core.optics import group_index_function
from ..core.persistence import topology_hash
from ..core.vertices import resolve_vertex_matrices


class CoreMixin(GraphBase):
    """
    Mixin providing graph construction and save-path logic.
    """

    _allow_leads = True

    def __init__(self, vertex_positions, bonds, leads, *,
                 vertex_matrices,
                 wavelength_range,
                 access_matrices=None,
                 neff=1.0,
                 ng=None,
                 bond_spacing=None,
                 lead_orientations=None,
                 name="MyOpenGraph",
                 save_path=None,
                 base_dir=None):
        """
        Initialize an open quantum graph.

        ``bonds``: ``[(i, p, j, q[, L]), ...]``; ``leads``: ``[(v, p), ...]``,
        numbered in this order in the S-matrix (see
        :mod:`quantum_graphs.core.topology`). ``vertex_matrices``: dict, list,
        array or callable σ(k), as for closed graphs; ``access_matrices``
        (same forms) overrides it on the vertices that carry a lead.
        ``lead_orientations`` {(v, p): degrees} is for drawing only (generated
        if None).
        """
        self.wavelength_range = self._check_wavelength_range(wavelength_range)
        self.neff             = neff
        self.ng               = ng if ng is not None else self._compute_group_index()
        self.lead_orientations = lead_orientations

        self._setup_topology(vertex_positions, bonds, leads)
        self.bond_spacing = bond_spacing

        matrices = resolve_vertex_matrices(vertex_matrices, self.n_vertices)
        if access_matrices is not None:
            access = resolve_vertex_matrices(access_matrices, self.n_vertices)
            for v in {v for v, _ in self.topology.leads}:
                matrices[v] = access[v]
        self._setup_vertex_matrices(matrices, self._k_samples())

        self._generate_lead_orientations()

        leaf = self._build_save_dir_name(getattr(self, "coupling_info", None))
        self._setup_save_path(name, leaf, save_path, base_dir)

    # -------------------------------------------------------------------------
    # Lead orientations
    # -------------------------------------------------------------------------

    def _generate_lead_orientations(self):
        """Check the given orientations, or point the leads of each vertex away
        from its neighbours (spread over 120° if there are several)."""
        t = self.topology
        if self.lead_orientations is not None:
            missing = [lead for lead in t.leads if lead not in self.lead_orientations]
            if missing:
                raise ValueError(f"Missing orientation for leads {missing}")
            return
        self.lead_orientations = {}
        leads_by_vertex = defaultdict(list)
        for (v, p) in t.leads:
            leads_by_vertex[v].append(p)
        for v, ports in leads_by_vertex.items():
            vecs = [t.positions[t.bonds[b][2] if t.bonds[b][0] == v else t.bonds[b][0]]
                    - t.positions[v] for b in t.vertex_bonds(v)]
            avg_vec = np.mean(vecs, axis=0) if vecs else np.zeros(2)
            base_angle = (np.degrees(np.arctan2(-avg_vec[1], -avg_vec[0]))
                          if np.hypot(*avg_vec) > 0 else 0.0)
            if len(ports) == 1:
                self.lead_orientations[(v, ports[0])] = base_angle
            else:
                spread      = 120 / (len(ports) - 1)
                start_angle = base_angle - spread * (len(ports) - 1) / 2
                for i, port in enumerate(ports):
                    self.lead_orientations[(v, port)] = start_angle + i * spread

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
        info_str = f"Nv={self.n_vertices}_Nb={self.n_bonds}_Nl={self.n_leads}"
        if callable(self.neff):
            neff_str = self.neff.__name__ if hasattr(self.neff, "__name__") else "neff_func"
        else:
            neff_str = f"neff={float(self.neff):.3f}"
        lamb_min, lamb_max = self.wavelength_range
        lamb_str   = f"lambda={lamb_min:.3f}-{lamb_max:.3f}"
        hash_digest = topology_hash(self.topology)
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
            f.write(f"Vertices: {self.n_vertices}\n")
            f.write(f"Bonds: {self.n_bonds}\n")
            f.write(f"Leads: {self.n_leads}\n\n")
            f.write(f"Topology hash: {topology_hash(self.topology)}\n")
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
            f.write("Per-vertex matrices:\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v} (valency {self.topology.valency[v]}): function of k\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source code unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency {self.topology.valency[v]}):\n")
                    try:
                        np.savetxt(f, np.asarray(sigma), fmt="%-10.6f")
                    except Exception:
                        f.write(str(sigma))
                    f.write("\n")
            f.write("\n")

            f.write("### Vertex positions (µm) ###\n")
            for v, (x, y) in enumerate(self.topology.positions):
                f.write(f"Vertex {v}: ({x:.6f}, {y:.6f})\n")
            f.write("\n")

            f.write("### Bonds (i, p, j, q, L in µm) ###\n")
            for b, (i, p, j, q, L) in enumerate(self.topology.bonds):
                f.write(f"{b}: ({i}, {p}) — ({j}, {q}) : {L:.6f}\n")
            f.write("\n")

            f.write("### Leads (vertex, port) and drawing orientation (deg) ###\n")
            for m, lead in enumerate(self.topology.leads):
                f.write(f"{m}: {lead}  {self.lead_orientations.get(lead, '')}\n")
            if not self.topology.leads:
                f.write("(none)\n")
            f.write("\n")

        self._log(f"Graph parameters saved to {filepath}")
