"""
CoreMixin — __init__, lead orientations, save-path logic.
"""

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
    _parameters_title = "Open Quantum Graph Parameters"

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
