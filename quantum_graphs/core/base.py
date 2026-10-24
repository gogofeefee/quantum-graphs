"""
GraphBase — what every graph class shares: topology, vertex matrices, save
path, physics fingerprint and the bond scattering model.
"""

import copy

import numpy as np

from .config import log
from .persistence import fingerprint, resolve_save_path, sample_function
from .scattering import ScatteringModel
from .topology import build_topology
from .vertices import validate_vertex_matrices


class GraphBase:
    """Mixin used by the graph classes of every subpackage."""

    #: subclasses set True if ``None`` entries (leads) are allowed in port_map
    _allow_leads = False

    # -- construction ---------------------------------------------------------

    def _setup_topology(self, vertex_positions, port_map, lengths, virtual_vertex_spacing):
        """Validate and virtualise the graph; set the topology attributes."""
        topo = build_topology(vertex_positions, port_map, lengths,
                              virtual_vertex_spacing=virtual_vertex_spacing,
                              allow_leads=self._allow_leads)
        topo.apply_to(self)
        self._topology = topo
        self.port_map = port_map
        # keep the user's dict (and its key orientation); Euclidean lengths if none given
        self.lengths = dict(lengths) if lengths is not None else dict(topo.canonical_lengths)
        self.virtual_vertex_spacing = virtual_vertex_spacing
        return topo

    def _setup_vertex_matrices(self, matrices, k_samples=()):
        """Store {real vertex: σ} plus pass-through matrices for virtual vertices."""
        from .vertices import PASS_THROUGH
        validate_vertex_matrices(matrices, self.valencies, k_samples)
        self.vertex_scattering_matrix = {}
        for v in range(self.n_vertices_total):
            self.vertex_scattering_matrix[v] = (PASS_THROUGH if v in self.virtual_vertices
                                                else matrices[v])
        self._model = None

    def _setup_save_path(self, name, leaf, save_path, base_dir, write_parameters=True):
        self.name = name
        self.save_path = resolve_save_path(name, leaf, save_path, base_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        if write_parameters:
            self._save_parameters()

    # -- wavenumbers ----------------------------------------------------------

    def _k_range(self):
        lam_min, lam_max = self.wavelength_range
        return 2 * np.pi / lam_max, 2 * np.pi / lam_min

    def _k_samples(self, n=9):
        """Wavenumbers used to fingerprint and validate functions of k."""
        if getattr(self, "wavelength_range", None) is None:
            return []
        k_min, k_max = self._k_range()
        return list(np.linspace(k_min, k_max, n))

    # -- scattering model -------------------------------------------------------

    @property
    def model(self):
        """The :class:`~quantum_graphs.core.scattering.ScatteringModel` of the graph."""
        if getattr(self, "_model", None) is None:
            self._model = ScatteringModel(self._topology, self.vertex_scattering_matrix,
                                          getattr(self, "neff", 1.0))
        return self._model

    # -- fingerprint ----------------------------------------------------------

    def physics_spec(self):
        """Numerical description of everything that determines the results."""
        ks = self._k_samples()
        spec = {
            "class": type(self).__name__,
            "port_map": [[None if c is None else list(c) for c in ports] for ports in self.port_map],
            "lengths": sorted([list(a) + list(b) + [L] for (a, b), L
                               in self._topology.canonical_lengths.items()]),
            "vertex_matrices": {v: sample_function(s, ks)
                                for v, s in self.vertex_scattering_matrix.items()
                                if v not in self.virtual_vertices},
            "wavelength_range": list(getattr(self, "wavelength_range", None) or []),
        }
        if hasattr(self, "neff"):
            spec["neff"] = sample_function(self.neff, ks)
            spec["ng"] = sample_function(self.ng, ks)
        return spec

    @property
    def fingerprint(self):
        """16-hex digest of :meth:`physics_spec`, stored with every cache file."""
        if getattr(self, "_fingerprint", None) is None:
            self._fingerprint = fingerprint(self.physics_spec())
        return self._fingerprint

    # -- misc -------------------------------------------------------------------

    def _log(self, *args, verbose=None, **kwargs):
        log(*args, verbose=verbose, **kwargs)

    def copy_topology_kwargs(self):
        """``dict(vertex_positions, port_map, lengths)`` of the *original* graph, to
        build another graph class on the same topology."""
        return dict(vertex_positions=list(self.vertex_positions[:self.n_vertices]),
                    port_map=copy.deepcopy(self.port_map), lengths=dict(self.lengths))
