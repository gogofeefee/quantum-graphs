"""
GraphBase — what every graph class shares: topology, vertex matrices, save
path, physics fingerprint and the bond scattering model.
"""

import numpy as np

from ..stats.localization import bond_intensity, eigenspace_intensity, localization_measures
from .config import log
from .persistence import fingerprint, resolve_save_path, sample_function
from .scattering import ScatteringModel
from .topology import build_topology
from .vertices import validate_vertex_matrices


class GraphBase:
    """Mixin used by the graph classes of every subpackage."""

    #: subclasses set True if the graph may have leads
    _allow_leads = False

    # -- construction ---------------------------------------------------------

    def _setup_topology(self, vertex_positions, bonds, leads=()):
        """Validate the graph and set ``self.topology``."""
        self.topology = build_topology(vertex_positions, bonds, leads,
                                       allow_leads=self._allow_leads)
        return self.topology

    def _setup_vertex_matrices(self, matrices, k_samples=()):
        """Store {vertex: σ} after checking their shapes (and unitarity)."""
        validate_vertex_matrices(matrices, self.topology.valency, k_samples)
        self.vertex_scattering_matrix = dict(matrices)
        self._model = None

    # -- shortcuts to the topology -------------------------------------------------

    @property
    def n_vertices(self):
        return self.topology.n_vertices

    @property
    def n_bonds(self):
        return self.topology.n_bonds

    @property
    def n_leads(self):
        return self.topology.n_leads

    @property
    def total_length(self):
        """Sum of the bond lengths (µm)."""
        return self.topology.total_length

    def _setup_save_path(self, name, leaf, save_path, base_dir):
        """Resolve the output directory; nothing is written until the first save."""
        self.name = name
        self._save_path = resolve_save_path(name, leaf, save_path, base_dir)
        self._parameters_written = False

    @property
    def save_path(self):
        """Output directory of the graph (Path). Created on first use, together
        with ``graph_parameters.txt``: building a graph writes nothing to disk."""
        if not self._parameters_written:
            self._parameters_written = True
            self._save_path.mkdir(parents=True, exist_ok=True)
            self._save_parameters()
        return self._save_path

    # -- drawing ----------------------------------------------------------------

    @property
    def bond_paths(self):
        """:class:`~quantum_graphs.plotting.drawing.BondPaths` used by every figure."""
        if getattr(self, "_bond_paths", None) is None:
            from ..plotting.drawing import BondPaths
            self._bond_paths = BondPaths(self.topology, getattr(self, "bond_spacing", None),
                                         getattr(self, "lead_orientations", None))
        return self._bond_paths

    # -- wavenumbers ----------------------------------------------------------

    @staticmethod
    def _check_wavelength_range(wavelength_range):
        """(λ_min, λ_max) in µm, 0 < λ_min < λ_max; there is no default window."""
        try:
            lam_min, lam_max = (float(x) for x in wavelength_range)
        except (TypeError, ValueError):
            raise ValueError("wavelength_range must be a pair (lambda_min, lambda_max) in µm, "
                             f"got {wavelength_range!r}.") from None
        if not 0 < lam_min < lam_max:
            raise ValueError(f"wavelength_range must satisfy 0 < lambda_min < lambda_max, "
                             f"got {wavelength_range!r}.")
        return lam_min, lam_max

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
            self._model = ScatteringModel(self.topology, self.vertex_scattering_matrix,
                                          getattr(self, "neff", 1.0))
        return self._model

    # -- matrices ---------------------------------------------------------------

    def U(self, k):
        """Bond evolution map U(k) = D(k) Σ_αα(k), shape (2B, 2B); k may be complex."""
        return self.model.U(k)

    def sigma_min(self, k):
        """Smallest singular value of I − U(k), for one k or an array of k."""
        if np.ndim(k) == 0:
            return self.model.sigma_min(k)
        return self.model.sigma_min_many(np.asarray(k))

    def secular_det(self, k):
        """Secular determinant det(I − U(k))."""
        return np.linalg.det(np.eye(self.topology.n_directed) - self.model.U(k))

    # -- wavefunction localisation -------------------------------------------

    def bond_intensities(self, vecs):
        """Intensity on each bond (order of ``topology.bonds``).

        ``vecs``: one vector of directed-bond amplitudes, or the orthonormal
        rows of a degenerate eigenspace (diagonal of its projector / g). See
        :mod:`quantum_graphs.stats.localization`.
        """
        return bond_intensity(eigenspace_intensity(vecs))

    def localization_measures(self, vecs):
        """``(S, S_norm, IPR)`` of :meth:`bond_intensities` (IPR = N_b Σ p_b²,
        N_b = number of bonds)."""
        return localization_measures(self.bond_intensities(vecs))

    # -- fingerprint ----------------------------------------------------------

    def physics_spec(self):
        """Numerical description of everything that determines the results."""
        ks = self._k_samples()
        spec = {
            "class": type(self).__name__,
            "bonds": [list(b) for b in self.topology.bonds],
            "leads": [list(lead) for lead in self.topology.leads],
            "vertex_matrices": {v: sample_function(s, ks)
                                for v, s in self.vertex_scattering_matrix.items()},
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

