"""
GraphBase — what every graph class shares: topology, vertex matrices, save
path, physics fingerprint and the bond scattering model.
"""

import datetime
import inspect

import numpy as np

from ..stats.localization import bond_intensity, eigenspace_intensity, localization_measures
from .config import log
from .persistence import fingerprint, resolve_save_path, sample_function
from .scattering import NULLITY_TOL, ScatteringModel
from .sectors import find_sectors
from .topology import build_topology
from .vertices import validate_vertex_matrices


class GraphBase:
    """Mixin used by the graph classes of every subpackage."""

    #: subclasses set True if the graph may have leads
    _allow_leads = False
    #: first line of ``graph_parameters.txt``
    _parameters_title = "Quantum Graph Parameters"

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
        self._sectors = None

    # -- shortcuts to the topology -------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices."""
        return self.topology.n_vertices

    @property
    def n_bonds(self):
        """Number of (undirected) bonds B; U has 2B rows."""
        return self.topology.n_bonds

    @property
    def n_leads(self):
        """Number of leads (0 for a closed graph)."""
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

    @save_path.setter
    def save_path(self, path):
        """Send the next outputs to another directory (created on first use)."""
        from pathlib import Path
        self._save_path = Path(path)
        self._parameters_written = False

    def _save_parameters(self):
        """Write a human-readable description of the graph to graph_parameters.txt."""
        t = self.topology
        lines = [f"### {self._parameters_title} ###",
                 f"Graph name: {self.name}",
                 f"Saved on: {datetime.datetime.now()}",
                 f"Physics fingerprint: {self.fingerprint}",
                 "",
                 f"Vertices: {self.n_vertices}",
                 f"Bonds: {self.n_bonds}",
                 f"Directed bonds: {t.n_directed}"]
        if self._allow_leads:
            lines.append(f"Leads: {self.n_leads}")
        lines.append(f"Total length: {self.total_length:.6f} µm")
        if getattr(self, "wavelength_range", None) is not None:
            lam_min, lam_max = self.wavelength_range
            lines.append(f"Wavelength range (vacuum): {lam_min:.3f} – {lam_max:.3f} µm")
        lines.append("")
        for label, value in (("Effective index neff", getattr(self, "neff", None)),
                             ("Group index ng", getattr(self, "ng", None))):
            if callable(value):
                lines += [f"{label}: function of k", _source_or_placeholder(value)]
            elif value is not None:
                lines.append(f"{label}: {float(value):.6f}")
        lines += self._extra_parameter_lines()

        matrices = getattr(self, "vertex_scattering_matrix", None)
        if matrices:
            lines += ["", "### Vertex scattering matrices ###"]
            for v, sigma in matrices.items():
                if callable(sigma):
                    lines += [f"Vertex {v} (valency {t.valency[v]}): function of k",
                              _source_or_placeholder(getattr(sigma, "coupling", sigma))]
                else:
                    lines.append(f"Vertex {v} (valency {t.valency[v]}):")
                    lines += ["  " + "  ".join(_format_entry(x) for x in row)
                              for row in np.real_if_close(np.asarray(sigma))]
        lines += ["", "### Vertex positions (µm) ###"]
        lines += [f"Vertex {v}: ({x:.6f}, {y:.6f})" for v, (x, y) in enumerate(t.positions)]
        lines += ["", "### Bonds (i, p, j, q, L in µm) ###"]
        lines += [f"{b}: ({i}, {p}) — ({j}, {q}) : {L:.6f}"
                  for b, (i, p, j, q, L) in enumerate(t.bonds)]
        if self._allow_leads:
            orientations = getattr(self, "lead_orientations", None) or {}
            lines += ["", "### Leads (vertex, port) and drawing orientation (deg) ###"]
            lines += [f"{m}: {lead}  {orientations.get(lead, '')}".rstrip()
                      for m, lead in enumerate(t.leads)] or ["(none)"]

        filepath = self._save_path / "graph_parameters.txt"
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"Graph parameters saved to {filepath}")

    def _extra_parameter_lines(self):
        """Class-specific lines of graph_parameters.txt (after the indices)."""
        return []

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

    @property
    def sectors(self):
        """:class:`~quantum_graphs.core.sectors.Sectors` of the bond dynamics:
        independent blocks of U(k), reversed partners, expected ensemble."""
        if getattr(self, "_sectors", None) is None:
            self._sectors = find_sectors(self.topology, self.model, self._k_samples())
        return self._sectors

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

    def null_vectors(self, k, multiplicity=None, tol=NULLITY_TOL):
        """Orthonormal basis of ker(I − U(k)), one vector per row.

        The basis has ``multiplicity`` vectors (the right singular vectors of
        the smallest singular values) when it is given — for a level of the
        computed spectrum, its stored multiplicity — and otherwise one vector
        per singular value at or below ``tol``. Component d of a vector is the
        amplitude arriving at the end of directed bond d (a = U a); for a
        degenerate level the basis is arbitrary.
        """
        _, s, Vh = np.linalg.svd(np.eye(self.topology.n_directed) - self.model.U(k))
        n = int(np.sum(s <= tol)) if multiplicity is None else int(multiplicity)
        return Vh[s.size - n:].conj()               # rows of Vh are v^H

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
                                for v, s in getattr(self, "vertex_scattering_matrix", {}).items()},
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


def _source_or_placeholder(func):
    try:
        return inspect.getsource(func).rstrip("\n")
    except (OSError, TypeError):
        return "(source code unavailable)"


def _format_entry(x):
    if np.iscomplexobj(x):
        return f"{x.real:+.6f}{x.imag:+.6f}j"
    return f"{x:+.6f}"
