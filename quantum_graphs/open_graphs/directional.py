#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectionalGraphs — open quantum graph where every vertex is a 4-port
directional coupler.  Inherits all OpenGraphs functionality; scattering
matrices are built automatically from coupling constants.
"""

from ..core.vertices import directional_coupler_matrix
from .core           import CoreMixin
from .scattering     import ScatteringMixin
from .resonances     import ResonancesMixin
from .experimental   import ExperimentalMixin
from .spectral_stats import SpectralStatsMixin
from .plotting.graph_plot        import GraphPlotMixin
from .plotting.s_matrix_plots    import SMatrixPlotsMixin
from .plotting.spectral_plots    import SpectralPlotsMixin
from .plotting.wavefunction_plots import WavefunctionPlotsMixin


class DirectionalGraphs(CoreMixin, ScatteringMixin, ResonancesMixin,
                        ExperimentalMixin, SpectralStatsMixin,
                        GraphPlotMixin, SMatrixPlotsMixin, SpectralPlotsMixin,
                        WavefunctionPlotsMixin):
    """
    Open quantum graph where every real vertex is a lossless 4-port
    directional coupler.

    Coupling constants determine the scattering matrices automatically.
    Optionally, per-vertex couplings can be specified via ``vertex_couplings``.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of each vertex (µm).
    bonds : list of (i, p, j, q) or (i, p, j, q, L)
        Port ``p`` of vertex ``i`` joined to port ``q`` of vertex ``j``, length
        ``L`` (µm; Euclidean if omitted).
    leads : list of (vertex, port)
        External leads, numbered in this order in the S-matrix.
    vertex_couplings : dict, optional
        Per-vertex coupling constants ``{v: C}`` (float or callable).
        When provided, overrides ``internal_coupling`` / ``external_coupling``
        for the specified vertices.
    internal_coupling : float, callable, or None
        Default coupling for internal (non-lead) vertices.
    external_coupling : float, callable, or None
        Default coupling for external (lead-connected) vertices.
    neff : float or callable
        Effective refractive index.
    ng : float, callable, or None
        Group index.  Derived from ``neff`` if ``None``.
    wavelength_range : (float, float)
        Vacuum wavelength range (µm).
    bond_spacing : float, sequence, or None
        Drawing only: distance between parallel bonds (see BondPaths).
    lead_orientations : dict or None
        ``{(vertex, port): angle_deg}`` for plotting.
    name : str
        Graph label (used in save paths and plot titles).
    save_path : str, Path, or None
        Override the default save directory.
    """

    def __init__(self,
                 vertex_positions,
                 bonds,
                 leads=(),
                 vertex_couplings=None,
                 internal_coupling=None,
                 external_coupling=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 bond_spacing=None,
                 lead_orientations=None,
                 name="MyOpenDirectionalGraph",
                 save_path=None,
                 base_dir=None):

        self.vertex_couplings   = vertex_couplings
        self.internal_coupling  = internal_coupling
        self.external_coupling  = external_coupling

        vertex_matrices    = self._build_vertex_matrices(vertex_positions, leads)
        self.coupling_info = self._get_coupling_description()

        super().__init__(
            vertex_positions=vertex_positions,
            bonds=bonds,
            leads=leads,
            vertex_matrices=vertex_matrices,
            internal_matrix=None,
            external_matrix=None,
            neff=neff,
            ng=ng,
            wavelength_range=wavelength_range,
            bond_spacing=bond_spacing,
            lead_orientations=lead_orientations,
            name=name,
            save_path=save_path,
            base_dir=base_dir,
        )

    # -------------------------------------------------------------------------
    # Coupling helpers
    # -------------------------------------------------------------------------

    def _build_vertex_matrices(self, vertex_positions, leads):
        """Directional-coupler matrix of every vertex from its coupling constant
        (``vertex_couplings[v]``, else ``external_coupling`` for a vertex carrying
        a lead, else ``internal_coupling``)."""
        lead_vertices = {int(v) for v, _ in leads}
        matrices = {}
        for v in range(len(vertex_positions)):
            if self.vertex_couplings is not None and v in self.vertex_couplings:
                C = self.vertex_couplings[v]
            else:
                kind = "external" if v in lead_vertices else "internal"
                C = self.external_coupling if kind == "external" else self.internal_coupling
                if C is None:
                    raise ValueError(f"No coupling constant for vertex {v} ({kind}). Provide "
                                     f"``{kind}_coupling`` or ``vertex_couplings``.")
            matrices[v] = directional_coupler_matrix(C)
        return matrices

    def _get_coupling_description(self):
        """Build a compact coupling string for the deterministic save-path."""
        def _str(C):
            if C is None:
                return None
            if callable(C):
                return getattr(C, "__name__", "Cfunc")
            return f"{C:.2f}"

        if self.vertex_couplings is not None:
            return "custom_vertex_couplings"

        parts = []
        cin  = _str(self.internal_coupling)
        cext = _str(self.external_coupling)
        if cin  is not None:
            parts.append(f"Cin={cin}")
        if cext is not None:
            parts.append(f"Cext={cext}")
        return "_".join(parts) if parts else "no_coupling"
