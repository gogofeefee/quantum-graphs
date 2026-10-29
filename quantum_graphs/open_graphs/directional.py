#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectionalGraphs — open quantum graph where every vertex is a 4-port
directional coupler.  Inherits all OpenGraphs functionality; scattering
matrices are built automatically from coupling constants.
"""

from ..core.vertices import couplings_for_vertices, directional_coupler_matrix
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

    The coupling constants determine the scattering matrices.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of each vertex (µm).
    bonds : list of (i, p, j, q) or (i, p, j, q, L)
        Port ``p`` of vertex ``i`` joined to port ``q`` of vertex ``j``, length
        ``L`` (µm; Euclidean if omitted).
    leads : list of (vertex, port)
        External leads, numbered in this order in the S-matrix.
    couplings : float, callable, list or dict
        Power cross-coupling C of the couplers: one value or function C(k) for
        all, a list (one per vertex) or a dict {vertex: C}.
    access_couplings : same forms, or None
        Coupling of the access couplers (vertices carrying a lead); default:
        ``couplings``.
    wavelength_range : (float, float)
        Vacuum wavelength window (µm), required.
    neff : float or callable
        Effective refractive index.
    ng : float, callable, or None
        Group index.  Derived from ``neff`` if ``None``.
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
                 leads,
                 *,
                 couplings,
                 wavelength_range,
                 access_couplings=None,
                 neff=1.0,
                 ng=None,
                 bond_spacing=None,
                 lead_orientations=None,
                 name="MyOpenDirectionalGraph",
                 save_path=None,
                 base_dir=None):

        self.couplings = couplings
        self.access_couplings = access_couplings
        n = len(vertex_positions)
        matrices = _coupler_matrices(couplings, range(n), n, "couplings")
        access = None
        if access_couplings is not None:
            lead_vertices = sorted({int(v) for v, _ in leads})
            access = matrices.copy()
            access.update(_coupler_matrices(access_couplings, lead_vertices, n, "access_couplings"))
        self.coupling_info = self._get_coupling_description()

        super().__init__(
            vertex_positions=vertex_positions,
            bonds=bonds,
            leads=leads,
            vertex_matrices=matrices,
            access_matrices=access,
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

    def _get_coupling_description(self):
        """Compact coupling string for the deterministic save path."""
        def _str(C):
            if isinstance(C, (dict, list, tuple)):
                return "custom"
            if callable(C):
                return getattr(C, "__name__", "Cfunc")
            return f"{C:.2f}"
        parts = [f"C={_str(self.couplings)}"]
        if self.access_couplings is not None:
            parts.append(f"Caccess={_str(self.access_couplings)}")
        return "_".join(parts)


def _coupler_matrices(couplings, vertices, n, name):
    """{v: directional-coupler matrix} for ``vertices`` from a float, a function
    C(k), a list (one value per vertex) or a dict {vertex: C}."""
    values = couplings_for_vertices(couplings, n)
    missing = [v for v in vertices if values[v] is None]
    if missing:
        raise ValueError(f"Missing coupling constant in {name} for vertices {missing}.")
    return {v: directional_coupler_matrix(values[v]) for v in vertices}
