#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectionalGraphs — closed quantum graph made of 4-port directional couplers.

Inherits all ClosedGraphs functionality via the same mixin chain.
Scattering matrices are built automatically from coupling constants
(constant float or callable C(k)).
"""

from ..core.vertices import couplings_for_vertices, directional_coupler_matrix
from .core           import CoreMixin
from .spectrum       import SpectrumMixin
from .spectral_stats import SpectralStatsMixin
from .plotting.graph_plot        import GraphPlotMixin
from .plotting.spectral_plots    import SpectralPlotsMixin
from .plotting.wavefunction_plots import WavefunctionPlotsMixin


class DirectionalGraphs(CoreMixin, SpectrumMixin, SpectralStatsMixin,
                        GraphPlotMixin, SpectralPlotsMixin, WavefunctionPlotsMixin):
    """
    Closed quantum graph where every vertex is a lossless 4-port directional
    coupler.

    Parameters
    ----------
    vertex_positions : list of (float, float)
    bonds : list of (i, p, j, q[, L])
    vertex_couplings : dict, list, float, or callable
        Coupling constants.  float/callable → same for all vertices.
    neff : float or callable
    ng : float, callable, or None
    wavelength_range : (float, float)
    bond_spacing : float, sequence, or None
        Drawing only: distance between parallel bonds (see BondPaths).
    name : str
    save_path : str, Path, or None
    base_dir : str, Path, or None
    """

    def __init__(self,
                 vertex_positions,
                 bonds,
                 vertex_couplings=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 bond_spacing=None,
                 name="MyClosedDirectionalGraph",
                 save_path=None,
                 base_dir=None):

        if vertex_couplings is None:
            raise ValueError("vertex_couplings is required: a float C, a function C(k), "
                             "a list (one C per vertex) or a dict {vertex: C}.")
        self.vertex_couplings = vertex_couplings
        vertex_matrices = self._build_vertex_matrices(vertex_positions, bonds)
        self.coupling_info = self._get_coupling_description()

        super().__init__(
            vertex_positions       = vertex_positions,
            bonds                  = bonds,
            vertex_matrices        = vertex_matrices,
            neff                   = neff,
            ng                     = ng,
            wavelength_range       = wavelength_range,
            bond_spacing           = bond_spacing,
            name                   = name,
            save_path              = save_path,
            base_dir               = base_dir,
        )

    # =========================================================================
    # Coupling helpers
    # =========================================================================

    def _build_vertex_matrices(self, vertex_positions, bonds):
        n = len(vertex_positions)
        couplings = couplings_for_vertices(self.vertex_couplings, n)
        missing = [v for v, C in enumerate(couplings) if C is None]
        if missing:
            raise ValueError(f"Missing coupling constant for vertices {missing}.")
        return {v: directional_coupler_matrix(C) for v, C in enumerate(couplings)}

    def _get_coupling_description(self):
        if isinstance(self.vertex_couplings, (dict, list, tuple)):
            return "custom_vertex_couplings"
        C = self.vertex_couplings
        if callable(C):
            return getattr(C, "__name__", "Cfunc")
        return f"C={C:.2f}"
