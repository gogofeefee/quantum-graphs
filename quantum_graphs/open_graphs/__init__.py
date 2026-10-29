#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
open_graphs — modular open quantum graph package.

Usage
-----
from open_graphs import OpenGraphs, DirectionalGraphs, NNSD_func

Classes
-------
OpenGraphs
    Base class for open quantum graphs.  Assembles all computation and
    plotting mixins.  Accepts arbitrary per-vertex scattering matrices
    (constant arrays or callables ``σ(k)``).

DirectionalGraphs
    Specialisation where every vertex is a lossless 4-port directional
    coupler.  Scattering matrices are built automatically from coupling
    constants.

Functions
---------
NNSD_func(s, wigner)
    Nearest-Neighbour Spacing Distribution (Wigner surmise).
"""

from .core            import CoreMixin
from .scattering      import ScatteringMixin
from .resonances      import ResonancesMixin
from .experimental    import ExperimentalMixin
from .spectral_stats  import SpectralStatsMixin, NNSD_func
from .plotting        import GraphPlotMixin, SMatrixPlotsMixin, SpectralPlotsMixin, WavefunctionPlotsMixin
from .directional     import DirectionalGraphs


class OpenGraphs(CoreMixin, ScatteringMixin, ResonancesMixin,
                 ExperimentalMixin, SpectralStatsMixin,
                 GraphPlotMixin, SMatrixPlotsMixin, SpectralPlotsMixin,
                 WavefunctionPlotsMixin):
    """
    Open quantum graph in the vertex scattering matrix formalism.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of each vertex (µm).
    bonds : list of (i, p, j, q) or (i, p, j, q, L)
        Port ``p`` of vertex ``i`` joined to port ``q`` of vertex ``j`` by a bond
        of length ``L`` (µm; Euclidean distance between the vertices if
        omitted). Self-loops and multiple bonds are allowed.
    leads : list of (vertex, port)
        External leads, numbered in this order in the S-matrix.
    vertex_matrices : dict, list, ndarray, or callable
        Scattering matrices: dict/list one per vertex, one array for all, or a
        callable ``σ(k)`` for all.
    access_matrices : same forms, or None
        Overrides ``vertex_matrices`` on the vertices that carry a lead.
    neff : float or callable
        Effective refractive index (or ``neff(k)``).
    ng : float, callable, or None
        Group index.  Derived numerically from ``neff(k)`` if ``None``.
    wavelength_range : (float, float)
        Vacuum wavelength range (µm), required.
    bond_spacing : float, sequence, or None
        Drawing only: distance between parallel bonds (see
        :class:`quantum_graphs.plotting.drawing.BondPaths`).
    lead_orientations : dict or None
        ``{(vertex, port): angle_deg}`` for plotting lead stubs.
    name : str
        Graph label used in save-path construction and plot titles.
    save_path : str, Path, or None
        Override the default deterministic save directory.
    """
    pass


__all__ = [
    "OpenGraphs",
    "DirectionalGraphs",
    "NNSD_func",
    "WavefunctionPlotsMixin",
]
