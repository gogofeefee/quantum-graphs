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
    port_map : list of lists
        Connectivity.  ``port_map[i][p] = (j, q)`` connects port p of vertex i
        to port q of vertex j.  ``port_map[i][p] = None`` marks an external lead.
    lengths : dict or None
        Bond lengths ``{((i,p),(j,q)): L_um}``.  ``None`` → Euclidean distances
        from ``vertex_positions``.
    vertex_matrices : dict or None
        Per-vertex scattering matrices ``{v: σ}`` where ``σ`` is a 2-D array or
        a callable ``σ(k)``.  Overrides ``internal_matrix`` / ``external_matrix``
        for the specified vertices.
    internal_matrix : array, callable, or None
        Default scattering matrix for internal (non-lead) vertices.
    external_matrix : array, callable, or None
        Default scattering matrix for external (lead-connected) vertices.
    neff : float or callable
        Effective refractive index (or ``neff(k)``).
    ng : float, callable, or None
        Group index.  Derived numerically from ``neff(k)`` if ``None``.
    wavelength_range : (float, float)
        Vacuum wavelength range (µm).
    virtual_vertex_spacing : float, sequence, or None
        Spacing when splitting self-loops / multiple bonds into virtual chains.
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
