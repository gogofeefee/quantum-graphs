#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
closed_graphs — modular closed quantum graph package.

Usage
-----
from closed_graphs import ClosedGraphs, DirectionalGraphs, NNSD_func

Classes
-------
ClosedGraphs
    Base class for closed quantum graphs.  Assembles all computation and
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

from .core           import CoreMixin
from .spectrum       import SpectrumMixin
from .spectral_stats import SpectralStatsMixin, NNSD_func
from .plotting       import GraphPlotMixin, SpectralPlotsMixin, WavefunctionPlotsMixin
from .directional    import DirectionalGraphs


class ClosedGraphs(CoreMixin, SpectrumMixin, SpectralStatsMixin,
                   GraphPlotMixin, SpectralPlotsMixin, WavefunctionPlotsMixin):
    """
    Closed quantum graph in the vertex scattering matrix formalism.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of each vertex (µm).
    port_map : list of lists
        Connectivity.  ``port_map[i][p] = (j, q)`` connects port p of vertex i
        to port q of vertex j.  No ``None`` entries (closed graph: no leads).
    lengths : dict or None
        Bond lengths ``{((i,p),(j,q)): L_um}``.  ``None`` → Euclidean distances
        from ``vertex_positions``.
    vertex_matrices : dict, list, ndarray, or callable
        Scattering matrices.  dict/list: one per vertex.  ndarray: same for all.
        callable: same function for all.
    neff : float or callable
        Effective refractive index (or ``neff(k)``).
    ng : float, callable, or None
        Group index.  Derived numerically from ``neff(k)`` if ``None``.
    wavelength_range : (float, float)
        Vacuum wavelength range (µm).
    virtual_vertex_spacing : float, sequence, or None
        Spacing when splitting self-loops / multiple bonds into virtual chains.
    name : str
        Graph label used in save-path construction and plot titles.
    save_path : str, Path, or None
        Override the default deterministic save directory.
    base_dir : str, Path, or None
        Base directory for the save path.  Priority: ``save_path`` > ``base_dir``
        > ``sys.argv[0].parent`` > ``cwd``.
    """
    pass


__all__ = [
    "ClosedGraphs",
    "DirectionalGraphs",
    "NNSD_func",
]
