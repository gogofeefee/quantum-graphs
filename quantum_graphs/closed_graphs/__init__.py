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
    bonds : list of (i, p, j, q) or (i, p, j, q, L)
        Port ``p`` of vertex ``i`` joined to port ``q`` of vertex ``j`` by a bond
        of length ``L`` (µm; Euclidean distance between the vertices if
        omitted). Self-loops and multiple bonds are allowed.
    vertex_matrices : dict, list, ndarray, or callable
        Scattering matrices.  dict/list: one per vertex.  ndarray: same for all.
        callable: same function for all.
    neff : float or callable
        Effective refractive index (or ``neff(k)``).
    ng : float, callable, or None
        Group index.  Derived numerically from ``neff(k)`` if ``None``.
    wavelength_range : (float, float)
        Vacuum wavelength range (µm).
    bond_spacing : float, sequence, or None
        Drawing only: distance between parallel bonds (see
        :class:`quantum_graphs.plotting.drawing.BondPaths`).
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
