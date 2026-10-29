#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directional.py — DirectionalClassicalGraphs.

Specialisation for 4-port directional-coupler graphs: the vertex matrices can
be built from coupling constants.
"""

from ..core.vertices import couplings_for_vertices, directional_coupler_matrix

from .core      import CoreMixin
from .dynamics  import DynamicsMixin
from .plotting  import GraphPlotMixin, RhoPlotMixin, AnimationMixin


class DirectionalClassicalGraphs(
    CoreMixin, DynamicsMixin,
    GraphPlotMixin, RhoPlotMixin, AnimationMixin,
):
    """
    Classical directional graph: all dynamics and plotting of ClassicalGraphs,
    with vertex matrices built from the coupling constants ``couplings`` (a
    float, a function C(k), a list or a dict {vertex: C}). ``plot_graph(show_port_markers=True)`` shows the ports (blue:
    0, 1; orange: 2, 3).
    """

    def __init__(
        self,
        vertex_positions,
        bonds,
        *,
        couplings,
        bond_spacing=None,
        name="MyDirectionalClassicalGraph",
        save_path=None,
        base_dir=None,
    ):
        values = couplings_for_vertices(couplings, len(vertex_positions))
        missing = [v for v, C in enumerate(values) if C is None]
        if missing:
            raise ValueError(f"Missing coupling constant for vertices {missing}.")
        self.couplings = couplings
        vertex_matrices = {v: directional_coupler_matrix(C) for v, C in enumerate(values)}
        super().__init__(
            vertex_positions=vertex_positions,
            bonds=bonds,
            vertex_matrices=vertex_matrices,
            bond_spacing=bond_spacing,
            name=name,
            save_path=save_path,
            base_dir=base_dir,
        )
