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
    with vertex matrices built from coupling constants if ``vertex_couplings``
    is given. ``plot_graph(show_port_markers=True)`` shows the ports (blue:
    0, 1; orange: 2, 3).
    """

    def __init__(
        self,
        vertex_positions,
        bonds,
        vertex_matrices=None,
        bond_spacing=None,
        name="MyDirectionalClassicalGraph",
        save_path=None,
        base_dir=None,
        vertex_couplings=None,
    ):
        """``vertex_couplings`` (float, list or dict of C) builds the directional-
        coupler matrices, as in the quantum DirectionalGraphs classes; it is an
        alternative to ``vertex_matrices``."""
        if vertex_couplings is not None:
            if vertex_matrices is not None:
                raise ValueError("Give vertex_couplings or vertex_matrices, not both.")
            couplings = couplings_for_vertices(vertex_couplings, len(vertex_positions))
            vertex_matrices = {v: directional_coupler_matrix(C) for v, C in enumerate(couplings)}
        super().__init__(
            vertex_positions=vertex_positions,
            bonds=bonds,
            vertex_matrices=vertex_matrices,
            bond_spacing=bond_spacing,
            name=name,
            save_path=save_path,
            base_dir=base_dir,
        )
