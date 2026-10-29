#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting/rho_plots.py — RhoPlotMixin for classical_graphs.

Arrow-style helpers and static rho visualisation methods:
  plot_rho, plot_rho_after_steps.
"""

from ...core.config import log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .._utils import _normalize, _perpendicular, _clip01


class RhoPlotMixin:
    """Static visualisation of probability vectors on directed bonds."""

    # ==================================================================
    # Style defaults
    # ==================================================================

    def _get_default_arrow_style_parameters(self):
        return {
            "bond_offset_ratio": 0.06,
            "arrow_length_ratio": 0.32,
            "arrow_length_min_ratio": 0.12,
            "arrow_length_max_ratio": 0.55,
            "arrow_head_ratio": 0.28,
            "arrow_width_min": 0.8,
            "arrow_width_max": 7.0,
            "arrow_alpha_min": 0.18,
            "arrow_alpha_max": 1.0,
            "zero_threshold": 1e-15,
            "rho_color": "tab:purple",
        }

    def _get_median_bond_length(self):
        return float(np.median(self.topology.length))

    # ==================================================================
    # Arrow geometry
    # ==================================================================

    def _compute_arrow_segment_for_dbond(
        self,
        dbond,
        tau=0.5,
        width=None,
        arrow_width_min=3.0,
        arrow_width_max=10.0,
        bond_offset_ratio_min=0.020,
        bond_offset_ratio_max=0.045,
        arrow_length_mode="bond_relative",
        constant_arrow_length_frac=0.35,
        arrow_length_ratio=0.32,
        arrow_length_min_ratio=0.12,
        arrow_length_max_ratio=0.55,
        vertex_radius_frac=0.065,
    ):
        """
        Start/end points of the displayed arrow for directed bond ``dbond`` (int).

        The arrow is a straight chord between two points of the drawn bond
        curve, shifted to the left of the direction of propagation (so the two
        directions of a bond sit on opposite sides), by more where the curve
        turns so that the chord clears it.

        tau in [0, 1]:
          0 → tail at start-vertex border
          1 → head at end-vertex border
          0.5 → centered in the usable segment

        Returns
        -------
        start, end : np.ndarray, np.ndarray
        """
        tau = _clip01(tau)
        d = int(dbond)
        paths = self.bond_paths
        b = self.topology.bond_of(d)
        drawn = paths.drawn_length(b)

        Lmin = float(np.min(self.topology.length))
        radius = float(vertex_radius_frac) * Lmin
        usable_length = max(drawn - 2.0 * radius, 0.0)

        # Arrow length
        if arrow_length_mode == "bond_relative":
            arrow_length = float(np.clip(
                arrow_length_ratio, arrow_length_min_ratio, arrow_length_max_ratio
            )) * usable_length
        elif arrow_length_mode == "constant":
            if constant_arrow_length_frac <= 0:
                raise ValueError("constant_arrow_length_frac must be > 0.")
            arrow_length = float(constant_arrow_length_frac) * Lmin
        else:
            raise ValueError("arrow_length_mode must be 'bond_relative' or 'constant'.")
        arrow_length = min(arrow_length, usable_length)

        # Placement along the curve (arc-length fractions)
        max_start = max(usable_length - arrow_length, 0.0)
        s0 = tau * max_start
        s_tail = (radius + s0) / drawn
        s_head = (radius + s0 + arrow_length) / drawn
        raw_start, raw_end = paths.directed_point(d, np.array([s_tail, s_head]))
        dir_start, dir_end = paths.directed_tangent(d, np.array([s_tail, s_head]))

        # Width-dependent lateral offset
        if width is None:
            t_width = 0.5
        elif arrow_width_max <= arrow_width_min:
            t_width = 0.0
        else:
            t_width = _clip01(
                (float(width) - float(arrow_width_min))
                / (float(arrow_width_max) - float(arrow_width_min))
            )

        bond_offset_ratio = (
            (1.0 - t_width) * float(bond_offset_ratio_min)
            + t_width * float(bond_offset_ratio_max)
        )
        offset_mag = bond_offset_ratio * self._get_median_bond_length()
        # a straight chord dips towards a curved bond by its sagitta, ~chord·angle/8
        angle = float(np.arccos(np.clip(np.dot(dir_start, dir_end), -1.0, 1.0)))
        offset_mag += 1.3 * float(np.linalg.norm(raw_end - raw_start)) * angle / 8.0

        return (raw_start + offset_mag * _perpendicular(dir_start),
                raw_end + offset_mag * _perpendicular(dir_end))

    def _rho_to_width_alpha(
        self,
        rho_value,
        rho_max=1.0,
        arrow_width_min=3.0,
        arrow_width_max=10.0,
        arrow_alpha_min=1.0,
        arrow_alpha_max=1.0,
        zero_threshold=1e-15,
    ):
        """Map a rho value to (width, alpha)."""
        rho_value = max(float(rho_value), 0.0)
        if rho_value <= zero_threshold or rho_max <= zero_threshold:
            return 0.0, 0.0
        x = _clip01(rho_value / rho_max)
        width = arrow_width_min + x * (arrow_width_max - arrow_width_min)
        alpha = arrow_alpha_min + x * (arrow_alpha_max - arrow_alpha_min)
        return width, alpha

    def _add_single_directed_arrow(
        self,
        ax,
        start,
        end,
        width,
        alpha,
        color="tab:purple",
        arrow_head_scale=12.0,
        zorder=6,
    ):
        """Add one FancyArrowPatch to the axis."""
        mutation_scale = max(6.0, float(arrow_head_scale) + 0.8 * float(width))
        patch = mpatches.FancyArrowPatch(
            posA=(start[0], start[1]),
            posB=(end[0], end[1]),
            arrowstyle='-|>',
            mutation_scale=mutation_scale,
            linewidth=width,
            color=color,
            alpha=alpha,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
            capstyle='butt',
            joinstyle='miter',
        )
        ax.add_patch(patch)
        return patch

    def _draw_arrows_for_rho(
        self,
        ax,
        bonds_to_draw,
        rho_max,
        arrow_width_min,
        arrow_width_max,
        arrow_alpha_min,
        arrow_alpha_max,
        zero_threshold,
        bond_offset_ratio_min,
        bond_offset_ratio_max,
        arrow_length_mode,
        constant_arrow_length_frac,
        arrow_length_ratio,
        arrow_length_min_ratio,
        arrow_length_max_ratio,
        arrow_head_scale,
        rho_color,
        base_zorder=6,
    ):
        """
        Draw one arrow per entry in bonds_to_draw.

        Parameters
        ----------
        bonds_to_draw : list of (dbond, rho_val, tau, alpha_multiplier)
            - dbond : directed bond index
            - rho_val : probability value
            - tau : position along bond in [0, 1]
            - alpha_multiplier : factor applied to computed alpha (for fading)
        """
        for dbond, rho_val, tau, alpha_mult in bonds_to_draw:
            width, alpha = self._rho_to_width_alpha(
                rho_value=rho_val,
                rho_max=rho_max,
                arrow_width_min=arrow_width_min,
                arrow_width_max=arrow_width_max,
                arrow_alpha_min=arrow_alpha_min,
                arrow_alpha_max=arrow_alpha_max,
                zero_threshold=zero_threshold,
            )
            alpha *= alpha_mult
            if width <= 0.0 or alpha <= 0.0:
                continue
            start, end = self._compute_arrow_segment_for_dbond(
                dbond=dbond,
                tau=tau,
                width=width,
                arrow_width_min=arrow_width_min,
                arrow_width_max=arrow_width_max,
                bond_offset_ratio_min=bond_offset_ratio_min,
                bond_offset_ratio_max=bond_offset_ratio_max,
                arrow_length_mode=arrow_length_mode,
                constant_arrow_length_frac=constant_arrow_length_frac,
                arrow_length_ratio=arrow_length_ratio,
                arrow_length_min_ratio=arrow_length_min_ratio,
                arrow_length_max_ratio=arrow_length_max_ratio,
            )
            self._add_single_directed_arrow(
                ax=ax, start=start, end=end,
                width=width, alpha=alpha,
                color=rho_color,
                arrow_head_scale=arrow_head_scale,
                zorder=base_zorder,
            )

    # ==================================================================
    # Public plot methods
    # ==================================================================

    def plot_rho(
        self,
        rho,
        ax=None,
        title=None,
        figsize=(10, 8),
        rho_max=1.0,
        bond_offset_ratio_min=0.020,
        bond_offset_ratio_max=0.045,
        arrow_length_mode="constant",
        constant_arrow_length_frac=0.35,
        arrow_length_ratio=0.32,
        arrow_length_min_ratio=0.12,
        arrow_length_max_ratio=0.55,
        arrow_head_scale=12.0,
        arrow_width_min=3.0,
        arrow_width_max=10.0,
        arrow_alpha_min=1.0,
        arrow_alpha_max=1.0,
        rho_color="tab:purple",
        zero_threshold=1e-15,
        save_fig=False,
        filename="rho_on_graph.pdf",
        dpi=300,
    ):
        """
        Plot the probability vector rho on the directed bonds.

        Arrow width encodes rho; centred arrows (tau=0.5) for a clear snapshot.

        Returns
        -------
        fig, ax
        """
        rho = self.validate_rho(rho, normalize=False)
        if rho_max is None:
            rho_max = float(np.max(rho)) if rho.size else 1.0

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            fig = ax.figure

        self.plot_graph(ax=ax, show_lengths=False, show_port_markers=True,
                        title=None, save_fig=False)

        # Build arrow list
        bonds_to_draw = [(d, float(rho[d]), 0.5, 1.0) for d in range(self.topology.n_directed)]

        self._draw_arrows_for_rho(
            ax=ax,
            bonds_to_draw=bonds_to_draw,
            rho_max=float(rho_max),
            arrow_width_min=arrow_width_min,
            arrow_width_max=arrow_width_max,
            arrow_alpha_min=arrow_alpha_min,
            arrow_alpha_max=arrow_alpha_max,
            zero_threshold=zero_threshold,
            bond_offset_ratio_min=bond_offset_ratio_min,
            bond_offset_ratio_max=bond_offset_ratio_max,
            arrow_length_mode=arrow_length_mode,
            constant_arrow_length_frac=constant_arrow_length_frac,
            arrow_length_ratio=arrow_length_ratio,
            arrow_length_min_ratio=arrow_length_min_ratio,
            arrow_length_max_ratio=arrow_length_max_ratio,
            arrow_head_scale=arrow_head_scale,
            rho_color=rho_color,
        )

        if title is None:
            title = f"{self.name} — rho on directed bonds"
        if title:
            ax.set_title(title)

        if created_fig:
            fig.tight_layout()

        if save_fig:
            out = self.save_path / filename
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            log(f"✓ Figure saved to: {out}")

        return fig, ax

    def plot_rho_after_steps(
        self,
        rho0,
        n_steps,
        k=None,
        normalize_input=True,
        ax=None,
        title=None,
        figsize=(10, 8),
        rho_max=1.0,
        bond_offset_ratio_min=0.020,
        bond_offset_ratio_max=0.045,
        arrow_length_mode="constant",
        constant_arrow_length_frac=0.35,
        arrow_length_ratio=0.32,
        arrow_length_min_ratio=0.12,
        arrow_length_max_ratio=0.55,
        arrow_head_scale=12.0,
        arrow_width_min=3.0,
        arrow_width_max=10.0,
        arrow_alpha_min=1.0,
        arrow_alpha_max=1.0,
        rho_color="tab:purple",
        zero_threshold=1e-15,
        save_fig=False,
        filename=None,
        dpi=300,
    ):
        """
        Evolve rho0 for n_steps and call plot_rho on the result.

        Returns
        -------
        rho_n, fig, ax
        """
        rho_n = self.evolve_rho(
            rho0=rho0, n_steps=n_steps, k=k,
            normalize_input=normalize_input,
        )
        if title is None:
            title = f"{self.name} — rho after {n_steps} step(s)"
        if filename is None:
            filename = f"rho_after_{int(n_steps)}_steps.pdf"

        fig, ax = self.plot_rho(
            rho=rho_n, ax=ax, title=title, figsize=figsize,
            rho_max=rho_max,
            bond_offset_ratio_min=bond_offset_ratio_min,
            bond_offset_ratio_max=bond_offset_ratio_max,
            arrow_length_mode=arrow_length_mode,
            constant_arrow_length_frac=constant_arrow_length_frac,
            arrow_length_ratio=arrow_length_ratio,
            arrow_length_min_ratio=arrow_length_min_ratio,
            arrow_length_max_ratio=arrow_length_max_ratio,
            arrow_head_scale=arrow_head_scale,
            arrow_width_min=arrow_width_min,
            arrow_width_max=arrow_width_max,
            arrow_alpha_min=arrow_alpha_min,
            arrow_alpha_max=arrow_alpha_max,
            rho_color=rho_color,
            zero_threshold=zero_threshold,
            save_fig=save_fig,
            filename=filename,
            dpi=dpi,
        )
        return rho_n, fig, ax
