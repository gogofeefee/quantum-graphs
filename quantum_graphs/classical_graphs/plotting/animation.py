#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting/animation.py — AnimationMixin for classical_graphs.

Animated visualisation of the Perron-Frobenius evolution:
  - "step" style: one frame per step, arrows centred on bonds.
  - "flow" style: smooth motion + fade-out/fade-in between steps.
"""

from ...core.config import log
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


class AnimationMixin:
    """Animated rho evolution on directed bonds."""

    # ==================================================================
    # Flow-frame state computation
    # ==================================================================

    def _compute_flow_frame_state(
        self,
        subframe_idx,
        n_subframes_move=12,
        n_subframes_fade=8,
    ):
        """
        Map a subframe index to (phase, tau_move, fade_out_factor, fade_in_factor).

        The transition rho_n -> rho_{n+1} has two phases:
        1. MOVE  : arrows slide from tail to head (tau goes 0 → 1).
        2. FADE  : outgoing arrows fade out at tau=1; incoming fade in at tau=0.

        Returns
        -------
        phase : str — "move" or "fade"
        tau_move : float
        fade_out_factor : float
        fade_in_factor : float
        """
        n_subframes_move = int(max(1, n_subframes_move))
        n_subframes_fade = int(max(1, n_subframes_fade))

        if subframe_idx < n_subframes_move:
            tau = subframe_idx / (n_subframes_move - 1) if n_subframes_move > 1 else 1.0
            return "move", float(tau), 1.0, 0.0

        local_idx = subframe_idx - n_subframes_move
        x = local_idx / (n_subframes_fade - 1) if n_subframes_fade > 1 else 1.0
        return "fade", 1.0, float(1.0 - x), float(x)

    # ==================================================================
    # Background graph helper
    # ==================================================================

    def _draw_background_graph(self, ax, title=None, show_port_markers=True):
        """Clear ax and redraw the background graph."""
        ax.clear()
        try:
            self.plot_graph(
                ax=ax, show_lengths=False, show_port_markers=show_port_markers,
                marker_radius_frac=0.04, marker_offset_frac=0.02,
                title=None, save_fig=False,
            )
        except TypeError:
            self.plot_graph(ax=ax, show_lengths=False, title=None, save_fig=False)

    # ==================================================================
    # Single-frame drawing
    # ==================================================================

    def _draw_rho_frame_on_axis(
        self,
        ax,
        rho_current,
        rho_next=None,
        style="flow",
        subframe_idx=0,
        n_subframes_move=12,
        n_subframes_fade=8,
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
        rho_color="tab:blue",
        zero_threshold=1e-15,
        step_label=None,
        tau_move_start=0.0,
        tau_move_end=1.0,
        title=None,
    ):
        """
        Draw one animation frame on ax.

        Style "step" : arrows centred (tau=0.5) — static snapshot.
        Style "flow" : subframe drives motion/fade transition from
                       rho_current to rho_next.
        """
        self._draw_background_graph(ax, title=title)

        rho_current = self.validate_rho(rho_current, normalize=False)
        if rho_next is not None:
            rho_next = self.validate_rho(rho_next, normalize=False)

        if rho_max is None:
            vals = [np.max(rho_current)]
            if rho_next is not None:
                vals.append(np.max(rho_next))
            rho_max = float(max(vals)) if vals else 1.0

        # Shared kwargs forwarded to _draw_arrows_for_rho
        arrow_kw = dict(
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

        if style not in {"flow", "step"}:
            raise ValueError("style must be 'flow' or 'step'.")

        # ----------------------------------------------------------
        # STEP: static, arrows centred
        # ----------------------------------------------------------
        if style == "step" or rho_next is None:
            bonds = [
                (d, float(rho_current[self.directed_bond_index[d]]), 0.5, 1.0)
                for d in self.directed_bonds
            ]
            self._draw_arrows_for_rho(ax=ax, bonds_to_draw=bonds, **arrow_kw)

        # ----------------------------------------------------------
        # FLOW: motion then fade
        # ----------------------------------------------------------
        else:
            phase, tau_move, fade_out, fade_in = self._compute_flow_frame_state(
                subframe_idx=subframe_idx,
                n_subframes_move=n_subframes_move,
                n_subframes_fade=n_subframes_fade,
            )
            tau_motion = float(
                tau_move_start + tau_move * (tau_move_end - tau_move_start)
            )

            if phase == "move":
                bonds = [
                    (d, float(rho_current[self.directed_bond_index[d]]), tau_motion, 1.0)
                    for d in self.directed_bonds
                ]
                self._draw_arrows_for_rho(ax=ax, bonds_to_draw=bonds, **arrow_kw)
            else:
                # Outgoing (fade out) at tau = 1
                bonds_out = [
                    (d, float(rho_current[self.directed_bond_index[d]]), 1.0, fade_out)
                    for d in self.directed_bonds
                ]
                self._draw_arrows_for_rho(ax=ax, bonds_to_draw=bonds_out, **arrow_kw)
                # Incoming (fade in) at tau = 0
                bonds_in = [
                    (d, float(rho_next[self.directed_bond_index[d]]), 0.0, fade_in)
                    for d in self.directed_bonds
                ]
                self._draw_arrows_for_rho(ax=ax, bonds_to_draw=bonds_in,
                                          base_zorder=7, **arrow_kw)

        if title:
            ax.set_title(title)
        if step_label is not None:
            ax.text(
                0.08, 0.97, step_label,
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="0.7", alpha=0.85),
                zorder=20, clip_on=False,
            )
        ax.set_aspect("equal")
        ax.axis("off")

    def _draw_settle_frame(
        self,
        ax,
        rho_final,
        tau_settle,
        rho_max,
        arrow_kw,
        title,
        step_label,
    ):
        """Draw the final settling phase (tau from 0 → 0.5, no fade)."""
        self._draw_background_graph(ax, title=title)
        rho_final = self.validate_rho(rho_final, normalize=False)
        bonds = [
            (d, float(rho_final[self.directed_bond_index[d]]), tau_settle, 1.0)
            for d in self.directed_bonds
        ]
        self._draw_arrows_for_rho(ax=ax, bonds_to_draw=bonds, rho_max=rho_max, **arrow_kw)
        if title:
            ax.set_title(title)
        ax.text(
            0.08, 0.97, step_label,
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="0.7", alpha=0.85),
            zorder=20, clip_on=False,
        )
        ax.set_aspect("equal")
        ax.axis("off")

    # ==================================================================
    # Public animate_rho
    # ==================================================================

    def animate_rho(
        self,
        rho0,
        n_steps,
        style="flow",
        k=None,
        rebuild=False,
        normalize_input=True,
        figsize=(10, 8),
        fps=15,
        n_subframes_move=12,
        n_subframes_fade=8,
        rho_max_mode="global",
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
        rho_color="tab:blue",
        zero_threshold=1e-15,
        title=None,
        repeat=False,
        blit=False,
        save_animation=False,
        filename=None,
        writer="mp4",
        dpi=200,
    ):
        """
        Animate the Perron-Frobenius evolution of rho.

        Flow animation convention
        ------------------------
        - First frame: rho0 centred (tau=0.5).
        - Each transition n→n+1: move phase (tau 0→1) + fade phase.
          * First transition starts from tau=0.5 (already centred).
        - Final settle: rho_n_steps moves from tau=0 → 0.5.

        Parameters
        ----------
        rho0 : array-like, shape (2B,)
        n_steps : int
        style : "flow" or "step"
        rho_max_mode : "global" | "fixed" | "frame"
        save_animation : bool
        writer : "mp4" or "gif"
        filename : str or None

        Returns
        -------
        anim : FuncAnimation
        trajectory : np.ndarray, shape (n_steps+1, 2B)
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0.")
        style = str(style).lower()
        if style not in {"flow", "step"}:
            raise ValueError("style must be 'flow' or 'step'.")
        writer = str(writer).lower()
        if writer not in {"mp4", "gif"}:
            raise ValueError("writer must be 'mp4' or 'gif'.")

        trajectory = self.evolve_rho_trajectory(
            rho0=rho0, n_steps=n_steps, k=k,
            rebuild=rebuild, normalize_input=normalize_input,
        )
        self.last_rho_trajectory = trajectory

        # rho_max strategy
        if rho_max_mode == "global":
            rho_max_global = float(np.max(trajectory)) if trajectory.size else 1.0
        elif rho_max_mode == "fixed":
            if rho_max is None:
                raise ValueError("rho_max must be provided when rho_max_mode='fixed'.")
            rho_max_global = float(rho_max)
        elif rho_max_mode == "frame":
            rho_max_global = None
        else:
            raise ValueError("rho_max_mode must be 'global', 'fixed', or 'frame'.")

        def _resolve_rho_max(rho_cur, rho_nxt=None):
            if rho_max_mode in {"global", "fixed"}:
                return rho_max_global
            vals = [np.max(rho_cur)]
            if rho_nxt is not None:
                vals.append(np.max(rho_nxt))
            return float(max(vals)) if vals else 1.0

        # Build frame records
        if style == "step":
            frame_records = [("step", n, 0) for n in range(n_steps + 1)]
        else:
            n_trans = int(max(1, n_subframes_move)) + int(max(1, n_subframes_fade))
            n_settle = int(max(1, n_subframes_move))
            frame_records = [("step", 0, 0)]
            for n in range(n_steps):
                for sub in range(n_trans):
                    frame_records.append(("flow", n, sub))
            if n_steps >= 1:
                for sub in range(n_settle):
                    frame_records.append(("final_settle", n_steps, sub))

        # Shared arrow kwargs (everything except rho_max and bonds_to_draw)
        arrow_kw = dict(
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

        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()

        def _update(frame_idx):
            mode, n, sub = frame_records[frame_idx]

            if mode == "step":
                rho_cur = trajectory[n]
                self._draw_rho_frame_on_axis(
                    ax=ax, rho_current=rho_cur, rho_next=None,
                    style="step", subframe_idx=0,
                    n_subframes_move=n_subframes_move,
                    n_subframes_fade=n_subframes_fade,
                    rho_max=_resolve_rho_max(rho_cur),
                    title=title, step_label=f"step = {n}",
                    **arrow_kw,
                )

            elif mode == "flow":
                rho_cur = trajectory[n]
                rho_nxt = trajectory[n + 1]
                tau_start = 0.5 if n == 0 else 0.0
                self._draw_rho_frame_on_axis(
                    ax=ax, rho_current=rho_cur, rho_next=rho_nxt,
                    style="flow", subframe_idx=sub,
                    n_subframes_move=n_subframes_move,
                    n_subframes_fade=n_subframes_fade,
                    rho_max=_resolve_rho_max(rho_cur, rho_nxt),
                    title=title,
                    step_label=f"step = {n} → {n + 1}",
                    tau_move_start=tau_start,
                    tau_move_end=1.0,
                    **arrow_kw,
                )

            elif mode == "final_settle":
                rho_final = trajectory[n]
                cur_rho_max = _resolve_rho_max(rho_final)
                n_settle = int(max(1, n_subframes_move))
                tau_local = sub / (n_settle - 1) if n_settle > 1 else 1.0
                tau_settle = 0.5 * tau_local
                self._draw_settle_frame(
                    ax=ax, rho_final=rho_final,
                    tau_settle=tau_settle,
                    rho_max=cur_rho_max,
                    arrow_kw=arrow_kw,
                    title=title,
                    step_label=f"step = {n} (final)",
                )
            else:
                raise ValueError(f"Unknown frame mode: {mode}")
            return []

        anim = FuncAnimation(
            fig, _update,
            frames=len(frame_records),
            interval=1000.0 / max(1, fps),
            repeat=repeat,
            blit=blit,
        )

        if save_animation:
            if filename is None:
                ext = "mp4" if writer == "mp4" else "gif"
                filename = f"rho_animation_{style}.{ext}"
            out = self.save_path / filename
            if writer == "mp4":
                try:
                    anim.save(out, writer=FFMpegWriter(fps=fps), dpi=dpi)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to save mp4 (is ffmpeg installed?). Error: {e}"
                    )
            else:
                anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
            log(f"✓ Animation saved to: {out}")

        return anim, trajectory
