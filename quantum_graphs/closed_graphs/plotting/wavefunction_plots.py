#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WavefunctionPlotsMixin — plot_wavefunction, plot_entropy, plot_ipr.

``plot_entropy`` and ``plot_ipr`` share the per-mode loop in
``_compute_mode_quantities``. Entropy and IPR are computed per bond,
on the eigenspace projector for degenerate levels
(:mod:`quantum_graphs.stats.localization`).
"""

from ...core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt

from ...plotting.drawing import (add_colorbar, draw_bond_values, draw_vertices, finish_axes,
                                size_scale)
from .style import MAJOR_TICK_LEN, MINOR_TICK_LEN, MAJOR_TICK_W, MINOR_TICK_W


def _tick_params(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in",
                   length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=18, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                   top=True, right=True)


def _moving_average(y, window):
    if y.size < 3:
        return None
    w = int(window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > y.size:
        w = y.size if y.size % 2 == 1 else y.size - 1
        if w < 3:
            return None
    kernel = np.ones(w, dtype=float) / w
    pad    = w // 2
    ypad   = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


class WavefunctionPlotsMixin:

    # =========================================================================
    # Shared helper
    # =========================================================================

    def _compute_mode_quantities(self, wavelength_range=None):
        """
        Iterate over the spectrum and compute per-mode Shannon entropy and IPR.

        Per bond; a degenerate level counts once, through the
        diagonal of its eigenspace projector.

        Parameters
        ----------
        wavelength_range : (float, float) or None

        Returns
        -------
        kept_k       : ndarray
        x_kL_vals    : ndarray   k * L_tot
        S_norm_vals  : ndarray
        I_vals       : ndarray   IPR = N_b * Σ p_b²
        """
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum. Run compute_spectrum() first.")

        k_all = np.sort(np.real(np.asarray(self.spectrum, dtype=float)))
        k_all = k_all[np.isfinite(k_all)]
        if k_all.size == 0:
            raise ValueError("Empty spectrum.")

        if wavelength_range is not None:
            lam_min, lam_max = wavelength_range
            lam_all = 2 * np.pi / k_all
            k_all   = k_all[(lam_all >= lam_min) & (lam_all <= lam_max)]

        if k_all.size == 0:
            raise ValueError("No modes in requested wavelength range.")

        L_tot  = float(self.total_length)

        S_norm_list  = []
        I_list       = []
        x_kL_list    = []
        kept_k_list  = []

        for k in k_all:
            coeffs_set = self.compute_wavefunction(k, tol=1e-4)
            if coeffs_set is None or coeffs_set.size == 0:
                continue

            _, S_norm, I = self.localization_measures(coeffs_set)
            if not np.isfinite(I):
                continue

            S_norm_list.append(S_norm)
            I_list.append(I)
            x_kL_list.append(k * L_tot)
            kept_k_list.append(k)

        kept_k      = np.asarray(kept_k_list,  dtype=float)
        x_kL_vals   = np.asarray(x_kL_list,    dtype=float)
        S_norm_vals = np.asarray(S_norm_list,   dtype=float)
        I_vals      = np.asarray(I_list,        dtype=float)

        order       = np.argsort(kept_k)
        return (kept_k[order], x_kL_vals[order],
                S_norm_vals[order], I_vals[order])

    # =========================================================================
    # Wavefunction plot
    # =========================================================================

    def plot_wavefunction(self, a_in, k, cmap=None, save_fig=False, file_format="pdf"):
        """
        Plot the intensity on the bonds of the closed graph for one or more
        eigenvectors: each bond is coloured by |a+|² + |a-|² (normalised to its
        maximum); S and S_norm in the title are those of that vector.

        Parameters
        ----------
        a_in : ndarray
            Wavefunction coefficients from ``compute_wavefunction(k)``.
            Shape (n_modes, 2 n_bonds) or (2 n_bonds,).
        k : float
        cmap : str, Colormap, or None
        save_fig : bool
        file_format : str
        """
        a_in = np.asarray(a_in)
        if a_in.ndim == 1:
            a_in = a_in[np.newaxis, :]
        N_modes    = a_in.shape[0]
        wavelength = 2 * np.pi / k
        paths      = self.bond_paths
        radius     = 0.065 * size_scale(self.topology)

        for mode in range(N_modes):
            fig, ax = plt.subplots(figsize=(10, 8))

            coeffs     = a_in[mode]
            energy_raw = self.bond_intensities(coeffs)
            S, S_norm, _ = self.localization_measures(coeffs)

            max_energy = np.max(energy_raw) if energy_raw.size else 0.0
            energy_densities = energy_raw / max_energy if max_energy > 0 else energy_raw

            edges = draw_bond_values(ax, paths, energy_densities, cmap=cmap)
            draw_vertices(ax, self.topology, radius)
            add_colorbar(ax, edges, "Energy Density")

            mode_label = f"Mode {mode + 1}, " if N_modes > 1 else ""
            title = (f"{mode_label}"
                     f"$k = {k:.4f}\\,\\mu m^{{-1}}$, "
                     f"$\\lambda = {wavelength:.3f}\\,\\mu m$, "
                     f"$S = {S:.3f}$, $S_{{\\rm norm}} = {S_norm:.3f}$")
            ax.set_title(title, fontsize=20)
            finish_axes(ax, paths)
            plt.tight_layout(pad=3.0)

            if save_fig:
                base_str = f"wavefunction_mode{mode+1}" if N_modes > 1 else "wavefunction"
                fname    = (f"{base_str}_lambda_{wavelength:.3f}"
                            f"_S_{S:.3f}_Snorm_{S_norm:.3f}.{file_format}")
                fpath    = os.path.join(self.save_path, fname)
                plt.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
                log(f"✓ Wavefunction plot saved to {fpath}")
            show_figure()

    # =========================================================================
    # Entropy
    # =========================================================================

    def plot_entropy(self, wavelength_range=None, show_smooth=False,
                     smooth_window=11, save_fig=False, file_format="pdf"):
        """
        Normalised Shannon entropy S_norm vs k·L_tot and vs mode index.

        Per bond, S_norm = S / ln N_b; a degenerate level counts once
        (eigenspace projector). See :mod:`quantum_graphs.stats.localization`.

        Parameters
        ----------
        wavelength_range : (float, float) or None
        show_smooth : bool     Overlay moving-average curve.
        smooth_window : int
        save_fig : bool
        file_format : str
        """
        kept_k, x_kL_vals, S_norm_vals, _ = self._compute_mode_quantities(wavelength_range)

        if x_kL_vals.size == 0:
            raise ValueError("No valid modes.")

        if wavelength_range is None:
            title_range = "full spectrum"
            range_tag   = "__full"
        else:
            lam_min, lam_max = wavelength_range
            title_range = fr"$\lambda \in [{lam_min:.3f},\,{lam_max:.3f}]\,\mu m$"
            range_tag   = f"__range_[{lam_min:.3f}-{lam_max:.3f}]"

        Sn_mean = float(np.mean(S_norm_vals))
        Sn_std  = float(np.std(S_norm_vals, ddof=1)) if S_norm_vals.size >= 2 else 0.0
        log(f"⟨S_norm⟩ = {Sn_mean:.6f} ± {Sn_std:.6f}   (N={S_norm_vals.size}, {title_range})")

        smooth_y   = _moving_average(S_norm_vals, smooth_window) if show_smooth else None
        smooth_tag = f"__smoothW{smooth_window}" if show_smooth and smooth_y is not None else ""
        stats_title = fr"$\langle S_{{\rm norm}}\rangle = {Sn_mean:.3f}\ \pm\ {Sn_std:.3f}$"

        for fig_type, xdata, xlabel, xlim, xtag in [
            ("kL",    x_kL_vals,           r"$k\,L_{\rm tot}$",
             (float(x_kL_vals.min()), float(x_kL_vals.max())), "kL"),
            ("index", np.arange(S_norm_vals.size), "Mode index",
             (0, max(1, S_norm_vals.size - 1)), "index"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(xdata, S_norm_vals, "ko-", markersize=5, linewidth=1)
            ax.axhline(0.0, color="black", linestyle="-.", linewidth=1.5)
            ax.axhline(1.0, color="black", linestyle="-.", linewidth=1.5)
            if smooth_y is not None:
                ax.plot(xdata, smooth_y, "r-", linewidth=2)
            ax.set_title(fr"{stats_title}   ({title_range})", fontsize=20)
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel(r"$S_{\rm norm}$", fontsize=22)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.03, 1.03)
            _tick_params(ax)
            fig.tight_layout()
            if save_fig:
                fname = f"entropy_vs_{xtag}{range_tag}{smooth_tag}.{file_format}"
                fpath = os.path.join(self.save_path, fname)
                fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
                log(f"✓ Entropy ({xtag}) saved to: {fpath}")
            show_figure()

    # =========================================================================
    # IPR
    # =========================================================================

    def plot_ipr(self, wavelength_range=None, show_smooth=False,
                 smooth_window=20, save_fig=False, file_format="pdf"):
        """
        Inverse participation ratio I = N_b · Σ p_b² vs k·L_tot and vs mode index.

        Per bond (N_b = number of bonds); a degenerate
        level counts once (eigenspace projector).

        Parameters
        ----------
        wavelength_range : (float, float) or None
        show_smooth : bool
        smooth_window : int
        save_fig : bool
        file_format : str
        """
        kept_k, x_kL_vals, _, I_vals = self._compute_mode_quantities(wavelength_range)

        if x_kL_vals.size == 0:
            raise ValueError("No valid modes.")

        if wavelength_range is None:
            title_range = "full spectrum"
            range_tag   = "__full"
        else:
            lam_min, lam_max = wavelength_range
            title_range = fr"$\lambda \in [{lam_min:.3f},\,{lam_max:.3f}]\,\mu m$"
            range_tag   = f"__range_[{lam_min:.3f}-{lam_max:.3f}]"

        I_mean = float(np.mean(I_vals))
        I_std  = float(np.std(I_vals, ddof=1)) if I_vals.size >= 2 else 0.0
        log(f"⟨I⟩ = {I_mean:.6f} ± {I_std:.6f}   (N={I_vals.size}, {title_range})")

        smooth_y   = _moving_average(I_vals, smooth_window) if show_smooth else None
        smooth_tag = f"__smoothW{smooth_window}" if show_smooth and smooth_y is not None else ""
        stats_title = fr"$\langle I\rangle = {I_mean:.3f}\ \pm\ {I_std:.3f}$"

        y_min = float(np.min(I_vals))
        y_max = float(np.max(I_vals))
        pad   = 0.05 * (y_max - y_min) if y_max > y_min else 0.1

        for fig_type, xdata, xlabel, xlim, xtag in [
            ("kL",    x_kL_vals,           r"$k\,L_{\rm tot}$",
             (float(x_kL_vals.min()), float(x_kL_vals.max())), "kL"),
            ("index", np.arange(I_vals.size), "Mode index",
             (0, max(1, I_vals.size - 1)), "index"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(xdata, I_vals, "ko-", markersize=5, linewidth=1)
            ax.axhline(2.0, color="black", linestyle="--", linewidth=2.0)
            if smooth_y is not None:
                ax.plot(xdata, smooth_y, "r-", linewidth=3.0)
            ax.set_title(fr"{stats_title}   ({title_range})", fontsize=20)
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel(r"$IPR$", fontsize=22)
            ax.set_xlim(*xlim)
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
            _tick_params(ax)
            fig.tight_layout()
            if save_fig:
                fname = f"ipr_vs_{xtag}{range_tag}{smooth_tag}.{file_format}"
                fpath = os.path.join(self.save_path, fname)
                fig.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
                log(f"✓ IPR ({xtag}) saved to: {fpath}")
            show_figure()
