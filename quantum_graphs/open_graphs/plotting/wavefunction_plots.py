#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WavefunctionPlotsMixin — plot_resonance_wavefunction, plot_entropy, plot_ipr.

For open graphs the "wavefunction" at a complex resonance k is the null-space
of (I − U_b(k)), where U_b = D(k) Σ_αα(k) is the internal bond-scattering
evolution operator.  Energy density on bond b is |a_fwd[b]|² + |a_bwd[b]|².
Entropy and IPR are computed per bond (amplitudes arriving at its two end
vertices), on the projector onto the null-space when it has dimension > 1
(:mod:`quantum_graphs.stats.localization`).

Helpers
-------
_compute_resonance_wavefunction_vectors(k)   null-space of (I − U_b(k))
_compute_bond_entropy_ipr(energy_per_bond)   Shannon entropy + IPR scalars
_compute_all_resonance_quantities()          iterate spectrum → (S_norm, I) arrays
"""

from ...core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt

from ...plotting.drawing import (add_colorbar, draw_bond_values, draw_vertices, finish_axes,
                                size_scale)
from ...stats.localization import localization_measures
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
    # Shared internal helpers
    # =========================================================================

    def _compute_resonance_wavefunction_vectors(self, k, nullity_tol=1e-3):
        """
        Compute the null-space of (I − U_b(k)).

        Parameters
        ----------
        k : complex
            Resonance wavenumber.
        nullity_tol : float
            Singular values < nullity_tol * s_max are treated as zero.

        Returns
        -------
        vecs : ndarray, shape (dim_kernel, 2B)
            Rows are orthonormal kernel vectors (conjugated rows of Vh).
        """
        U = self._U(k)
        A = np.eye(U.shape[0], dtype=np.complex128) - U
        _, s, Vh = np.linalg.svd(A)
        s_max = s[0] if s[0] > 0 else 1.0
        null_mask = (s / s_max) < nullity_tol
        null_indices = np.where(null_mask)[0]
        if len(null_indices) == 0:
            null_indices = np.array([len(s) - 1])
        return Vh[null_indices, :].conj()   # rows of Vh are v^H

    def _compute_bond_entropy_ipr(self, energy_per_bond):
        """
        Shannon entropy and IPR of a bond energy distribution.

        Parameters
        ----------
        energy_per_bond : ndarray, shape (N_b,)
            Raw (unnormalised) energy on each bond (``bond_intensities``).

        Returns
        -------
        S : float        raw Shannon entropy
        S_norm : float   S / log(N_b)
        I : float        IPR = N_b * Σ p_b²
        """
        return localization_measures(energy_per_bond)

    def _compute_all_resonance_quantities(self, nullity_tol=1e-3):
        """
        Iterate over self.spectrum and compute per-resonance entropy and IPR.

        Per bond; when the null-space has dimension > 1 (degenerate
        or nearly degenerate resonance, see ``nullity_tol``) the diagonal of
        its projector is used, not one arbitrary basis vector.

        Returns
        -------
        k_vals    : ndarray (complex)
        S_norm    : ndarray (float)
        I_vals    : ndarray (float)
        """
        if not hasattr(self, "spectrum") or np.asarray(self.spectrum).size == 0:
            raise AttributeError(
                "No complex spectrum available. Run find_resonances() first.")

        spectrum = np.asarray(self.spectrum)

        k_list = []
        S_list = []
        I_list = []

        for k in spectrum:
            vecs = self._compute_resonance_wavefunction_vectors(k, nullity_tol)
            _, S_norm, I = self.localization_measures(vecs)
            k_list.append(k)
            S_list.append(S_norm)
            I_list.append(I)

        return (np.asarray(k_list, dtype=complex),
                np.asarray(S_list, dtype=float),
                np.asarray(I_list, dtype=float))

    # =========================================================================
    # 2a — plot_resonance_wavefunction
    # =========================================================================

    def plot_resonance_wavefunction(self, index, nullity_tol=1e-3, cmap=None,
                                    save_fig=False, file_format="pdf"):
        """
        Plot the wavefunction(s) at complex resonance self.spectrum[index].

        Computes the null-space of (I − U_b(k)) at the selected resonance.
        One plot per kernel vector (degenerate resonances produce multiple plots).
        Leads are not shown. Each bond is coloured by its intensity; S_norm in
        the title is that of the vector.

        Parameters
        ----------
        index : int
            Index into self.spectrum.
        nullity_tol : float
            Threshold for null-space detection (relative to largest singular value).
        cmap : str, Colormap, or None
        save_fig : bool
        file_format : str
        """
        if not hasattr(self, "spectrum") or np.asarray(self.spectrum).size == 0:
            raise AttributeError(
                "No complex spectrum available. Run find_resonances() first.")

        k          = complex(self.spectrum[index])
        wavelength = 2 * np.pi / float(np.real(k))
        im_k       = float(np.imag(k))
        Q          = float(np.real(k)) / (2 * abs(im_k)) if im_k != 0 else np.inf

        vecs    = self._compute_resonance_wavefunction_vectors(k, nullity_tol)
        N_modes = len(vecs)
        paths   = self.bond_paths
        radius  = 0.065 * size_scale(self.topology)

        for mode_idx, a in enumerate(vecs):
            rho          = self.bond_intensities(a)
            _, S_norm, _ = self.localization_measures(a)

            max_e            = np.max(rho) if rho.size else 1.0
            energy_densities = rho / max_e if max_e > 0 else rho

            fig, ax = plt.subplots(figsize=(10, 8))
            edges = draw_bond_values(ax, paths, energy_densities, cmap=cmap)
            draw_vertices(ax, self.topology, radius)
            add_colorbar(ax, edges, "Energy Density")

            Q_str      = f"{Q:.1f}" if np.isfinite(Q) else r"\infty"
            mode_label = f"Mode {mode_idx + 1}, " if N_modes > 1 else ""
            title = (f"{mode_label}"
                     f"$\\lambda = {wavelength:.3f}\\,\\mu m$, "
                     f"$Q = {Q_str}$, "
                     f"$S_{{\\rm norm}} = {S_norm:.3f}$")
            ax.set_title(title, fontsize=20)
            finish_axes(ax, paths)
            plt.tight_layout(pad=3.0)

            if save_fig:
                base_str = f"resonance_wf_mode{mode_idx+1}" if N_modes > 1 \
                    else "resonance_wf"
                fname = (f"{base_str}_idx{index}_lambda{wavelength:.3f}"
                         f"_Q{Q:.0f}_Snorm{S_norm:.3f}.{file_format}")
                fpath = os.path.join(self.save_path, fname)
                plt.savefig(fpath, format=file_format, dpi=300,
                            bbox_inches="tight")
                log(f"✓ Resonance wavefunction saved to {fpath}")
            show_figure()

    # =========================================================================
    # 2c — plot_entropy
    # =========================================================================

    def plot_entropy(self, show_smooth=False, smooth_window=11,
                     show_mean_hline=True,
                     save_fig=False, file_format="pdf", nullity_tol=1e-3):
        """
        Normalised Shannon entropy S_norm vs Re(k)·L_tot and vs mode index.

        Per bond; see ``_compute_all_resonance_quantities``.

        Parameters
        ----------
        show_smooth : bool     Overlay a moving-average curve.
        smooth_window : int
        show_mean_hline : bool  Draw a red dashed horizontal line at the mean value.
        save_fig : bool
        file_format : str
        nullity_tol : float    Passed to _compute_resonance_wavefunction_vectors.
        """
        k_vals, S_norm_vals, _ = self._compute_all_resonance_quantities(nullity_tol)

        if S_norm_vals.size == 0:
            raise ValueError("No valid resonances for entropy plot.")

        x_kL_vals = np.real(k_vals) * self.total_length
        order      = np.argsort(np.real(k_vals))
        x_kL_vals  = x_kL_vals[order]
        S_norm_vals = S_norm_vals[order]

        Sn_mean = float(np.mean(S_norm_vals))
        Sn_std  = float(np.std(S_norm_vals, ddof=1)) if S_norm_vals.size >= 2 else 0.0
        log(f"⟨S_norm⟩ = {Sn_mean:.6f} ± {Sn_std:.6f}   (N={S_norm_vals.size})")

        smooth_y   = _moving_average(S_norm_vals, smooth_window) if show_smooth else None
        smooth_tag = f"__smoothW{smooth_window}" if show_smooth and smooth_y is not None else ""
        stats_title = fr"$\langle S_{{\rm norm}}\rangle = {Sn_mean:.3f}\ \pm\ {Sn_std:.3f}$"

        for fig_type, xdata, xlabel, xlim, xtag in [
            ("kL",    x_kL_vals,                 r"$\mathrm{Re}(k)\,L_{\rm tot}$",
             (float(x_kL_vals.min()), float(x_kL_vals.max())), "kL"),
            ("index", np.arange(S_norm_vals.size), "Mode index",
             (0, max(1, S_norm_vals.size - 1)), "index"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(xdata, S_norm_vals, "ko-", markersize=5, linewidth=1)
            ax.axhline(0.0, color="black", linestyle="-.", linewidth=1.5)
            ax.axhline(1.0, color="black", linestyle="-.", linewidth=1.5)
            if show_mean_hline:
                ax.axhline(Sn_mean, color="red", linestyle="--", linewidth=1.5)
            if smooth_y is not None:
                ax.plot(xdata, smooth_y, "b-", linewidth=2)
            ax.set_title(stats_title, fontsize=20)
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel(r"$S_{\rm norm}$", fontsize=22)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.03, 1.03)
            _tick_params(ax)
            fig.tight_layout()
            if save_fig:
                fname = f"entropy_vs_{xtag}{smooth_tag}.{file_format}"
                fpath = os.path.join(self.save_path, fname)
                fig.savefig(fpath, format=file_format, dpi=300,
                            bbox_inches="tight")
                log(f"✓ Entropy ({xtag}) saved to: {fpath}")
            show_figure()

    # =========================================================================
    # 2c — plot_ipr
    # =========================================================================

    def plot_ipr(self, show_smooth=False, smooth_window=20,
                 show_mean_hline=True,
                 save_fig=False, file_format="pdf", nullity_tol=1e-3):
        """
        Inverse participation ratio I = N_b · Σ p_b² vs Re(k)·L_tot and mode index.

        Per bond (N_b = number of bonds).

        Parameters
        ----------
        show_smooth : bool
        smooth_window : int
        show_mean_hline : bool  Draw a red dashed horizontal line at the mean value.
        save_fig : bool
        file_format : str
        nullity_tol : float
        """
        k_vals, _, I_vals = self._compute_all_resonance_quantities(nullity_tol)

        if I_vals.size == 0:
            raise ValueError("No valid resonances for IPR plot.")

        x_kL_vals = np.real(k_vals) * self.total_length
        order     = np.argsort(np.real(k_vals))
        x_kL_vals = x_kL_vals[order]
        I_vals    = I_vals[order]

        I_mean = float(np.mean(I_vals))
        I_std  = float(np.std(I_vals, ddof=1)) if I_vals.size >= 2 else 0.0
        log(f"⟨I⟩ = {I_mean:.6f} ± {I_std:.6f}   (N={I_vals.size})")

        smooth_y   = _moving_average(I_vals, smooth_window) if show_smooth else None
        smooth_tag = f"__smoothW{smooth_window}" if show_smooth and smooth_y is not None else ""
        stats_title = fr"$\langle I\rangle = {I_mean:.3f}\ \pm\ {I_std:.3f}$"

        y_min = float(np.min(I_vals))
        y_max = float(np.max(I_vals))
        pad   = 0.05 * (y_max - y_min) if y_max > y_min else 0.1

        for fig_type, xdata, xlabel, xlim, xtag in [
            ("kL",    x_kL_vals,                 r"$\mathrm{Re}(k)\,L_{\rm tot}$",
             (float(x_kL_vals.min()), float(x_kL_vals.max())), "kL"),
            ("index", np.arange(I_vals.size), "Mode index",
             (0, max(1, I_vals.size - 1)), "index"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(xdata, I_vals, "ko-", markersize=5, linewidth=1)
            ax.axhline(2.0, color="black", linestyle="--", linewidth=2.0)
            if show_mean_hline:
                ax.axhline(I_mean, color="red", linestyle="--", linewidth=1.5)
            if smooth_y is not None:
                ax.plot(xdata, smooth_y, "b-", linewidth=3.0)
            ax.set_title(stats_title, fontsize=20)
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel(r"$IPR$", fontsize=22)
            ax.set_xlim(*xlim)
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
            _tick_params(ax)
            fig.tight_layout()
            if save_fig:
                fname = f"ipr_vs_{xtag}{smooth_tag}.{file_format}"
                fpath = os.path.join(self.save_path, fname)
                fig.savefig(fpath, format=file_format, dpi=300,
                            bbox_inches="tight")
                log(f"✓ IPR ({xtag}) saved to: {fpath}")
            show_figure()
