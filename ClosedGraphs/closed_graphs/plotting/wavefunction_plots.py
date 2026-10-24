#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WavefunctionPlotsMixin — plot_wavefunction, plot_entropy, plot_ipr.

``plot_entropy`` and ``plot_ipr`` share the per-mode loop in
``_compute_mode_quantities``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        N_b    = int(self.n_internal_bonds)

        S_norm_list  = []
        I_list       = []
        x_kL_list    = []
        kept_k_list  = []

        for k in k_all:
            coeffs_set = self.compute_wavefunction(k, tol=1e-4)
            if coeffs_set is None or coeffs_set.size == 0:
                continue

            a = coeffs_set[0] if coeffs_set.ndim > 1 else coeffs_set
            a = np.asarray(a).ravel()

            if a.size != 2 * N_b:
                print(f"⚠ Skipping k={k:.6f}: unexpected coeff length {a.size}.")
                continue

            rho_pos = np.abs(a[:N_b])**2
            rho_neg = np.abs(a[N_b:])**2
            rho     = rho_pos + rho_neg
            total   = float(np.sum(rho))
            if total <= 0:
                continue

            p = rho / total

            # Shannon entropy
            m = p > 0
            S = -np.sum(p[m] * np.log(p[m]))
            S_norm = S / np.log(N_b) if N_b > 1 else 0.0

            # IPR
            I = float(N_b * np.sum(p**2))

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
        Plot energy density on the closed graph for one or more eigenvectors.

        Parameters
        ----------
        a_in : ndarray
            Wavefunction coefficients from ``compute_wavefunction(k)``.
            Shape (n_modes, n_directed_bonds) or (n_directed_bonds,).
        k : float
        cmap : str, Colormap, or None
        save_fig : bool
        file_format : str
        """
        from matplotlib.colors import LinearSegmentedColormap

        a_in = np.asarray(a_in)
        if a_in.ndim == 1:
            a_in = a_in[np.newaxis, :]
        N_modes    = a_in.shape[0]
        wavelength = 2 * np.pi / k

        edgelist_ordered = [(a[0], b[0]) for (a, b) in self.updated_bonds]

        if cmap is None:
            base     = plt.get_cmap("Reds")
            cmap_obj = LinearSegmentedColormap.from_list(
                "Reds_truncated",
                [base(x) for x in np.linspace(0.0, 0.75, 256)],
            )
        elif isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = cmap

        for mode in range(N_modes):
            fig, ax = plt.subplots(figsize=(10, 8))

            coeffs   = a_in[mode]
            rho_pos  = np.abs(coeffs[:self.n_internal_bonds])**2
            rho_neg  = np.abs(coeffs[self.n_internal_bonds:])**2
            energy_raw = rho_pos + rho_neg

            total_energy = np.sum(energy_raw)
            if total_energy > 0:
                p     = energy_raw / total_energy
                m     = p > 0
                S     = -np.sum(p[m] * np.log(p[m]))
                N_b   = energy_raw.size
                S_norm = S / np.log(N_b) if N_b > 1 else 0.0
            else:
                S = S_norm = 0.0

            max_energy = np.max(energy_raw) if energy_raw.size else 0.0
            energy_densities = energy_raw / max_energy if max_energy > 0 else energy_raw

            edge_widths   = 6 + 8 * energy_densities
            border_widths = edge_widths + 2

            pos         = nx.get_node_attributes(self.G, "pos")
            real_nodes  = [i for i in self.G.nodes if i not in self.virtual_vertices]
            virt_nodes  = self.virtual_vertices

            nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                                   node_size=900, node_color="black", ax=ax)
            nx.draw_networkx_nodes(self.G, pos, nodelist=virt_nodes,
                                   node_size=400, node_color="black", ax=ax)
            node_labels = {i: f"{i + 1}" for i in real_nodes}
            nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                    font_color="white", font_weight="bold",
                                    font_size=20, ax=ax)

            nx.draw_networkx_edges(self.G, pos, ax=ax,
                                   edgelist=edgelist_ordered,
                                   width=border_widths, edge_color="black")
            edges = nx.draw_networkx_edges(self.G, pos, ax=ax,
                                           edgelist=edgelist_ordered,
                                           width=edge_widths,
                                           edge_color=energy_densities,
                                           edge_cmap=cmap_obj,
                                           edge_vmin=0.0, edge_vmax=1.0)

            # Core-matplotlib inset instead of mpl_toolkits.axes_grid1: a stale
            # system mpl_toolkits can shadow the pip one and break append_axes.
            cax     = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
            cbar    = plt.colorbar(edges, cax=cax)
            cbar.set_label("Energy Density", fontsize=22)
            cbar.ax.tick_params(labelsize=22)

            mode_label = f"Mode {mode + 1}, " if N_modes > 1 else ""
            title = (f"{mode_label}"
                     f"$k = {k:.4f}\\,\\mu m^{{-1}}$, "
                     f"$\\lambda = {wavelength:.3f}\\,\\mu m$, "
                     f"$S = {S:.3f}$, $S_{{\\rm norm}} = {S_norm:.3f}$")
            ax.set_title(title, fontsize=20)
            ax.set_aspect("equal")
            ax.axis("off")
            plt.tight_layout(pad=3.0)

            if save_fig:
                base_str = f"wavefunction_mode{mode+1}" if N_modes > 1 else "wavefunction"
                fname    = (f"{base_str}_lambda_{wavelength:.3f}"
                            f"_S_{S:.3f}_Snorm_{S_norm:.3f}.{file_format}")
                fpath    = os.path.join(self.save_path, fname)
                plt.savefig(fpath, format=file_format, dpi=300, bbox_inches="tight")
                print(f"✓ Wavefunction plot saved to {fpath}")
            plt.show()

    # =========================================================================
    # Entropy
    # =========================================================================

    def plot_entropy(self, wavelength_range=None, show_smooth=False,
                     smooth_window=11, save_fig=False, file_format="pdf"):
        """
        Normalised Shannon entropy S_norm vs k·L_tot and vs mode index.

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
        print(f"⟨S_norm⟩ = {Sn_mean:.6f} ± {Sn_std:.6f}   (N={S_norm_vals.size}, {title_range})")

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
                print(f"✓ Entropy ({xtag}) saved to: {fpath}")
            plt.show()

    # =========================================================================
    # IPR
    # =========================================================================

    def plot_ipr(self, wavelength_range=None, show_smooth=False,
                 smooth_window=20, save_fig=False, file_format="pdf"):
        """
        Inverse participation ratio I = N_b · Σ p_b² vs k·L_tot and vs mode index.

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
        print(f"⟨I⟩ = {I_mean:.6f} ± {I_std:.6f}   (N={I_vals.size}, {title_range})")

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
                print(f"✓ IPR ({xtag}) saved to: {fpath}")
            plt.show()
