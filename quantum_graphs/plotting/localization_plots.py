"""
LocalizationPlotsMixin — ``plot_entropy`` and ``plot_ipr`` of closed and open
graphs, and the bond-intensity figure of one mode (``plot_bond_intensity``).

The graph class provides ``mode_localization(wavelength_range)`` → (k, S_norm,
IPR) per mode, sorted by Re k: levels of a closed graph, resonances of an open
one. Entropy and IPR are computed per bond, on the eigenspace projector for a
degenerate mode (:mod:`quantum_graphs.stats.localization`).
"""

import matplotlib.pyplot as plt
import numpy as np

from ..core.config import log, show_figure
from .drawing import add_colorbar, draw_bond_values, draw_vertices, finish_axes, size_scale
from .style import box_ticks, save_figure


def moving_average(y, window):
    """Centred moving average over an odd window of at least 3 points (edges
    padded with the end values), or None if ``y`` is too short."""
    if y.size < 3:
        return None
    w = max(int(window), 3)
    if w % 2 == 0:
        w += 1
    if w > y.size:
        w = y.size if y.size % 2 == 1 else y.size - 1
        if w < 3:
            return None
    pad = w // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(w) / w, mode="valid")


def select_modes(k, wavelength_range):
    """Mask of the finite modes whose wavelength 2π/Re k lies in ``wavelength_range``."""
    k_re = np.real(np.asarray(k))
    keep = np.isfinite(k_re)
    if wavelength_range is not None:
        lam_min, lam_max = wavelength_range
        with np.errstate(divide="ignore"):
            lam = 2 * np.pi / k_re
        keep &= (lam >= lam_min) & (lam <= lam_max)
    return keep


def plot_bond_intensity(graph, vec, title, cmap=None):
    """Figure of one mode: each bond coloured by its intensity |a+|² + |a-|²
    (normalised to the maximum), with the vertices. Returns ``(fig, ax)``."""
    rho = graph.bond_intensities(vec)
    rho_max = np.max(rho) if rho.size else 0.0
    paths = graph.bond_paths
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = draw_bond_values(ax, paths, rho / rho_max if rho_max > 0 else rho, cmap=cmap)
    draw_vertices(ax, graph.topology, 0.065 * size_scale(graph.topology))
    add_colorbar(ax, edges, "Energy Density")
    ax.set_title(title, fontsize=20)
    finish_axes(ax, paths)
    fig.tight_layout(pad=3.0)
    return fig, ax


class LocalizationPlotsMixin:
    """plot_entropy and plot_ipr from the class's ``mode_localization``."""

    def plot_entropy(self, wavelength_range=None, show_smooth=False, smooth_window=11,
                     show_mean_hline=True, save_fig=False, file_format="pdf", show=None):
        """
        Normalised Shannon entropy S_norm = S / ln N_b of every mode, vs
        Re(k)·L_tot and vs mode index (two figures).

        Parameters
        ----------
        wavelength_range : (float, float) or None
            Restrict to the modes in this window (µm).
        show_smooth : bool       Overlay a moving average over ``smooth_window`` modes.
        show_mean_hline : bool   Red dashed line at the mean.
        save_fig, file_format, show

        Returns
        -------
        list of (fig, ax): vs Re(k)·L_tot, vs index.
        """
        k, S_norm, _ = self.mode_localization(wavelength_range)
        return self._plot_mode_measure(
            k, S_norm, "entropy", r"$S_{\rm norm}$", r"S_{\rm norm}", wavelength_range,
            show_smooth, smooth_window, show_mean_hline, save_fig, file_format, show,
            hlines=((0.0, "-."), (1.0, "-.")), ylim=(-0.03, 1.03))

    def plot_ipr(self, wavelength_range=None, show_smooth=False, smooth_window=20,
                 show_mean_hline=True, save_fig=False, file_format="pdf", show=None):
        """
        Inverse participation ratio I = N_b · Σ p_b² of every mode, vs
        Re(k)·L_tot and vs mode index (two figures); the dashed line I = 2 is
        the value of a random (Gaussian) real vector.

        Parameters and returns: as ``plot_entropy``.
        """
        k, _, ipr = self.mode_localization(wavelength_range)
        return self._plot_mode_measure(
            k, ipr, "ipr", r"$IPR$", "I", wavelength_range,
            show_smooth, smooth_window, show_mean_hline, save_fig, file_format, show,
            hlines=((2.0, "--"),), ylim=None)

    def _plot_mode_measure(self, k, values, name, ylabel, symbol, wavelength_range,
                           show_smooth, smooth_window, show_mean_hline, save_fig, file_format,
                           show, hlines, ylim):
        if values.size == 0:
            raise ValueError(f"No mode for the {name} plot.")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
        if wavelength_range is None:
            title_range, range_tag = "", ""
        else:
            lam_min, lam_max = wavelength_range
            title_range = fr"   ($\lambda \in [{lam_min:.3f},\,{lam_max:.3f}]\,\mu m$)"
            range_tag = f"__lam{lam_min:.3f}-{lam_max:.3f}"
        log(f"⟨{'S_norm' if name == 'entropy' else 'IPR'}⟩ = {mean:.6f} ± {std:.6f}   "
            f"(N={values.size})")

        smooth = moving_average(values, smooth_window) if show_smooth else None
        smooth_tag = f"__smoothW{smooth_window}" if smooth is not None else ""
        if ylim is None:
            lo, hi = float(np.min(values)), float(np.max(values))
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            ylim = (max(0.0, lo - pad), hi + pad)
        kL = np.real(k) * self.total_length
        kL_label = (r"$\mathrm{Re}(k)\,L_{\rm tot}$" if np.iscomplexobj(k)
                    else r"$k\,L_{\rm tot}$")

        figures = []
        for x, xlabel, xtag in ((kL, kL_label, "kL"),
                                (np.arange(values.size), "Mode index", "index")):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(x, values, "ko-", markersize=5, linewidth=1)
            for y, style in hlines:
                ax.axhline(y, color="black", linestyle=style, linewidth=1.5)
            if show_mean_hline:
                ax.axhline(mean, color="red", linestyle="--", linewidth=1.5)
            if smooth is not None:
                ax.plot(x, smooth, "b-", linewidth=2)
            ax.set_title(fr"$\langle {symbol}\rangle = {mean:.3f}\ \pm\ {std:.3f}$"
                         + title_range, fontsize=20)
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel(ylabel, fontsize=22)
            ax.set_xlim(float(x[0]), float(x[-1]) if x.size > 1 else float(x[0]) + 1)
            ax.set_ylim(*ylim)
            box_ticks(ax, labelsize=18)
            fig.tight_layout()
            if save_fig:
                save_figure(fig, self.save_path,
                            f"{name}_vs_{xtag}{range_tag}{smooth_tag}.{file_format}",
                            f"{name} ({xtag})")
            show_figure(fig, show)
            figures.append((fig, ax))
        return figures
