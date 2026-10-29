#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WavefunctionPlotsMixin — mode_localization and plot_wavefunction of a closed
graph; plot_entropy and plot_ipr come from
:class:`quantum_graphs.plotting.localization_plots.LocalizationPlotsMixin`.
"""

import numpy as np

from ...core.config import show_figure
from ...plotting.localization_plots import LocalizationPlotsMixin, plot_bond_intensity, select_modes
from ...plotting.style import save_figure


class WavefunctionPlotsMixin(LocalizationPlotsMixin):
    """Localisation of the levels of a closed graph and their bond intensities."""

    def mode_localization(self, wavelength_range=None):
        """
        Normalised entropy and IPR of every level of ``compute_spectrum``
        (optionally restricted to ``wavelength_range``), per bond; a degenerate
        level counts once, through the diagonal of its eigenspace projector.

        Returns
        -------
        k, S_norm, ipr : ndarray, sorted by k
        """
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum. Run compute_spectrum() first.")
        k_all = np.real(np.asarray(self.spectrum, dtype=float))
        g_all = np.asarray(self.degeneracies, dtype=int)
        keep = select_modes(k_all, wavelength_range)
        if not keep.any():
            raise ValueError("No level in the requested wavelength range.")
        order = np.argsort(k_all[keep], kind="stable")
        k_all, g_all = k_all[keep][order], g_all[keep][order]

        k_list, S_list, I_list = [], [], []
        for k, g in zip(k_all, g_all):
            vecs = self.null_vectors(k, multiplicity=g)
            if vecs.size == 0:
                continue
            _, S_norm, ipr = self.localization_measures(vecs)
            if not np.isfinite(ipr):
                continue
            k_list.append(k)
            S_list.append(S_norm)
            I_list.append(ipr)
        return np.asarray(k_list), np.asarray(S_list), np.asarray(I_list)

    def plot_wavefunction(self, a_in, k, cmap=None, save_fig=False, file_format="pdf",
                          show=None):
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
        save_fig, file_format, show

        Returns
        -------
        list of (fig, ax), one per vector.
        """
        a_in = np.atleast_2d(np.asarray(a_in))
        n_modes = a_in.shape[0]
        wavelength = 2 * np.pi / k
        figures = []
        for mode, coeffs in enumerate(a_in):
            S, S_norm, _ = self.localization_measures(coeffs)
            mode_label = f"Mode {mode + 1}, " if n_modes > 1 else ""
            title = (f"{mode_label}"
                     f"$k = {k:.4f}\\,\\mu m^{{-1}}$, "
                     f"$\\lambda = {wavelength:.3f}\\,\\mu m$, "
                     f"$S = {S:.3f}$, $S_{{\\rm norm}} = {S_norm:.3f}$")
            fig, ax = plot_bond_intensity(self, coeffs, title, cmap)
            if save_fig:
                base = f"wavefunction_mode{mode + 1}" if n_modes > 1 else "wavefunction"
                save_figure(fig, self.save_path,
                            f"{base}_lambda_{wavelength:.3f}_S_{S:.3f}_Snorm_{S_norm:.3f}"
                            f".{file_format}", "Wavefunction plot")
            show_figure(fig, show)
            figures.append((fig, ax))
        return figures
