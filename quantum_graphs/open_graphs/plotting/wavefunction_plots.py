#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WavefunctionPlotsMixin — resonance_vectors, mode_localization and
plot_resonance_wavefunction of an open graph; plot_entropy and plot_ipr come
from :class:`quantum_graphs.plotting.localization_plots.LocalizationPlotsMixin`.

The "wavefunction" at a complex resonance k is the null space of I − U(k),
U = D(k) Σ_αα(k) being the internal bond-scattering operator (leads closed).
"""

import numpy as np

from ...core.config import show_figure
from ...plotting.localization_plots import LocalizationPlotsMixin, plot_bond_intensity, select_modes
from ...plotting.style import save_figure


class WavefunctionPlotsMixin(LocalizationPlotsMixin):
    """Localisation of the resonances of an open graph and their bond intensities."""

    def _require_resonances(self):
        if not hasattr(self, "spectrum") or np.asarray(self.spectrum).size == 0:
            raise AttributeError("No complex spectrum available. Run find_resonances() first.")

    def resonance_vectors(self, index):
        """Orthonormal basis of ker(I − U(k)) at resonance ``self.spectrum[index]``,
        with as many vectors as its multiplicity (rows; see ``null_vectors``)."""
        g = max(int(self.degeneracies[index]), 1)
        return self.null_vectors(complex(self.spectrum[index]), multiplicity=g)

    def mode_localization(self, wavelength_range=None):
        """
        Normalised entropy and IPR of every resonance of ``find_resonances``
        (optionally restricted to ``wavelength_range`` on Re k), per bond; a
        degenerate resonance counts once, through the diagonal of the
        projector onto its null space.

        Returns
        -------
        k (complex), S_norm, ipr : ndarray, sorted by Re k
        """
        self._require_resonances()
        spectrum = np.asarray(self.spectrum, dtype=complex)
        index = np.flatnonzero(select_modes(spectrum, wavelength_range))
        index = index[np.argsort(np.real(spectrum[index]), kind="stable")]
        measures = [self.localization_measures(self.resonance_vectors(i)) for i in index]
        return (spectrum[index],
                np.asarray([m[1] for m in measures], dtype=float),
                np.asarray([m[2] for m in measures], dtype=float))

    def plot_resonance_wavefunction(self, index, cmap=None, save_fig=False, file_format="pdf",
                                    show=None):
        """
        Plot the wavefunction(s) at the complex resonance ``self.spectrum[index]``:
        one figure per vector of the null space of I − U(k) (as many as its
        multiplicity), each bond coloured by its intensity. Leads are not
        shown; S_norm in the title is that of the vector.

        Parameters
        ----------
        index : int
            Index into self.spectrum.
        cmap : str, Colormap, or None
        save_fig, file_format, show

        Returns
        -------
        list of (fig, ax), one per vector.
        """
        self._require_resonances()
        k = complex(self.spectrum[index])
        wavelength = 2 * np.pi / k.real
        Q = k.real / (2 * abs(k.imag)) if k.imag != 0 else np.inf
        Q_str = f"{Q:.1f}" if np.isfinite(Q) else r"\infty"

        vecs = self.resonance_vectors(index)
        figures = []
        for mode, a in enumerate(vecs):
            _, S_norm, _ = self.localization_measures(a)
            mode_label = f"Mode {mode + 1}, " if len(vecs) > 1 else ""
            title = (f"{mode_label}"
                     f"$\\lambda = {wavelength:.3f}\\,\\mu m$, "
                     f"$Q = {Q_str}$, "
                     f"$S_{{\\rm norm}} = {S_norm:.3f}$")
            fig, ax = plot_bond_intensity(self, a, title, cmap)
            if save_fig:
                base = f"resonance_wf_mode{mode + 1}" if len(vecs) > 1 else "resonance_wf"
                save_figure(fig, self.save_path,
                            f"{base}_idx{index}_lambda{wavelength:.3f}_Q{Q:.0f}"
                            f"_Snorm{S_norm:.3f}.{file_format}", "Resonance wavefunction")
            show_figure(fig, show)
            figures.append((fig, ax))
        return figures
