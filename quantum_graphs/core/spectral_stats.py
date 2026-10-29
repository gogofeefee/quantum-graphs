"""
SpectralStatsMixin — the levels used by the spectral statistics and their
Weyl law, shared by closed graphs (``compute_spectrum``) and open graphs
(real parts of the resonances of ``find_resonances``).

Which levels (the ``sector`` argument)
--------------------------------------
The spectrum is the union of the spectra of the sectors of the graph
(:mod:`quantum_graphs.core.sectors`); two reciprocal partner sectors carry the
same spectrum, so every level of theirs is doubly degenerate. Statistics are
only meaningful on the spectrum of one sector, so they use:

* ``sector=None`` (default): the **desymmetrised spectrum**, the levels of
  the independent sectors (one per pair of partners), each counted with its
  multiplicity inside its sector. A graph whose sectors carry several
  independent spectra gives a superposition, and a warning says so;
* ``sector=i``: the spectrum of sector ``i`` (and of its partner);
* ``sector="all"``: every level repeated by its full multiplicity.

The Weyl law of the selection is N(k) = (L_sel / 2π) ∫ n_g dk, L_sel the
summed length of the directed bonds of the selected sectors (2 L for a graph
made of one sector, L for one of two partner sectors): no ad hoc factor.

An explicit ``spectrum`` (measured peaks, a spectrum computed elsewhere) is
taken as given, one entry per level, and unfolded with the Weyl law of the
selection: give the peaks of a graph with partner sectors once per pair.

The multiplicities measured at the levels are a check: a level degenerate
*inside* a sector (geometric symmetry, accidental near-degeneracy) is
reported when the spectrum is computed, and appears as a zero spacing.
"""

import warnings

import numpy as np

from ..stats.spectral import number_variance, spectral_rigidity, weyl_staircase


class SpectralStatsMixin:

    # =========================================================================
    # Levels of the computed spectrum
    # =========================================================================

    def _set_levels(self, spectrum, degeneracies, level_sectors):
        """Store a computed spectrum: ``spectrum`` (one entry per distinct
        level), ``degeneracies`` (full multiplicity) and ``level_sectors``
        (representative sector of each level)."""
        self.spectrum = spectrum
        self.degeneracies = np.asarray(degeneracies, dtype=int)
        self.level_sectors = np.asarray(level_sectors, dtype=int)
        zero = spectrum[self.degeneracies == 0]
        if zero.size:
            warnings.warn(
                f"{zero.size} resonance(s) have multiplicity 0 (no singular value of I − U below "
                f"nullity_tol): loosen nullity_tol or tighten the refinement. Examples: k = "
                f"{', '.join(f'{k:.8g}' for k in zero[:5])}.", RuntimeWarning, stacklevel=3)
        inner = self.degeneracies // np.array([self.sectors.multiplicity(s)
                                               for s in self.level_sectors], dtype=int)
        multiple = spectrum[inner > 1]
        if multiple.size:
            warnings.warn(
                f"{multiple.size} level(s) are degenerate inside their sector (geometric "
                "symmetry, or levels closer than the resolution of the search): they give zero "
                "spacings in the statistics; desymmetrise by the geometric symmetry if needed. "
                "Examples: "
                f"k = {', '.join(f'{k:.8g}' for k in np.real(multiple[:5]))}.",
                RuntimeWarning, stacklevel=3)

    def _selection(self, sector):
        """{representative sector: number of times each of its levels counts}."""
        secs = self.sectors
        if sector is None:
            return {s.index: 1 for s in secs.independent}
        if isinstance(sector, str):
            if sector != "all":
                raise ValueError(f"sector must be None, 'all' or an index, got {sector!r}.")
            return {s.index: secs.multiplicity(s.index) for s in secs.independent}
        if not 0 <= int(sector) < len(secs):
            raise ValueError(f"sector {sector} does not exist (the graph has {len(secs)}).")
        return {secs.class_of(int(sector)): 1}

    def _selected_levels(self, sector=None):
        """Levels of the selection as stored (complex for resonances), sorted by
        real part, each repeated by the number of times it counts."""
        if getattr(self, "spectrum", None) is None:
            raise AttributeError("No spectrum: run compute_spectrum() / find_resonances() "
                                 "first, or pass spectrum=.")
        sel = self._selection(sector)
        secs = self.sectors
        counts = np.array([
            sel.get(s, 0) * (g // secs.multiplicity(s))
            for s, g in zip(self.level_sectors, self.degeneracies)], dtype=int)
        k = np.repeat(np.asarray(self.spectrum), counts)
        return k[np.argsort(np.real(k), kind="stable")]

    def levels(self, sector=None):
        """Real wavenumbers of the selected levels (see the module docstring),
        sorted, repeated by multiplicity within the selection."""
        return np.real(self._selected_levels(sector))

    # =========================================================================
    # Weyl law
    # =========================================================================

    def weyl_length(self, sector=None):
        """Directed length L_sel (µm) of the selection: N(k) = (L_sel/2π) ∫ n_g dk."""
        return float(sum(w * self.sectors[s].length for s, w in self._selection(sector).items()))

    def weyl_staircase(self, k, sector=None, k0=0.0):
        """Smooth counting function N(k) = (L_sel/2π) ∫_{k0}^{k} n_g dκ of the
        selection (see :func:`quantum_graphs.stats.weyl_staircase`)."""
        return weyl_staircase(k, self.weyl_length(sector), self.ng, k0)

    def weyl_count(self, k_min=None, k_max=None, sector=None):
        """Weyl estimate of the number of selected levels in [k_min, k_max]
        (default: the graph's window). ``sector="all"`` counts every level
        with its multiplicity."""
        lo, hi = self._k_range()
        k_min = lo if k_min is None else k_min
        k_max = hi if k_max is None else k_max
        return float(self.weyl_staircase(k_max, sector, k0=k_min))

    def spectral_ensemble(self, sector=None):
        """Random-matrix ensemble expected for the selection if the graph is
        chaotic: "GOE" (self-reverse sector with time-reversal symmetry) or
        "GUE" (one of two partner sectors, or no reciprocity); None if the
        selected sectors differ."""
        ensembles = {self.sectors[s].ensemble for s in self._selection(sector)}
        return ensembles.pop() if len(ensembles) == 1 else None

    # =========================================================================
    # Unfolding, Σ², Δ₃
    # =========================================================================

    def _stats_levels(self, spectrum=None, sector=None, wavelength_range=None):
        """Sorted real levels used by a statistic (computed or given), restricted
        to ``wavelength_range`` (µm) if it is given."""
        if spectrum is None:
            sel = self._selection(sector)
            if len(sel) > 1 or any(w > 1 for w in sel.values()):
                warnings.warn(
                    f"The selected levels superpose {sum(sel.values())} sector spectra "
                    f"(sectors {sorted(sel)}): their statistics are not those of one sector. "
                    "Pass sector=i to analyse one of them.", RuntimeWarning, stacklevel=3)
            k = self.levels(sector)
        else:
            k = np.sort(np.real(np.asarray(spectrum)).astype(float).ravel())
        if wavelength_range is not None:
            lam_min, lam_max = wavelength_range
            k = k[(k >= 2 * np.pi / lam_max) & (k <= 2 * np.pi / lam_min)]
        return k

    def unfolded_spectrum(self, spectrum=None, sector=None, wavelength_range=None):
        """Unfolded levels x_i = N(k_i) − N(k_1) of the selection (unit mean
        spacing). ``spectrum``: explicit levels (e.g. measured peaks) instead of
        the computed ones; ``wavelength_range``: keep the levels in this window (µm)."""
        k = self._stats_levels(spectrum, sector, wavelength_range)
        if k.size == 0:
            return k
        return weyl_staircase(k, self.weyl_length(sector), self.ng, k0=k[0])

    def sigma2(self, L_max=10.0, L_step=0.2, spectrum=None, sector=None,
               wavelength_range=None):
        """Number variance Σ²(L) of the unfolded selection
        (:func:`quantum_graphs.stats.number_variance`). Returns (L, Σ²)."""
        x = self.unfolded_spectrum(spectrum, sector, wavelength_range)
        return number_variance(x, L_step, L_max)

    def delta3(self, L_max=10.0, L_step=0.2, spectrum=None, sector=None,
               wavelength_range=None, windows="level"):
        """Spectral rigidity Δ₃(L) of the unfolded selection
        (:func:`quantum_graphs.stats.spectral_rigidity`). Returns (L, Δ₃)."""
        x = self.unfolded_spectrum(spectrum, sector, wavelength_range)
        return spectral_rigidity(x, L_step, L_max, windows=windows)
