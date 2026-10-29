"""
SpectrumMixin — U(k), secular function, eigenwavenumbers and eigenvectors.

Algorithm of :meth:`compute_spectrum`
-------------------------------------
For each independent sector s (one per pair of reciprocal partners, see
:mod:`quantum_graphs.core.sectors`), the levels are the k at which an
eigenphase of the unitary U_s(k) crosses 0 mod 2π
(:func:`quantum_graphs.core.level_search.find_levels`):

1. the number of levels in any interval is computed exactly from the
   eigenphases at its ends (counting function, no tracking of the
   eigenphases), on a grid of 2 points per mean level spacing;
2. every level is isolated by bisection on the count and located by Brent's
   method on the real secular function Z(k) = Π_j 2 sin(θ_j / 2);
3. the multiplicity of a level is the number of eigenphases crossing there
   (simultaneous crossings), times 2 if the sector has a partner.

The search is complete by construction: no level is missed, and close
levels are not merged. :meth:`compute_spectrum` warns when the number of
levels of a sector departs from the Weyl estimate (∫ n_g dk) of its sector,
i.e. when the group index is inconsistent with the effective index.
"""

import os
import warnings

import numpy as np

from ..core.level_search import find_levels
from ..core.persistence import use_cache, write_cache_meta
from ..core.scattering import NULLITY_TOL


class SpectrumMixin:

    # =========================================================================
    # Spectrum computation
    # =========================================================================

    def compute_spectrum(self, tol=1e-15, show_progress=False, save_spectrum=True,
                         overwrite=False):
        """
        Closed-graph spectrum on the real k-axis: the eigenphase crossings of
        U_s(k), sector by sector (module docstring).

        Only one sector of each pair of reciprocal partners is searched (see
        :mod:`quantum_graphs.core.sectors`): its levels are those of the
        partner, so their multiplicity is known by construction.

        Parameters
        ----------
        tol : float
            Absolute accuracy of the levels (µm⁻¹; the relative accuracy is
            limited to 4 ε).
        show_progress : bool
            Print a summary per sector.
        save_spectrum : bool
            Cache result as .npz (with a ``.meta.json`` parameter record).
        overwrite : bool
            Recompute even if a valid cached file exists.

        Returns
        -------
        spectrum : np.ndarray
            Sorted real eigen-wavenumbers in (k_min, k_max], one per distinct level.
        degeneracies : np.ndarray (int)
            Multiplicity of each level (2 for the levels of partner sectors).

        Also sets ``self.spectrum``, ``self.degeneracies`` and
        ``self.level_sectors`` (representative sector of each level).

        Raises ValueError if U(k) is not unitary (lossy or gain vertices).
        """
        lam_min, lam_max = self.wavelength_range

        filepath_npz = os.path.join(
            self.save_path,
            f"spectrum_lamMin_{lam_min:.3f}_lamMax_{lam_max:.3f}_tol{tol:.0e}.npz")
        params = dict(tol=tol, wavelength_range=list(self.wavelength_range),
                      method="eigenphase crossings")

        if save_spectrum and use_cache(filepath_npz, self.fingerprint, params, overwrite,
                                       label="spectrum"):
            self._log(f"✓ Spectrum loaded from file: {filepath_npz}", verbose=show_progress)
            data = np.load(filepath_npz)
            spectrum = np.array(data["spectrum"], dtype=float)
            degeneracies = np.array(data["degeneracies"], dtype=int)
            level_sectors = np.array(data["level_sectors"], dtype=int)
            self._set_levels(spectrum, degeneracies, level_sectors)
            return spectrum, degeneracies

        k_min, k_max = self._k_range()
        roots, multiplicities, sectors = [], [], []
        for sec in self.sectors.independent:
            found = find_levels(self.model.sector(sec.directed), k_min, k_max, tol=tol)
            self._log(f"Sector {sec.index} ({sec.directed.size} directed bonds, "
                      f"×{self.sectors.multiplicity(sec.index)}): {found.count} levels "
                      f"({found.levels.size} distinct), {found.n_evaluations} eigenvalue "
                      f"decompositions, defect {found.defect:.1e}", verbose=show_progress)
            self._check_weyl_count(sec, found.count)
            roots.append(found.levels)
            multiplicities.append(found.multiplicities * self.sectors.multiplicity(sec.index))
            sectors.append(np.full(found.levels.size, sec.index))

        order = np.argsort(np.concatenate(roots), kind="stable")
        spectrum = np.concatenate(roots)[order]
        degeneracies = np.concatenate(multiplicities)[order].astype(int)
        level_sectors = np.concatenate(sectors)[order].astype(int)

        if save_spectrum:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez(filepath_npz, spectrum=spectrum, degeneracies=degeneracies,
                     level_sectors=level_sectors)
            write_cache_meta(filepath_npz, self.fingerprint, params)
            self._log(f"✓ Spectrum saved to {filepath_npz}", verbose=show_progress)

        self._set_levels(spectrum, degeneracies, level_sectors)
        return spectrum, degeneracies

    def _check_weyl_count(self, sector, count, rel_tol=0.05):
        """Warn when the number of levels of a sector departs from its Weyl estimate."""
        if count < 20:
            return
        expected = self.weyl_count(sector=sector.index)
        if expected > 0 and abs(count - expected) > max(rel_tol * expected, 3.0):
            warnings.warn(
                f"Sector {sector.index} has {count} levels where the Weyl law (∫ n_g dk) "
                f"predicts {expected:.1f}: the group index ng is inconsistent with the "
                "effective index neff.", RuntimeWarning, stacklevel=3)

    # =========================================================================
    # Wavefunction
    # =========================================================================

    def compute_wavefunction(self, k, tol=NULLITY_TOL):
        """
        Basis of ker(I − U(k)) at wavenumber k (singular values ≤ ``tol``).

        Component b of each vector is the amplitude of the wave on directed
        bond b = (u → v) when it arrives at v, so that a = U(k) a. Vectors are
        normalised (Σ|a_b|² = 1), with an arbitrary global phase; for a
        degenerate level they form an arbitrary orthonormal basis of the
        eigenspace. For level i of the computed spectrum,
        ``null_vectors(spectrum[i], degeneracies[i])`` uses its multiplicity.

        Returns
        -------
        null_space : ndarray, shape (n_modes, 2 n_bonds)
            Each row is one wavefunction coefficient vector.
        """
        rows = self.null_vectors(k, tol=tol)
        if rows.size == 0:
            self._log(f"⚠ Warning: No null space found at k = {k:.6f} (tol={tol}).")
        else:
            self._log(f"✓ Found {rows.shape[0]} wavefunction(s) at k = {k:.6f}.")
        return rows
