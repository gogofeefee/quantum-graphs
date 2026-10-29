#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResonancesMixin — find_resonances, count_resonances, σ_min heatmap.

Algorithm of :meth:`find_resonances`
------------------------------------
The search rectangle is R = [k_min, k_max] × [im_min, im_max] (k_min, k_max
from the wavelength window). For each independent sector s (one per pair of
reciprocal partners, see :mod:`quantum_graphs.core.sectors`):

1. the zeros of det(I − U_s) in R are counted by the argument principle
   (:mod:`quantum_graphs.core.argument_principle`), in sub-windows of about
   ``COUNT_WINDOW`` resonances of the Weyl law;
2. they are found by Newton's method on log det(I − U_s) from a lattice of
   seeds, then by deflation (the zeros found divided out) in the sub-windows
   where fewer are found than counted (:mod:`quantum_graphs.core.zero_search`);
3. multiplicity of a resonance = (sectors sharing the spectrum of s) ×
   (singular values of I − U_s at or below ``nullity_tol``).

For im_max ≥ 0 the upper edge of R is raised to im_max + (im_max − im_min) / 10,
so that real roots (bound states in the continuum) are not on the contour; a
passive graph has no resonance with Im k > 0.

Checks (warnings, repeated when a cached spectrum is loaded)
------------------------------------------------------------
* completeness: a sub-window where the resonances found (with multiplicity)
  differ from the count after the deflation rounds;
* resonances found just below R (the rectangle cuts through the resonances);
* Weyl law: an exponential polynomial whose highest term
  det(Σ_s) Π e^{i k n_eff L_d} does not vanish has (L_s / 2π) ∫ n_g dk zeros
  per window, all Im k together (up to a bounded number). Fewer with
  Im k ≥ im_min means that broader resonances lie below im_min. If det(Σ_s)
  = 0 (e.g. a vertex with as many leads as bonds) the count is non-Weyl
  (smaller) and no check is made.
"""

from ..core.config import resolve, show_figure
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..core.argument_principle import ZeroCount, count_zeros
from ..core.persistence import use_cache, write_cache_meta
from ..core.scattering import NULLITY_TOL
from ..core.zero_search import find_zeros, mean_spacing
from ..plotting.style import save_figure

#: target number of resonances per sub-window of the count (and of the deflation)
COUNT_WINDOW = 20
#: |det Σ_s| below this value: non-Weyl sector, no Weyl check
NON_WEYL_TOL = 1e-8


class ResonancesMixin:
    """Complex resonances of an open graph: search, certified counts, heatmap."""

    # ── Cache paths ───────────────────────────────────────────────────────────

    def _sigma_grid_path(self, N_re: int, N_im: int,
                         im_min: float, im_max: float) -> str:
        lam_min, lam_max = self.wavelength_range
        fname = (f"sigma_grid"
                 f"_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_Nre_{N_re}_Nim_{N_im}"
                 f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}.npy")
        return os.path.join(self.save_path, fname)

    def _spectrum_path(self, im_min, im_max, seeds_per_spacing, tol, maxiter,
                       max_rounds, eps_accept, nullity_tol) -> str:
        lam_min, lam_max = self.wavelength_range
        fname = (f"resonances"
                 f"_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}"
                 f"_seeds_{seeds_per_spacing:g}_tol_{tol:.1e}_maxiter_{int(maxiter)}"
                 f"_rounds_{int(max_rounds)}_eps_{eps_accept:.1e}_nulltol_{nullity_tol:.1e}.npz")
        return os.path.join(self.save_path, fname)

    # ── Grid open / fill ──────────────────────────────────────────────────────

    def _open_sigma_grid(self, N_re: int, N_im: int,
                         im_min: float, im_max: float,
                         grid_dtype: str = "float32",
                         overwrite: bool = False,
                         verbose: bool = None):
        verbose = resolve("verbose", verbose)
        lam_min, lam_max = self.wavelength_range
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
        re_vals = np.linspace(k_min, k_max, N_re, dtype=np.float64)
        im_vals = np.linspace(im_max, im_min, N_im, dtype=np.float64)
        grid_path = self._sigma_grid_path(N_re, N_im, im_min, im_max)
        os.makedirs(self.save_path, exist_ok=True)
        grid_params = dict(N_re=N_re, N_im=N_im, im_min=im_min, im_max=im_max,
                           wavelength_range=list(self.wavelength_range),
                           **self._complex_k_params())
        if os.path.exists(grid_path) and not use_cache(grid_path, self.fingerprint, grid_params,
                                                       overwrite, label="σ_min grid"):
            overwrite = True
        if overwrite and os.path.exists(grid_path):
            os.remove(grid_path)
            if verbose:
                print(f"[grid] Removed existing grid: {grid_path}")
        if not os.path.exists(grid_path):
            if verbose:
                print(f"[grid] Creating  ({N_im}×{N_re}, {grid_dtype}): {grid_path}")
            mem = np.lib.format.open_memmap(grid_path, mode="w+",
                                            dtype=grid_dtype, shape=(N_im, N_re))
            mem[:] = np.nan
            mem.flush()
            write_cache_meta(grid_path, self.fingerprint, grid_params)
        else:
            if verbose:
                print(f"[grid] Reusing existing grid: {grid_path}")
            mem = np.lib.format.open_memmap(grid_path, mode="r+")
            if mem.shape != (N_im, N_re):
                raise ValueError(f"Existing grid shape {mem.shape} != requested ({N_im}, {N_re}).")
        return mem, re_vals, im_vals, grid_path

    def _fill_sigma_grid(self, mem, re_vals, im_vals, grid_path,
                         verbose: bool = None, flush_every: int = 1):
        verbose = resolve("verbose", verbose)
        N_im, N_re = mem.shape
        n_done = int(np.sum(~np.isnan(mem).any(axis=1)))
        if verbose:
            print(f"[grid] Filling σ_min grid ({n_done}/{N_im} rows already done) ...")
        flush_counter = 0
        for i in range(N_im):
            row = mem[i]
            if not np.isnan(row).any():
                continue
            im       = im_vals[i]
            nan_cols = np.where(np.isnan(row))[0]
            row[nan_cols] = self.model.sigma_min_many(re_vals[nan_cols] + 1j * im)
            flush_counter += 1
            if flush_counter >= flush_every:
                mem.flush()
                flush_counter = 0
            if verbose:
                done = int(np.sum(~np.isnan(mem).any(axis=1)))
                print(f"\r[grid]  {done}/{N_im} rows ({100*done/N_im:.1f}%)", end="", flush=True)
        mem.flush()
        if verbose:
            print("\r[grid] Done – grid complete.              ")

    # ── Public methods ────────────────────────────────────────────────────────

    def find_resonances(self,
                        im_min: float = -1e-4,
                        im_max: float = 0.0,
                        seeds_per_spacing: float = 4,
                        tol: float = 1e-14,
                        maxiter: int = 100,
                        max_rounds: int = 6,
                        eps_accept: float = 1e-4,
                        nullity_tol: float = NULLITY_TOL,
                        overwrite: bool = False,
                        plot: bool = False,
                        verbose: bool = None):
        """
        Find the complex resonances of the open quantum graph in the rectangle
        [k_min, k_max] × [im_min, im_max] by Newton's method, sector by
        sector, until the argument-principle count is reached (see the module
        docstring for the algorithm and the checks).

        Parameters
        ----------
        im_min, im_max : float
            Imaginary-k bounds of the rectangle (µm⁻¹).
        seeds_per_spacing : float
            Density of the initial lattice of Newton seeds, per mean spacing
            of the resonances (in Re k and in Im k).
        tol : float
            Newton stops when the step falls below ``tol`` |k|.
        maxiter : int
            Maximum Newton iterations per seed.
        max_rounds : int
            Maximum deflation rounds (seed density doubled each round).
        eps_accept : float
            Maximum σ_min(I − U_s(k*)) for a converged point to be a resonance.
        nullity_tol : float
            Singular values of I − U_s(k*) at or below it count in the
            multiplicity.
        overwrite : bool
            Recompute even if a valid cached spectrum exists.
        plot : bool
            Plot the resonances over the σ_min heatmap
            (:meth:`plot_resonances_sigma`, saved).

        Returns
        -------
        spectrum      : np.ndarray, complex128, shape (N,), sorted by Re k
        degeneracies  : np.ndarray, int, shape (N,)

        Also sets ``self.spectrum``, ``self.degeneracies``,
        ``self.level_sectors`` and ``self.resonance_counts`` ({independent
        sector: :class:`~quantum_graphs.core.argument_principle.ZeroCount`}).
        """
        verbose = resolve("verbose", verbose)
        os.makedirs(self.save_path, exist_ok=True)

        k_min, k_max = self._k_range()
        upper = self._upper_edge(im_min, im_max)
        spec_path = self._spectrum_path(im_min, im_max, seeds_per_spacing, tol, maxiter,
                                        max_rounds, eps_accept, nullity_tol)
        spec_params = dict(im_min=im_min, im_max=im_max,
                           wavelength_range=list(self.wavelength_range),
                           seeds_per_spacing=seeds_per_spacing, tol=tol, maxiter=maxiter,
                           max_rounds=max_rounds, eps_accept=eps_accept,
                           nullity_tol=nullity_tol, method="newton",
                           **self._complex_k_params())

        if use_cache(spec_path, self.fingerprint, spec_params, overwrite,
                     label="resonance spectrum"):
            if verbose:
                print(f"[spectrum] Loading the resonances from: {spec_path}")
            npz = np.load(spec_path)
            self._set_levels(npz["spectrum"].astype(np.complex128),
                             npz["degeneracies"].astype(int), npz["level_sectors"].astype(int))
            self.resonance_counts = self._load_counts(npz)
            self._check_resonances(im_min, npz["below"], verbose=False)
        else:
            cuts = self._count_cuts()
            zeros, degeneracies, sectors, below = [], [], [], []
            self.resonance_counts = {}
            for sec in self.sectors.independent:
                model = self.model.sector(sec.directed)
                if verbose:
                    print(f"[sector {sec.index}] {sec.directed.size} directed bonds, "
                          f"×{self.sectors.multiplicity(sec.index)}: counting ...")
                try:
                    count = count_zeros(model, k_min, k_max, im_min, upper, cuts)
                    self.resonance_counts[sec.index] = count
                except RuntimeError as err:
                    count = None
                    warnings.warn(f"Resonance count of sector {sec.index} failed ({err}); the "
                                  "completeness of the search is not checked.",
                                  RuntimeWarning, stacklevel=2)
                search = find_zeros(model, k_min, k_max, im_min, upper, count,
                                    seeds_per_spacing=seeds_per_spacing, tol=tol,
                                    maxiter=maxiter, max_rounds=max_rounds,
                                    eps_accept=eps_accept, nullity_tol=nullity_tol)
                if verbose:
                    print(f"[sector {sec.index}] {search.zeros.size} resonances found"
                          + (f", {count.total} counted" if count is not None else "")
                          + f" ({search.rounds} deflation round(s)).")
                zeros.append(search.zeros)
                degeneracies.append(search.multiplicities * self.sectors.multiplicity(sec.index))
                sectors.append(np.full(search.zeros.size, sec.index))
                below.append(search.below)

            order = np.argsort(np.real(np.concatenate(zeros)), kind="stable")
            spectrum = np.concatenate(zeros)[order]
            self._set_levels(spectrum, np.concatenate(degeneracies)[order].astype(int),
                             np.concatenate(sectors)[order].astype(int))
            below = np.sort_complex(np.concatenate(below))
            self._check_resonances(im_min, below, verbose=verbose)

            np.savez(spec_path, spectrum=self.spectrum, degeneracies=self.degeneracies,
                     level_sectors=self.level_sectors, below=below, **self._counts_arrays())
            write_cache_meta(spec_path, self.fingerprint, spec_params)
            if verbose:
                print(f"[spectrum] Resonances saved to: {spec_path}")

        if verbose:
            print(f"[done] {self.spectrum.size} resonances, "
                  f"total multiplicity = {self.degeneracies.sum()}.")
        if plot:
            self.plot_resonances_sigma(im_min=im_min, im_max=im_max, save_fig=True,
                                       verbose=verbose)
        return self.spectrum, self.degeneracies

    def count_resonances(self, im_min: float = -1e-4, im_max: float = 0.0,
                         sector=None, cuts=()):
        """Number of resonances, with multiplicity, in the rectangle
        [k_min, k_max] × [im_min, im_max] (argument principle, see
        :mod:`quantum_graphs.core.argument_principle`), and in its
        sub-rectangles between the vertical lines Re k = ``cuts``.

        ``sector=None`` counts the zeros of det(I − U) (every sector, partners
        included), ``sector=i`` those of det(I − U_i). For im_max ≥ 0 the
        upper edge is raised to im_max + (im_max − im_min) / 10, so that real
        roots are not on the contour. Fast (~0.4 s for the flower graph over
        1.48–1.64 µm), it also tells how deep im_min must go: the Weyl count
        :meth:`weyl_count` gives all the resonances, all Im k together.

        Returns a :class:`~quantum_graphs.core.argument_principle.ZeroCount`.
        """
        k_min, k_max = self._k_range()
        model = (self.model if sector is None
                 else self.model.sector(self.sectors[int(sector)].directed))
        return count_zeros(model, k_min, k_max, im_min, self._upper_edge(im_min, im_max), cuts)

    # ── Checks ────────────────────────────────────────────────────────────────

    @staticmethod
    def _upper_edge(im_min, im_max):
        """Upper edge of the counting contour (raised above the real axis)."""
        return im_max if im_max < 0 else im_max + 0.1 * (im_max - im_min)

    def _found_per_sector(self, edges):
        """{independent sector: roots found (with multiplicity in the sector)
        in each sub-window [edges[j], edges[j+1]]}."""
        found = {}
        re = np.real(self.spectrum)
        for sec in self.sectors.independent:
            mine = self.level_sectors == sec.index
            weight = self.degeneracies[mine] // self.sectors.multiplicity(sec.index)
            j = np.clip(np.searchsorted(edges, re[mine], side="right") - 1, 0, edges.size - 2)
            found[sec.index] = np.bincount(j, weights=weight, minlength=edges.size - 1).astype(int)
        return found

    def _count_cuts(self):
        """Re k of the cuts between the sub-windows of the count: every
        COUNT_WINDOW resonances of the Weyl law (independent sectors)."""
        k_min, k_max = self._k_range()
        k = np.linspace(k_min, k_max, 4097)
        N = self.weyl_staircase(k, k0=k_min)
        return np.interp(np.arange(COUNT_WINDOW, N[-1] - COUNT_WINDOW / 2, COUNT_WINDOW), N, k)

    def _complex_k_params(self):
        """Cache parameter of the results computed off the real axis: the
        continuation of a dispersive index (see :mod:`quantum_graphs.core.scattering`)."""
        return dict(index_continuation="first order") if callable(self.neff) else {}

    def _counts_arrays(self):
        """The resonance counts as arrays for the spectrum cache."""
        c = self.resonance_counts
        if not c:
            return {}
        first = next(iter(c.values()))
        return dict(count_sectors=np.array(list(c), dtype=int),
                    count_edges=first.edges,
                    count_im=np.array([first.im_min, first.im_max]),
                    count_winding=np.array([z.winding for z in c.values()]),
                    count_samples=np.array([z.n_samples for z in c.values()]))

    @staticmethod
    def _load_counts(npz):
        if "count_sectors" not in npz.files:
            return {}
        im_lo, im_hi = (float(x) for x in npz["count_im"])
        return {int(s): ZeroCount(edges=npz["count_edges"], im_min=im_lo, im_max=im_hi,
                                  winding=w, n_samples=int(n))
                for s, w, n in zip(npz["count_sectors"], npz["count_winding"],
                                   npz["count_samples"])}

    def _check_resonances(self, im_min, below, verbose=False):
        """Warn about resonances found below im_min, counts that differ from the
        resonances found, and a Weyl law they fall short of (module docstring)."""
        if below.size:
            warnings.warn(
                f"{below.size} resonance(s) lie just below the search rectangle "
                f"(Im k down to {np.imag(below).min():.4e} < im_min = {im_min:.4e}) and are "
                "left out: the rectangle cuts through the resonances; lower im_min to include "
                "them.", RuntimeWarning, stacklevel=3)

        for s, zc in self.resonance_counts.items():
            found = self._found_per_sector(zc.edges)[s]
            bad = np.flatnonzero(found != zc.counts)
            if verbose:
                print(f"[count] sector {s}: {zc.total} resonances in the rectangle "
                      f"(argument principle), {found.sum()} found.")
            if zc.defect > 0.1:
                warnings.warn(f"Resonance count of sector {s}: winding numbers {zc.defect:.2f} "
                              "away from integers (contour not resolved).",
                              RuntimeWarning, stacklevel=3)
            if bad.size:
                lam = 2 * np.pi / zc.edges
                where = "; ".join(
                    f"λ ∈ [{lam[j + 1]:.6f}, {lam[j]:.6f}] µm: {zc.counts[j]} counted, "
                    f"{found[j]} found" for j in bad[:5])
                warnings.warn(
                    f"find_resonances found {found.sum()} resonances (with multiplicity) in "
                    f"sector {s} where the argument principle counts {zc.total} in the "
                    f"rectangle; {bad.size} sub-window(s) differ ({where}"
                    f"{'; ...' if bad.size > 5 else ''}). Resonances missed (increase max_rounds or "
                    "seeds_per_spacing) or a wrong multiplicity (nullity_tol).",
                    RuntimeWarning, stacklevel=3)

        # Weyl law: compared with the count in the rectangle (the roots found
        # without certificate), so that it only reports the depth of the search
        k_mid = 0.5 * sum(self._k_range())
        for sec in self.sectors.independent:
            if sec.index in self.resonance_counts:
                n_in, how = self.resonance_counts[sec.index].total, "counts"
            else:
                n_in = int(np.sum(self.degeneracies[self.level_sectors == sec.index])
                           // self.sectors.multiplicity(sec.index))
                how = "found"
            if abs(np.linalg.det(self.model.sector(sec.directed).block("aa", k_mid))) \
                    < NON_WEYL_TOL:
                if verbose:
                    print(f"[weyl] sector {sec.index}: det Σ_s = 0, non-Weyl resonance "
                          "count, no Weyl check.")
                continue
            expected = self.weyl_count(sector=sec.index)
            if verbose:
                print(f"[weyl] sector {sec.index}: {n_in} with Im k ≥ im_min, Weyl law "
                      f"{expected:.1f} (all Im k).")
            if expected >= 20 and abs(n_in - expected) > max(0.05 * expected, 3.0):
                warnings.warn(
                    f"The search rectangle {how} {n_in} resonances in sector {sec.index} "
                    f"(Im k ≥ im_min = {im_min:.1e}) where the Weyl law predicts "
                    f"{expected:.1f} with any Im k: lower im_min to include the broader "
                    "resonances (count_resonances tells how many lie above a given depth), "
                    "or check the group index.", RuntimeWarning, stacklevel=3)

    # ── Resonance heatmap plot ────────────────────────────────────────────────

    def plot_resonances_sigma(self,
                              lam_min: float = None,
                              lam_max: float = None,
                              N_re: int = None,
                              N_im: int = 25,
                              im_min: float = -1e-4,
                              im_max: float = 0.0,
                              plot_heatmap: bool = True,
                              overwrite_grid: bool = False,
                              grid_dtype: str = "float32",
                              max_cols: int = None,
                              save_fig: bool = False,
                              file_format: str = "pdf",
                              verbose: bool = None,
                              show: bool = None):
        """
        Plot complex resonances in the k-plane, optionally overlaying the
        σ_min(I − U(k)) heatmap.

        Requires ``self.spectrum`` to exist (run ``find_resonances`` first).

        Parameters
        ----------
        lam_min, lam_max : float or None
            Wavelength bounds (µm). Defaults to ``self.wavelength_range``.
        N_re, N_im : int
            Size of the σ_min grid of the heatmap (cached). ``N_re=None``: 20
            points per mean spacing of the resonances.
        im_min, im_max : float
            Imaginary-k axis bounds.
        plot_heatmap : bool
            If True, load (or compute) the σ_min grid and display as heatmap.
        overwrite_grid : bool
            If True, recompute the grid even if a file already exists.
        grid_dtype : str
            Storage dtype for the σ_min grid (``"float32"`` or ``"float64"``).
        max_cols : int or None
            Cap on the number of image columns.  Above it, the grid is reduced
            by taking the **minimum** over blocks of columns.  ``None`` (default)
            plots the grid as it stands.  Worth setting whenever the grid is far
            wider than the figure: matplotlib's own resampling *averages*, which
            erases σ_min valleys narrower than a pixel — that is, the very
            resonances the plot is about — while the block minimum keeps them.
        save_fig : bool
            Save the figure to ``self.save_path``.
        file_format : str
            Figure format (``"pdf"``, ``"png"``, …).
        verbose : bool
            Print progress messages.
        show : bool
            Show the figure (default: the ``show`` option); otherwise it is closed.

        Returns
        -------
        fig, ax
        """
        verbose = resolve("verbose", verbose)
        from matplotlib.ticker import ScalarFormatter

        if not hasattr(self, "spectrum"):
            raise RuntimeError(
                "'self.spectrum' is required but missing. "
                "Run 'find_resonances' first."
            )

        spectrum = np.asarray(self.spectrum, dtype=np.complex128)

        lam_min0, lam_max0 = self.wavelength_range
        if lam_min is None:
            lam_min = lam_min0
        if lam_max is None:
            lam_max = lam_max0

        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_heatmap:
            if N_re is None:
                k0, k1 = self._k_range()
                N_re = max(int(20 * (k1 - k0) / mean_spacing(self.model, k0, k1)), 1000)
            mem, re_vals, im_vals, grid_path = self._open_sigma_grid(
                N_re=N_re, N_im=N_im, im_min=im_min, im_max=im_max,
                grid_dtype=grid_dtype, overwrite=overwrite_grid, verbose=verbose,
            )
            # Fill any missing rows
            self._fill_sigma_grid(mem, re_vals, im_vals, grid_path, verbose=verbose)

            # Restrict to the displayed range first, so that a zoom does not
            # carry (and resample) the whole grid.
            j0 = max(int(np.searchsorted(re_vals, k_min, side="left")) - 1, 0)
            j1 = min(int(np.searchsorted(re_vals, k_max, side="right")) + 1,
                     re_vals.size)
            img    = np.asarray(mem[:, j0:j1], dtype=np.float64)
            img    = np.where(np.isfinite(img), img, np.inf)
            re_img = np.asarray(re_vals[j0:j1], dtype=np.float64)

            # Then reduce the column count by block MINIMUM (see max_cols).
            if max_cols is not None and img.shape[1] > int(max_cols):
                block  = int(np.ceil(img.shape[1] / int(max_cols)))
                n_keep = (img.shape[1] // block) * block
                img    = img[:, :n_keep].reshape(img.shape[0], -1, block).min(axis=2)
                re_img = re_img[:n_keep].reshape(-1, block).mean(axis=1)
                if verbose:
                    print(f"[plot] {j1 - j0} grid columns → {img.shape[1]} image "
                          f"columns by block minimum (block = {block}).")

            finite = img[np.isfinite(img)]
            vmin = float(np.nanmin(finite)) if finite.size else 1e-16
            if vmin <= 0.0:
                vmin = 1e-16
            vmax = float(np.nanmax(finite)) if finite.size else 1.0

            imshow_obj = ax.imshow(
                img,
                extent=[re_img[0], re_img[-1], im_vals[-1], im_vals[0]],
                aspect="auto",
                cmap="inferno_r",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                origin="upper",
            )
            cbar = plt.colorbar(imshow_obj, ax=ax)
            cbar.set_label(r"$\sigma_{\min}(I - U(k))$", fontsize=22)
            cbar.ax.tick_params(labelsize=18)

        if spectrum.size > 0:
            cross_color = "red" if plot_heatmap else "black"
            ax.scatter(np.real(spectrum), np.imag(spectrum),
                       marker="x", color=cross_color, s=40, linewidths=1.5,
                       zorder=5)

        ax.set_xlim(k_min, k_max)
        ax.set_ylim(im_min, im_max)
        ax.set_xlabel(r"Re$(k)$  [µm$^{-1}$]", fontsize=22)
        ax.set_ylabel(r"Im$(k)$  [µm$^{-1}$]", fontsize=22)

        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(yfmt)
        ax.tick_params(axis="both", which="major", labelsize=18,
                       direction="in", top=True, right=True)

        fig.tight_layout()

        if save_fig:
            tag = "heatmap" if plot_heatmap else "crossesOnly"
            save_figure(fig, self.save_path,
                        f"ResonancesSigma_{tag}_lam{lam_min:.3f}-{lam_max:.3f}.{file_format}",
                        "Figure", verbose)

        show_figure(fig, show)
        return fig, ax
