"""
Levels of a closed graph: the real k at which an eigenphase of the unitary
U(k) crosses 0 (mod 2π). The search is complete by construction: the number
of crossings in any interval is computed exactly, without following the
eigenphases.

Eigenphases and counting function
---------------------------------
For a closed graph with unitary vertex matrices, U(k) = D(k) Σ(k) is unitary
at real k; its eigenvalues are e^{iθ_j(k)}, and k is a level when some θ_j(k)
is a multiple of 2π. With Σ independent of k, dθ_j/dk = n_g ⟨v_j|L|v_j⟩ > 0
(v_j the eigenvector, L the diagonal of bond lengths): the eigenphases
increase with k, so they cross every multiple of 2π once, upwards.

Let θ_j(k) be continuous branches of the eigenphases and Θ(k) = Σ_j θ_j(k)
the continuous phase of det U(k),

    Θ(k) = k n_eff(k) Σ_d L_d + arg det Σ(k)          (lifted continuously).

The counting function N(k) = Σ_j ⌊θ_j(k) / 2π⌋, whose jumps are the
levels, only needs the principal eigenphases φ_j ∈ (−π, π]:

    N(k) = G(k) − #{j : φ_j(k) < 0},    G(k) = (Θ(k) − Σ_j φ_j(k)) / 2π ∈ ℤ.

The number of levels in (a, b] is N(b) − N(a), counted with multiplicity,
for any a < b: a degenerate level is a simultaneous crossing of several
eigenphases. Rounding G to an integer only needs Θ to better than π; the
distance of G to an integer before rounding (the *defect*) checks it.

The real secular function

    Z(k) = Π_j 2 sin(θ_j(k) / 2) = (−1)^{G(k)} Π_j 2 sin(φ_j(k) / 2),

|Z| = |det(I − U)|, changes sign at every simple level (Kottos-Smilansky).

Algorithm of :func:`find_levels`
--------------------------------
1. N and Z on a grid of ``samples_per_spacing`` points per mean level spacing
   Δ (batched eigenvalue decompositions); arg det Σ is unwrapped along the
   grid and interpolated between its points.
2. Every interval with one level: Brent's method on Z (``xtol=tol``).
   Every interval with several levels: bisection on N until the levels are
   isolated, or until the interval is narrower than the resolution
   max(tol, 4 ε |k|) (a degenerate level, multiplicity N(b) − N(a)).
3. Levels closer than ``SIMULTANEOUS`` Δ are one degenerate level (an exact
   degeneracy can be split by round-off, or straddle a grid point).

A Σ(k) that varies fast enough to make an eigenphase decrease gives a
negative count somewhere; :func:`find_levels` warns (the levels found are
then the net crossings).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .zero_search import mean_spacing

#: levels closer than SIMULTANEOUS × mean spacing are one degenerate level
SIMULTANEOUS = 1e-9
#: default grid density (points per mean level spacing)
SAMPLES_PER_SPACING = 2
#: largest allowed deviation of |eigenvalue of U| from 1
UNITARITY_TOL = 1e-6


@dataclass(frozen=True)
class LevelSearch:
    """Result of :func:`find_levels`.

    Attributes
    ----------
    levels : ndarray
        Levels in (k_min, k_max], sorted.
    multiplicities : ndarray, int
        Number of eigenphases crossing at each level.
    count : int
        N(k_max) − N(k_min): number of levels with multiplicity (equal to
        ``multiplicities.sum()``).
    defect : float
        Largest distance of G(k) to an integer before rounding (≪ 1).
    n_evaluations : int
        Number of eigenvalue decompositions of U.
    """
    levels: np.ndarray
    multiplicities: np.ndarray
    count: int
    defect: float
    n_evaluations: int


class _Phases:
    """N(k) and Z(k) of a model (see the module docstring)."""

    def __init__(self, model):
        self.model = model
        self.length = float(np.sum(model.lengths))
        self.grid = None
        self.arg_sigma = None
        self.defect = 0.0
        self.n_evaluations = 0

    def _free_phase(self, ks):
        """k n_eff(k) Σ_d L_d."""
        return ks * self.model.index(ks).real * self.length

    def _eigenphases(self, ks):
        lam = self.model.eigenvalues_many(ks)
        self.n_evaluations += ks.size
        deviation = np.max(np.abs(np.abs(lam) - 1.0)) if lam.size else 0.0
        if deviation > UNITARITY_TOL:
            raise ValueError(
                f"U(k) is not unitary (|eigenvalue| − 1 up to {deviation:.1e}): a closed graph "
                "with lossy or gain vertices has no real levels; search the complex zeros of "
                "det(I − U) instead (quantum_graphs.core.zero_search).")
        return np.angle(lam)

    def set_grid(self, ks):
        """Evaluate N and Z on the grid ``ks`` and unwrap arg det Σ along it."""
        phi = self._eigenphases(ks)
        raw = np.angle(np.exp(1j * (np.sum(phi, axis=1) - self._free_phase(ks))))
        self.grid = ks
        self.arg_sigma = np.unwrap(raw)
        return self._count(ks, phi)

    def __call__(self, ks):
        """(N, Z) at the wavenumbers ``ks`` (array)."""
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        return self._count(ks, self._eigenphases(ks))

    def _count(self, ks, phi):
        theta = self._free_phase(ks) + np.interp(ks, self.grid, self.arg_sigma)
        g = (theta - np.sum(phi, axis=1)) / (2 * np.pi)
        G = np.rint(g)
        if g.size:
            self.defect = max(self.defect, float(np.max(np.abs(g - G))))
        N = G.astype(np.int64) - np.sum(phi < 0, axis=1)
        Z = np.where(G % 2 == 0, 1.0, -1.0) * np.prod(2 * np.sin(phi / 2), axis=1)
        return N, Z

    def Z(self, k):
        return float(self(k)[1][0])


def _merge(levels, multiplicities, radius):
    """Merge consecutive levels closer than ``radius`` (one degenerate level,
    at their mean position, multiplicities added)."""
    if levels.size < 2:
        return levels, multiplicities
    order = np.argsort(levels, kind="stable")
    levels, multiplicities = levels[order], multiplicities[order]
    group = np.concatenate([[0], np.cumsum(np.diff(levels) > radius)])
    m = np.bincount(group, weights=multiplicities).astype(int)
    k = np.bincount(group, weights=levels * multiplicities) / m
    return k, m


def find_levels(model, k_min, k_max, tol=1e-15, samples_per_spacing=SAMPLES_PER_SPACING):
    """Levels of ``model`` (unitary U(k)) in (k_min, k_max], with their
    multiplicities (see the module docstring).

    Raises ValueError if U(k) is not unitary. Returns a :class:`LevelSearch`.
    """
    if not k_min < k_max:
        raise ValueError("Empty window.")
    spacing = mean_spacing(model, k_min, k_max)
    n_grid = int(np.ceil((k_max - k_min) / spacing * samples_per_spacing)) + 1
    ks = np.linspace(k_min, k_max, max(n_grid, 2))
    phases = _Phases(model)
    N, Z = phases.set_grid(ks)

    counts = np.diff(N)
    if np.any(counts < 0):
        k = ks[np.flatnonzero(counts < 0)[0]]
        warnings.warn(
            f"An eigenphase of U(k) decreases near k = {k:.10g}: the vertex matrices vary "
            "too fast with k, or n_g ≤ 0; the levels found are the net crossings.",
            RuntimeWarning, stacklevel=2)

    levels, mults = [], []
    stack = [(ks[i], ks[i + 1], N[i], N[i + 1], Z[i], Z[i + 1])
             for i in np.flatnonzero(counts > 0)]
    while stack:
        a, b, Na, Nb, Za, Zb = stack.pop()
        c = Nb - Na
        if c <= 0:
            continue
        if c == 1 and Za * Zb < 0:
            levels.append(brentq(phases.Z, a, b, xtol=tol, rtol=4 * np.finfo(float).eps))
            mults.append(1)
            continue
        if b - a <= max(tol, 4 * np.finfo(float).eps * abs(b)):
            levels.append(0.5 * (a + b))
            mults.append(int(c))
            continue
        m = 0.5 * (a + b)
        (Nm,), (Zm,) = phases(m)
        stack.append((a, m, Na, Nm, Za, Zm))
        stack.append((m, b, Nm, Nb, Zm, Zb))

    k, mult = _merge(np.array(levels, dtype=float), np.array(mults, dtype=int),
                     SIMULTANEOUS * spacing)
    return LevelSearch(levels=k, multiplicities=mult, count=int(N[-1] - N[0]),
                       defect=phases.defect, n_evaluations=phases.n_evaluations)
