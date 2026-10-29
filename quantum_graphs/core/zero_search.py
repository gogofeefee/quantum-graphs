"""
Zeros of det(I − U(k)) in a rectangle of the complex k-plane: Newton's
method from a lattice of seeds, completed by deflation until the
argument-principle count is reached.

Algorithm of :func:`find_zeros`
-------------------------------
R = [k_min, k_max] × [im_min, im_max]; Δ = mean spacing of the zeros of the
model, (k_max − k_min) / N with N = (Σ_d L_d / 2π) [k n_eff(k)] between k_min
and k_max (Weyl law).

1. Newton's method on log det(I − U),

       k ← k − 1 / (d/dk log det(I − U)),   d/dk log det(I − U) = −tr((I − U)⁻¹ U'),

   batched over every seed (:meth:`ScatteringModel.log_det_many`): a lattice of
   step Δ / ``seeds_per_spacing`` in Re k and in Im k covering R (at least its
   middle row). Steps are capped at Δ; iterates that leave R by more than
   max(Δ, height of R) are dropped; the iteration stops when the step falls
   below ``tol`` |k|.
2. Converged points with σ_min(I − U) ≤ ``eps_accept`` are zeros; points
   closer than ``MERGE`` Δ are the same zero. The multiplicity of a zero is
   the number of singular values of I − U at or below ``nullity_tol``; zeros
   closer than ``PAIR`` Δ of which one has multiplicity ≥ 2 are one
   (nearly) degenerate zero, kept once with its multiplicity.
3. Deflation: in every sub-window [e_j, e_{j+1}] of ``count`` (a
   :class:`~.argument_principle.ZeroCount`) where fewer zeros are found than
   counted, Newton's method is run again on

       det(I − U(k)) / Π_i (k − z_i)^{m_i},

   whose logarithmic derivative is that of det(I − U) minus Σ_i m_i / (k − z_i):
   the zeros z_i already found are divided out and repel the iterates. The
   seeds are a lattice twice as dense as the previous one covering the
   sub-window; at most ``max_rounds`` rounds, the density doubling each
   round. This recovers close pairs (one zero capturing the seeds of the
   other) and zeros near the edges of R.

Newton's method converges quadratically to a simple zero (|Δk| ~ 1e-15 µm⁻¹
in double precision) and linearly to a multiple one (the step halves at
each iteration for a double zero), hence the large ``maxiter``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .scattering import NULLITY_TOL

#: points closer than MERGE × mean spacing are the same zero
MERGE = 1e-6
#: a zero of multiplicity ≥ 2 absorbs the zeros closer than PAIR × mean spacing
PAIR = 1e-3


@dataclass(frozen=True)
class ZeroSearch:
    """Result of :func:`find_zeros`.

    Attributes
    ----------
    zeros : ndarray, complex
        Zeros in the rectangle, sorted by real part.
    multiplicities : ndarray, int
        Nullity of I − U at each zero.
    below : ndarray, complex
        Zeros found below the rectangle (Re k in the window, Im k < im_min),
        sorted by real part: the rectangle cuts through the zeros.
    rounds : int
        Deflation rounds run.
    """
    zeros: np.ndarray
    multiplicities: np.ndarray
    below: np.ndarray
    rounds: int


def mean_spacing(model, k_min, k_max):
    """Mean spacing (k_max − k_min) / N of the zeros of det(I − U), N from the
    Weyl law of the model (all Im k together)."""
    k = np.array([k_min, k_max])
    phase = k * model.index(k).real
    n_zeros = np.sum(model.lengths) / (2 * np.pi) * (phase[1] - phase[0])
    return (k_max - k_min) / max(n_zeros, 1.0)


def lattice(k_min, k_max, im_min, im_max, step, min_rows=1):
    """Seeds: a lattice of the given step covering the rectangle, cell-centred."""
    n_re = max(int(np.ceil((k_max - k_min) / step)), 1)
    n_im = max(int(np.ceil((im_max - im_min) / step)), min_rows)
    re = k_min + (np.arange(n_re) + 0.5) * (k_max - k_min) / n_re
    im = im_min + (np.arange(n_im) + 0.5) * (im_max - im_min) / n_im
    return (re[None, :] + 1j * im[:, None]).ravel()


def newton(model, seeds, box, zeros=(), multiplicities=(), max_step=np.inf,
           tol=1e-14, maxiter=100):
    """Batched Newton iterations on det(I − U(k)) / Π_i (k − z_i)^{m_i} from
    ``seeds``. ``box`` = (re_min, re_max, im_min, im_max): iterates leaving it
    are dropped. Returns the converged points."""
    k = np.array(seeds, dtype=complex).ravel()
    z = np.asarray(zeros, dtype=complex)
    m = np.asarray(multiplicities, dtype=float)
    re_min, re_max, im_min, im_max = box
    active = np.arange(k.size)
    converged = []
    for _ in range(maxiter):
        if active.size == 0:
            break
        ka = k[active]
        values = model.log_det_many(ka, derivative=True)
        exact = np.isneginf(values[:, 0].real) | ~np.isfinite(values[:, 1])   # I − U singular
        rate = np.where(exact, 1.0, values[:, 1])
        if z.size:
            rate = rate - np.sum(m / (ka[:, None] - z), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(exact, 0.0, -1.0 / rate)
        stuck = ~np.isfinite(step)                                       # stationary point
        step[stuck] = 0.0
        too_long = np.abs(step) > max_step
        step[too_long] *= max_step / np.abs(step[too_long])
        ka = ka + step
        k[active] = ka
        done = (np.abs(step) <= tol * np.abs(ka)) & ~stuck
        inside = ((ka.real >= re_min) & (ka.real <= re_max)
                  & (ka.imag >= im_min) & (ka.imag <= im_max))
        converged.append(ka[done & inside])
        active = active[~done & ~stuck & inside]
    return np.concatenate(converged) if converged else np.array([], dtype=complex)


def _clusters(points, radius):
    """Cluster label of each point: points closer than ``radius`` are linked."""
    pairs = cKDTree(np.column_stack([points.real, points.imag])).query_pairs(
        radius, output_type="ndarray")
    n = points.size
    links = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(links, directed=False)[1]


class _Zeros:
    """Zeros accumulated over the rounds of a search (see :func:`find_zeros`)."""

    def __init__(self, model, spacing, eps_accept, nullity_tol):
        self.model = model
        self.spacing = spacing
        self.eps_accept = eps_accept
        self.nullity_tol = nullity_tol
        self.k = np.array([], dtype=complex)
        self.mult = np.array([], dtype=int)
        self.sigma = np.array([], dtype=float)

    def add(self, points):
        """Accept the points that are zeros and not already known."""
        if points.size == 0:
            return
        sv = self.model.singular_values_many(points)
        ok = sv[:, -1] <= self.eps_accept
        if not np.any(ok):
            return
        k = np.concatenate([self.k, points[ok]])
        mult = np.concatenate([self.mult, np.sum(sv[ok] <= self.nullity_tol, axis=1)])
        sigma = np.concatenate([self.sigma, sv[ok, -1]])
        new = np.arange(k.size) >= self.k.size

        # one point per cluster of radius MERGE: a known zero, else the smallest σ_min
        label = _clusters(k, MERGE * self.spacing)
        order = np.lexsort((sigma, new, label))
        first = order[np.concatenate([[True], np.diff(label[order]) != 0])]
        k, mult, sigma = k[first], mult[first], sigma[first]

        # a zero of multiplicity >= 2 absorbs its neighbours within PAIR
        label = _clusters(k, PAIR * self.spacing)
        keep = np.ones(k.size, dtype=bool)
        for c in np.unique(label[mult >= 2]):
            members = np.flatnonzero(label == c)
            if members.size > 1:
                keep[members] = False
                keep[members[np.lexsort((sigma[members], -mult[members]))[0]]] = True
        self.k, self.mult, self.sigma = k[keep], mult[keep], sigma[keep]

    def per_window(self, edges, im_min, im_max):
        """Zeros found (with multiplicity) in each sub-window of the rectangle."""
        k = self.k
        inside = ((k.real >= edges[0]) & (k.real <= edges[-1])
                  & (k.imag >= im_min) & (k.imag <= im_max))
        j = np.clip(np.searchsorted(edges, k.real[inside], side="right") - 1, 0, edges.size - 2)
        return np.bincount(j, weights=self.mult[inside], minlength=edges.size - 1).astype(int)


def find_zeros(model, k_min, k_max, im_min, im_max, count=None, seeds_per_spacing=4,
               tol=1e-14, maxiter=100, max_rounds=6, eps_accept=1e-4,
               nullity_tol=NULLITY_TOL):
    """Zeros of det(I − U(k)) of ``model`` in [k_min, k_max] × [im_min, im_max]
    (see the module docstring). ``count``: the
    :class:`~.argument_principle.ZeroCount` of the same rectangle, whose
    sub-windows drive the deflation (no deflation if None).

    Returns a :class:`ZeroSearch`.
    """
    spacing = mean_spacing(model, k_min, k_max)
    height = im_max - im_min
    margin = max(spacing, height)
    box = (k_min - margin, k_max + margin, im_min - margin, im_max + margin)
    found = _Zeros(model, spacing, eps_accept, nullity_tol)

    step = spacing / seeds_per_spacing
    found.add(newton(model, lattice(k_min, k_max, im_min, im_max, step), box,
                     max_step=spacing, tol=tol, maxiter=maxiter))

    rounds = 0
    while count is not None and rounds < max_rounds:
        edges = count.edges
        missing = np.flatnonzero(found.per_window(edges, im_min, im_max) < count.counts)
        if missing.size == 0:
            break
        rounds += 1
        step /= 2
        points = []
        for j in missing:
            lo, hi = edges[j], edges[j + 1]
            width = hi - lo
            near = np.abs(found.k.real - 0.5 * (lo + hi)) <= 1.5 * width
            seeds = lattice(lo, hi, im_min, im_max, step, min_rows=3)
            points.append(newton(model, seeds, box, found.k[near], found.mult[near],
                                 max_step=spacing, tol=tol, maxiter=maxiter))
        found.add(np.concatenate(points))

    k = found.k
    in_window = (k.real >= k_min) & (k.real <= k_max)
    inside = in_window & (k.imag >= im_min) & (k.imag <= im_max)
    order = np.argsort(k.real[inside], kind="stable")
    below = np.sort_complex(k[in_window & (k.imag < im_min)])
    return ZeroSearch(zeros=k[inside][order], multiplicities=found.mult[inside][order],
                      below=below, rounds=rounds)
