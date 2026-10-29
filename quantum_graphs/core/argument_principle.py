"""
Number of zeros of det(I − U(k)) in a rectangle of the complex k-plane
(argument principle): the completeness certificate of a resonance search.

For a holomorphic f without zero on the boundary C of a rectangle R,

    N(R) = (1 / 2π) Δ_C arg f(k)      (C run counterclockwise)

is the number of zeros of f inside R counted with multiplicity. With
f = det(I − U(k)) = Π_s det(I − U_s(k)) (see :mod:`.sectors`), N(R) is the
number of resonances in R, each counted as many times as the dimension of
ker(I − U): a degenerate resonance counts twice, and so do two resonances
merged by a search.

Dispersion
----------
A dispersive n_eff is continued to first order off the real axis
(:mod:`.scattering`): the phase φ(x + iy) = (x + iy)(n_eff(x) + iy n_eff'(x))
is holomorphic up to terms of order y n_eff'', and the map k ↦ φ has the
Jacobian n_g² + O(y) > 0, so the winding (topological degree) counts every
zero of the function the search actually solves, with its multiplicity. The
same argument holds for vertex matrices σ(Re k) that vary slowly over the
width of a resonance.

Sampling
--------
arg det is sampled along each edge and points are inserted (bisection) until,
on every interval of length h between two samples, the phase changes by less
than ``max_step`` and h |d log det / dk| < ``max_step`` at both ends. The
second condition matters: a double zero (partner sectors) or a close pair
near the contour turns the phase by ≈ 2π within one interval, which the
first condition cannot see; |d log det/dk| ≈ m / distance near a zero of
multiplicity m forces the bisection down to its scale. The initial density
resolves the phase advance of the longest term of the determinant,
Σ_d n_g L_d per unit of k. Sharing the bottom and top edges, the
sub-rectangles between vertical ``cuts`` are counted for the price of their
vertical edges, which localises a discrepancy with a search.
"""

from dataclasses import dataclass

import numpy as np


#: largest phase change of det(I − U) allowed between two samples
MAX_PHASE_STEP = np.pi / 8
#: initial number of samples per 2π of the longest phase advance
SAMPLES_PER_TURN = 32


@dataclass(frozen=True)
class ZeroCount:
    """Zeros of det(I − U(k)) in the sub-rectangles
    [edges[j], edges[j+1]] × [im_min, im_max] of the complex k-plane.

    Attributes
    ----------
    edges : ndarray
        Re k boundaries of the sub-rectangles (µm⁻¹), increasing.
    im_min, im_max : float
        Lower and upper edges in Im k (µm⁻¹).
    winding : ndarray
        (1 / 2π) Δ arg det along each sub-rectangle boundary (≈ integers).
    n_samples : int
        Number of determinant evaluations.
    """
    edges: np.ndarray
    im_min: float
    im_max: float
    winding: np.ndarray
    n_samples: int

    @property
    def counts(self):
        """Number of zeros in each sub-rectangle (rounded winding numbers)."""
        return np.rint(self.winding).astype(int)

    @property
    def total(self):
        """Number of zeros in the whole rectangle."""
        return int(self.counts.sum())

    @property
    def defect(self):
        """Largest distance of a winding number to an integer (≪ 1 when the
        contour is sampled finely enough)."""
        return float(np.max(np.abs(self.winding - self.counts))) if self.winding.size else 0.0

    def __repr__(self):
        return (f"ZeroCount({self.total} zeros in Re k ∈ [{self.edges[0]:.6f}, "
                f"{self.edges[-1]:.6f}], Im k ∈ [{self.im_min:.3e}, {self.im_max:.3e}], "
                f"{self.edges.size - 1} sub-rectangles, defect {self.defect:.1e})")


def _phase_increments(model, z, max_step, max_points):
    """Unwrapped change of arg det(I − U) from z[0] to every point of the
    polyline z, inserting points between consecutive samples whose phases
    differ by more than ``max_step``. Returns (phase at the points of z,
    number of evaluations)."""
    z = np.asarray(z, dtype=complex)
    pts = z.copy()
    is_node = np.ones(pts.size, dtype=bool)
    values = model.log_det_many(pts, derivative=True)
    while True:
        logdet, rate = values[:, 0], np.abs(values[:, 1])
        if np.any(np.isneginf(logdet.real)):
            k = pts[np.isneginf(logdet.real)][0]
            raise RuntimeError(f"det(I − U) vanishes on the contour at k = {k:.12g}: "
                               "move the edge of the rectangle.")
        step = np.angle(np.exp(1j * np.diff(logdet.imag)))
        h = np.abs(np.diff(pts))
        bad = np.flatnonzero((np.abs(step) > max_step)
                             | (h * np.maximum(rate[:-1], rate[1:]) > max_step))
        if bad.size == 0:
            break
        tiny = np.abs(pts[bad + 1] - pts[bad]) <= 8 * np.finfo(float).eps * np.abs(pts[bad])
        if np.any(tiny) or pts.size + bad.size > max_points:
            k = pts[bad[0]]
            raise RuntimeError(f"Phase of det(I − U) not resolved near k = {k:.12g}: a zero "
                               "lies on (or extremely close to) the contour; move the edge "
                               "of the rectangle.")
        mid = 0.5 * (pts[bad] + pts[bad + 1])
        pts = np.insert(pts, bad + 1, mid)
        is_node = np.insert(is_node, bad + 1, False)
        values = np.insert(values, bad + 1, model.log_det_many(mid, derivative=True), axis=0)
    phase = np.concatenate([[0.0], np.cumsum(step)])
    return phase[is_node], pts.size


def _phase_rate(model, k_min, k_max):
    """Largest phase advance per unit of k of the terms of det(I − U):
    Σ_d n_g L_d (maximum of n_g over the window)."""
    n_g = model.phase_rate(np.linspace(k_min, k_max, 16))
    return float(np.max(np.abs(n_g)) * np.sum(model.lengths))


def count_zeros(model, k_min, k_max, im_min, im_max, cuts=(),
                max_step=MAX_PHASE_STEP, max_points=5_000_000):
    """Zeros of det(I − U(k)) of ``model`` in [k_min, k_max] × [im_min, im_max]
    and in its sub-rectangles separated by the vertical lines Re k = ``cuts``
    (see the module docstring).

    Raises RuntimeError if a zero lies on the contour. Returns a
    :class:`ZeroCount`.
    """
    if not (k_min < k_max and im_min < im_max):
        raise ValueError("Empty rectangle.")
    cuts = np.sort(np.asarray(cuts, dtype=float).ravel())
    edges = np.concatenate([[k_min], cuts[(cuts > k_min) & (cuts < k_max)], [k_max]])

    # bottom and top edges, left to right, through every cut
    n_init = int(np.ceil(SAMPLES_PER_TURN * _phase_rate(model, k_min, k_max)
                         * (k_max - k_min) / (2 * np.pi))) + 64
    x = np.unique(np.concatenate([np.linspace(k_min, k_max, n_init), edges]))
    at_edges = np.searchsorted(x, edges)
    bottom, n_b = _phase_increments(model, x + 1j * im_min, max_step, max_points)
    top, n_t = _phase_increments(model, x + 1j * im_max, max_step, max_points)
    n_samples = n_b + n_t

    # vertical edges, bottom to top
    y = np.linspace(im_min, im_max, 64)
    up = np.empty(edges.size)
    for j, xe in enumerate(edges):
        phase, n_v = _phase_increments(model, xe + 1j * y, max_step, max_points)
        up[j] = phase[-1]
        n_samples += n_v

    b, t = bottom[at_edges], top[at_edges]
    winding = (np.diff(b) + up[1:] - np.diff(t) - up[:-1]) / (2 * np.pi)
    return ZeroCount(edges=edges, im_min=float(im_min), im_max=float(im_max),
                     winding=winding, n_samples=int(n_samples))
