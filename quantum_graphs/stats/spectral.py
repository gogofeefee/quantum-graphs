"""
Spectral statistics of an (unfolded) spectrum, as pure functions.

* :func:`weyl_staircase` — smooth counting function N(k) = (L_s/2π) ∫ n_g dk;
* :func:`unfold` — unfolded levels x_i = N_Weyl(k_i) (unit mean spacing);
* :func:`spacings` — nearest-neighbour spacings of an unfolded spectrum;
* :func:`number_variance` — Σ²(L);
* :func:`spectral_rigidity` — Δ₃(L).

All statistics assume an unfolded spectrum with unit mean spacing; use
:func:`mean_spacing` to check it.
"""

import numpy as np


def _index_values(f, k):
    from ..core.scattering import index_function
    return index_function(f, k)


def weyl_staircase(k, directed_length, ng, k0=0.0):
    """Smooth counting function N(k) = (L_s/2π) ∫_{k0}^{k} n_g(κ) dκ.

    ``directed_length`` L_s is the summed length of the directed bonds that
    carry the levels counted: 2 L (L = total bond length) for the whole
    spectrum of a graph, the length of one sector for the spectrum of that
    sector (see :mod:`quantum_graphs.core.sectors`). ``ng`` is a float or a
    function of k. For a scalar k the integral uses 512 trapezoids; for an
    array, the trapezoid rule on the sorted values of k (and k0).
    """
    scale = float(directed_length) / (2 * np.pi)
    k0 = float(np.real(k0))
    if np.isscalar(k) or np.ndim(k) == 0:
        kr = float(np.real(k))
        if callable(ng):
            ks = np.linspace(k0, kr, 512)
            return scale * np.trapezoid(_index_values(ng, ks), ks)
        return scale * float(ng) * (kr - k0)
    k_arr = np.real(np.asarray(k)).astype(float)
    if not callable(ng):
        return scale * float(ng) * (k_arr - k0)
    nodes = np.unique(np.concatenate([k_arr.ravel(), [k0]]))
    vals = _index_values(ng, nodes)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (vals[1:] + vals[:-1]) * np.diff(nodes))])
    F -= F[np.searchsorted(nodes, k0)]
    return (scale * F[np.searchsorted(nodes, k_arr.ravel())]).reshape(k_arr.shape)


def unfold(spectrum, directed_length, ng, k0=0.0):
    """Unfolded levels N(k_i) of :func:`weyl_staircase`, sorted (real parts of
    complex k are used)."""
    k = np.sort(np.real(np.asarray(spectrum)))
    return weyl_staircase(k, directed_length, ng, k0)


def spacings(unfolded):
    """Nearest-neighbour spacings of a sorted unfolded spectrum."""
    return np.diff(np.sort(np.asarray(unfolded, dtype=float)))


def mean_spacing(unfolded):
    """Mean spacing of an unfolded spectrum (should be ≈ 1)."""
    s = spacings(unfolded)
    return float(np.mean(s)) if s.size else float("nan")


def number_variance(unfolded, dE, L_max):
    """Number variance Σ²(L) = ⟨(n(E, L) − L)²⟩.

    Windows [E, E + L) start on a uniform grid of step ``dE``; L takes the
    values 0, dE, 2 dE, ... ≤ ``L_max``.

    Returns
    -------
    L_vals, sigma2_vals : ndarray
    """
    eig = np.sort(np.asarray(unfolded, dtype=float))
    NR = int(np.floor((eig[-1] - eig[0]) / dE))
    grid = eig[0] + np.arange(1, NR + 1) * dE
    counts = np.searchsorted(eig, grid, side="left").astype(float)
    IR_max = min(NR - 1, int(np.floor(L_max / dE)))
    L_vals = np.arange(0, IR_max + 1) * dE
    sigma2 = np.zeros_like(L_vals)
    for IR in range(1, IR_max + 1):
        n = counts[IR:] - counts[:-IR]
        sigma2[IR] = np.mean((n - IR * dE) ** 2)
    return L_vals, sigma2


def _delta3_window(levels, x, L):
    """Δ₃ of one window [x, x + L] from the levels inside it (closed form,
    identical to the least-squares definition)."""
    sub = levels - (x + L / 2)
    n = sub.size
    weights = np.arange(n - 1, -n, -2)                 # n − 2i + 1, i = 1..n
    s1, s2 = np.sum(sub), np.sum(sub**2)
    return (n**2 / 16 - s1**2 / L**2 + 3 * n / (2 * L**2) * s2
            - 3 / L**4 * s2**2 + np.sum(weights * sub) / L)


def spectral_rigidity(unfolded, dE, L_max, windows="level", n_windows=None):
    """Spectral rigidity Δ₃(L) (Dyson-Mehta least-squares deviation).

    Parameters
    ----------
    unfolded : array
        Unfolded spectrum.
    dE : float
        Step of L (L = dE, 2 dE, ... ≤ L_max).
    windows : {"level", "uniform"}
        "level" (default, historical): one window (x, x + L] starting on each
        level x (excluded), as long as it fits in the spectrum — this is the
        convention of the tabulated RMT curves
        :func:`~quantum_graphs.stats.rmt.delta3_rmt`. "uniform": windows
        starting on a uniform grid of ``n_windows`` points (default: one per
        level), i.e. the textbook Dyson-Mehta definition, to be compared with
        :func:`~quantum_graphs.stats.rmt.delta3_dyson_mehta`. The two differ
        for L ≲ 3 (see docs/scientific_notes.md).

    Returns
    -------
    L_vals, delta3_vals : ndarray (L = 0 included with Δ₃ = 0)
    """
    eig = np.sort(np.asarray(unfolded, dtype=float))
    NR = int(np.floor((eig[-1] - eig[0]) / dE))
    IR_max = min(NR - 1, int(np.floor(L_max / dE)))
    L_vals = np.arange(1, IR_max + 1) * dE
    out = np.zeros(L_vals.size)
    for i, L in enumerate(L_vals):
        if windows == "level":
            starts = eig[eig + L <= eig[-1]]
            lo = np.searchsorted(eig, starts, side="right")       # levels > x
        elif windows == "uniform":
            m = n_windows or eig.size
            starts = np.linspace(eig[0], eig[-1] - L, m)
            lo = np.searchsorted(eig, starts, side="right")
        else:
            raise ValueError("windows must be 'level' or 'uniform'.")
        hi = np.searchsorted(eig, starts + L, side="right")       # levels <= x + L
        vals = [_delta3_window(eig[a:b], x, L) for x, a, b in zip(starts, lo, hi)]
        out[i] = np.mean(vals) if vals else np.nan
    return np.insert(L_vals, 0, 0.0), np.insert(out, 0, 0.0)
