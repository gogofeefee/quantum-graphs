"""
Random-matrix-theory reference curves.

* :func:`nnsd` — nearest-neighbour spacing distributions (Wigner surmises and
  Poisson), analytic.
* :func:`sigma2_rmt` — number variance Σ²(L).
* :func:`delta3_rmt` — spectral rigidity Δ₃(L).

For Σ² and Δ₃ the Poisson case is analytic (Σ² = L, Δ₃ = L/15). The GOE and
GUE curves are tabulated values computed by Barbara Dietz; the GSE curves come
from the tables ``*_RP_model_N65536_gamma_0.9_GSE.txt``. Tables are
interpolated with cubic splines and return NaN outside their range
(0 ≤ L ≤ 30 except Σ² GUE up to 240 and Σ² GSE from 0.2; see ``table_range``).

Convention note for Δ₃
----------------------
The Δ₃ tables agree with the estimator used by the package
(:func:`quantum_graphs.stats.spectral.spectral_rigidity` with
``windows="level"``: each window starts on an unfolded level, which is
excluded). For L ≲ 3 this differs from the textbook Dyson-Mehta Δ₃, which
averages over uniformly placed windows (e.g. GOE: 0.039 vs 0.060 at L = 1).
:func:`delta3_dyson_mehta` gives the exact Dyson-Mehta curve for GOE / GUE /
Poisson, to be compared with ``windows="uniform"``. Both conventions coincide
for L ≳ 5. See ``docs/scientific_notes.md``.
"""

from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import interp1d

_ENSEMBLES = ("poisson", "goe", "gue", "gse")
_ALIASES = {"poissonian": "poisson", "poisson": "poisson",
            "goe": "goe", "gue": "gue", "gse": "gse"}

_TABLE_FILES = {
    ("sigma2", "goe"): "sggoe.txt",
    ("sigma2", "gue"): "sggue.txt",
    ("sigma2", "gse"): "sigma2_RP_model_N65536_gamma_0.9_GSE.txt",
    ("delta3", "goe"): "delta3goe.txt",
    ("delta3", "gue"): "delta3gue.txt",
    ("delta3", "gse"): "delta3_RP_model_N65536_gamma_0.9_GSE.txt",
}


def normalize_ensemble(name):
    """Return the canonical ensemble name ('poisson', 'goe', 'gue', 'gse').

    Raises
    ------
    ValueError
        For an unknown name (no silent fallback to Poisson).
    """
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unknown ensemble {name!r}; expected one of "
                         f"'poisson' ('poissonian'), 'GOE', 'GUE', 'GSE'.")
    return _ALIASES[key]


# ---------------------------------------------------------------------------
# Nearest-neighbour spacing distribution
# ---------------------------------------------------------------------------

def nnsd(s, ensemble="GOE"):
    """Nearest-neighbour spacing distribution P(s) (unit mean spacing).

    Poisson: exp(-s). GOE/GUE/GSE: Wigner surmises
    (π/2) s e^{-πs²/4}, (32/π²) s² e^{-4s²/π}, (2¹⁸/3⁶π³) s⁴ e^{-64s²/9π}.
    """
    s = np.asarray(s, dtype=float)
    ens = normalize_ensemble(ensemble)
    if ens == "poisson":
        return np.exp(-s)
    if ens == "goe":
        return (np.pi / 2) * s * np.exp(-np.pi / 4 * s**2)
    if ens == "gue":
        return (32 / np.pi**2) * s**2 * np.exp(-4 / np.pi * s**2)
    return (2**18 / 3**6 / np.pi**3) * s**4 * np.exp(-64 / (9 * np.pi) * s**2)


# ---------------------------------------------------------------------------
# Tabulated Σ² and Δ₃
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _table(quantity, ensemble):
    fname = _TABLE_FILES[(quantity, ensemble)]
    with resources.files("quantum_graphs.stats").joinpath("data", fname).open("r") as fh:
        data = np.loadtxt(fh)
    L, y = data[:, 0], data[:, 1]
    if quantity == "delta3":
        # Δ₃(0) = 0, not present in the tables
        L, y = np.concatenate(([0.0], L)), np.concatenate(([0.0], y))
    return L, y, interp1d(L, y, kind="cubic", bounds_error=False, fill_value=np.nan)


def table_range(quantity, ensemble):
    """(L_min, L_max) covered by the table of ``quantity`` ('sigma2'|'delta3')."""
    L, _, _ = _table(quantity, normalize_ensemble(ensemble))
    return float(L[0]), float(L[-1])


def sigma2_rmt(L, ensemble="GOE"):
    """Number variance Σ²(L). Poisson: L; other ensembles: interpolated table."""
    ens = normalize_ensemble(ensemble)
    if ens == "poisson":
        return np.asarray(L, dtype=float) * 1.0
    return _table("sigma2", ens)[2](L)


def delta3_rmt(L, ensemble="GOE"):
    """Spectral rigidity Δ₃(L) in the convention of the tables (see module doc).

    Poisson: L/15; other ensembles: interpolated table.
    """
    ens = normalize_ensemble(ensemble)
    if ens == "poisson":
        return np.asarray(L, dtype=float) / 15.0
    return _table("delta3", ens)[2](L)


# ---------------------------------------------------------------------------
# Exact Dyson-Mehta curves from the two-level cluster function
# ---------------------------------------------------------------------------

def _cluster_function(r, ensemble):
    """Two-level cluster function Y₂(r) for GOE / GUE (Mehta)."""
    from scipy.special import sici
    r = np.asarray(r, dtype=float)
    s = np.sinc(r)
    if ensemble == "gue":
        return s**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = np.where(r > 0, (np.cos(np.pi * r) * np.pi * r - np.sin(np.pi * r))
                      / (np.pi * r**2), 0.0)
    return s**2 + (0.5 - sici(np.pi * r)[0] / np.pi) * ds


def _scalar_or_array(func, L):
    L_arr = np.asarray(L, dtype=float)
    out = np.array([func(float(x)) for x in L_arr.ravel()]).reshape(L_arr.shape)
    return out if out.ndim else float(out)


def sigma2_dyson_mehta(L, ensemble="GOE"):
    """Exact Σ²(L) = L - 2∫₀ᴸ (L - r) Y₂(r) dr (Poisson: L)."""
    from scipy.integrate import quad
    ens = normalize_ensemble(ensemble)
    if ens == "poisson":
        return np.asarray(L, dtype=float) * 1.0
    if ens == "gse":
        raise NotImplementedError("Exact GSE curve not implemented; use sigma2_rmt.")

    def one(x):
        if x <= 0:
            return 0.0
        return x - 2 * quad(lambda r: (x - r) * _cluster_function(r, ens), 0, x, limit=400)[0]
    return _scalar_or_array(one, L)


def delta3_dyson_mehta(L, ensemble="GOE"):
    """Exact Dyson-Mehta Δ₃(L) (uniformly placed windows).

    Δ₃(L) = L/15 - 1/(15 L⁴) ∫₀ᴸ (L - r)³ (2L² - 9Lr - 3r²) Y₂(r) dr.
    """
    from scipy.integrate import quad
    ens = normalize_ensemble(ensemble)
    if ens == "poisson":
        return np.asarray(L, dtype=float) / 15.0
    if ens == "gse":
        raise NotImplementedError("Exact GSE curve not implemented; use delta3_rmt.")

    def one(x):
        if x <= 0:
            return 0.0
        f = lambda r: (x - r) ** 3 * (2 * x**2 - 9 * x * r - 3 * r**2) * _cluster_function(r, ens)
        return x / 15.0 - quad(f, 0, x, limit=400)[0] / (15 * x**4)
    return _scalar_or_array(one, L)
