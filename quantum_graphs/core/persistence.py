"""
Save paths, parameter fingerprints and verified caches.

Every expensive result (spectrum, σ_min grid, resonances, S-matrix samples...)
is cached in the graph's ``save_path``. The cache file name encodes the
numerical parameters of the computation; a JSON side file
``<cache file>.meta.json`` additionally stores

* the *physics fingerprint* of the graph (topology, lengths, vertex matrices,
  effective index... evaluated numerically, so that a changed function body
  is detected), and
* the computation parameters,

and is checked before a cache is reused:

=================  ========================================================
status             behaviour
=================  ========================================================
``"ok"``           cache reused
``"mismatch"``     cache ignored and recomputed (warning)
``"legacy"``       file written by a version without metadata: reused with a
                   warning (it cannot be verified; ``overwrite=True`` to
                   recompute)
``"missing"``      computed
=================  ========================================================
"""

import datetime
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from .. import __version__

META_SUFFIX = ".meta.json"


# ---------------------------------------------------------------------------
# Save path
# ---------------------------------------------------------------------------

def resolve_save_path(name, leaf, save_path=None, base_dir=None):
    """Directory where a graph writes its results.

    Priority: explicit ``save_path`` > ``base_dir/name/leaf`` > directory of the
    running script ``/name/leaf`` > current directory ``/name/leaf``.
    """
    if save_path is not None:
        return Path(save_path)
    if base_dir is not None:
        base = Path(base_dir).resolve()
    else:
        argv0 = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
        base = argv0.parent if (argv0 is not None and argv0.suffix == ".py" and argv0.exists()) \
            else Path.cwd()
    return base / str(name) / leaf


def topology_hash(topology):
    """8-hex hash of the bonds and leads, used in the save-directory names."""
    return hashlib.md5((str(topology.bonds) + str(topology.leads)).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _round(x):
    """JSON-friendly value with 12 significant digits (complex -> [re, im])."""
    if isinstance(x, dict):
        return {str(k): _round(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, np.ndarray):
        return _round(x.tolist())
    if isinstance(x, (complex, np.complexfloating)):
        return [float(f"{x.real:.12g}"), float(f"{x.imag:.12g}")]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.12g}")
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if x is None or isinstance(x, str):
        return x
    return repr(x)


def sample_function(f, k_samples):
    """Numerical signature of a constant or of a function of k."""
    if not callable(f):
        return _round(np.asarray(f) if isinstance(f, (list, np.ndarray)) else f)
    return {"callable": getattr(f, "__name__", type(f).__name__),
            "samples": [_round(np.asarray(f(k))) for k in k_samples]}


def fingerprint(spec):
    """SHA-256 (16 hex digits) of a JSON-serialisable description."""
    text = json.dumps(_round(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Verified cache files
# ---------------------------------------------------------------------------

def meta_path(path):
    """Path of the metadata file (.meta.json) of the cache file ``path``."""
    return str(path) + META_SUFFIX


def cache_status(path, graph_fingerprint, params=None):
    """'missing' | 'ok' | 'mismatch' | 'legacy' (see module docstring)."""
    if not os.path.exists(path):
        return "missing"
    mpath = meta_path(path)
    if not os.path.exists(mpath):
        return "legacy"
    try:
        with open(mpath) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return "mismatch"
    if meta.get("fingerprint") != graph_fingerprint:
        return "mismatch"
    if params is not None and meta.get("params") != _round(params):
        return "mismatch"
    return "ok"


def use_cache(path, graph_fingerprint, params=None, overwrite=False, label="cache"):
    """Return True if the cache at ``path`` may be loaded (warns otherwise)."""
    if overwrite:
        return False
    status = cache_status(path, graph_fingerprint, params)
    if status == "ok":
        return True
    if status == "legacy":
        warnings.warn(f"Loading {label} {path} written by an earlier version without parameter "
                      "metadata: it cannot be verified against the current graph. Pass "
                      "overwrite=True to recompute it.", RuntimeWarning, stacklevel=3)
        return True
    if status == "mismatch":
        warnings.warn(f"{label} {path} was computed with different parameters; it is "
                      "recomputed.", RuntimeWarning, stacklevel=3)
    return False


def write_cache_meta(path, graph_fingerprint, params=None, extra=None):
    """Write the metadata side file of a freshly written cache."""
    meta = {"fingerprint": graph_fingerprint,
            "params": _round(params) if params is not None else None,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "quantum_graphs_version": __version__}
    if extra:
        meta.update(_round(extra))
    with open(meta_path(path), "w") as fh:
        json.dump(meta, fh, indent=1)
