"""
Vertex scattering matrices.

A vertex matrix σ is indexed ``σ[port_out, port_in]``. It can be a constant
array or a callable ``σ(k)`` of the (real part of the) vacuum wavenumber.
"""

import warnings

import numpy as np

#: 2-port pass-through matrix used for the virtual vertices
PASS_THROUGH = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def neumann_matrix(n):
    """Neumann-Kirchhoff matrix of an n-port vertex, σ = (2/n) J − I."""
    n = int(n)
    return 2.0 / n * np.ones((n, n), dtype=complex) - np.eye(n, dtype=complex)


def _check_coupling_value(C, where=""):
    C_arr = np.asarray(C, dtype=float)
    if not np.all(np.isfinite(C_arr)) or np.any(C_arr < 0) or np.any(C_arr > 1):
        raise ValueError(f"Coupling constant{where} must lie in [0, 1], got {C!r}.")


def directional_coupler_matrix(C):
    """4×4 matrix of a lossless directional coupler with power cross-coupling C.

    Ports (0, 1) form one arm and ports (2, 3) the other; there is no
    back-reflection::

        σ = [[0,      0,      √(1−C), i√C   ],
             [0,      0,      i√C,    √(1−C)],
             [√(1−C), i√C,    0,      0     ],
             [i√C,    √(1−C), 0,      0     ]]

    Parameters
    ----------
    C : float or callable
        Constant in [0, 1], or a function ``C(k)``; in the latter case a
        callable ``σ(k)`` is returned.
    """
    def matrix(c_val):
        _check_coupling_value(c_val)
        t, x = np.sqrt(1.0 - c_val), 1j * np.sqrt(c_val)
        return np.array([[0, 0, t, x], [0, 0, x, t], [t, x, 0, 0], [x, t, 0, 0]], dtype=complex)

    if callable(C):
        def sigma_k(k=None):
            return matrix(float(np.real(C(k))))
        sigma_k.coupling = C
        return sigma_k
    return matrix(float(C))


def couplings_for_vertices(vertex_couplings, n_vertices, default=None):
    """Return one coupling (float or callable) per vertex.

    ``vertex_couplings`` may be a float, a callable C(k), a list (one value per
    vertex) or a dict {vertex: C}; vertices missing from a dict get ``default``.
    """
    if isinstance(vertex_couplings, dict):
        out = [vertex_couplings.get(v, default) for v in range(n_vertices)]
    elif isinstance(vertex_couplings, (list, tuple, np.ndarray)):
        if len(vertex_couplings) != n_vertices:
            raise ValueError(f"vertex_couplings has {len(vertex_couplings)} values for "
                             f"{n_vertices} vertices.")
        out = list(vertex_couplings)
    else:
        out = [vertex_couplings] * n_vertices
    return out


def evaluate(sigma, k):
    """Value of a vertex matrix at wavenumber ``k`` (real part used)."""
    if callable(sigma):
        return np.asarray(sigma(None if k is None else np.real(k)), dtype=complex)
    return np.asarray(sigma, dtype=complex)


def validate_vertex_matrices(matrices, valencies, k_samples, unitarity_tol=1e-8):
    """Check shapes (always) and unitarity (warning only) of the vertex matrices.

    ``matrices`` is a dict {vertex: array or callable}; callables are evaluated
    at ``k_samples``.
    """
    lossy = []
    for v, sigma in matrices.items():
        if callable(sigma):
            values = [evaluate(sigma, k) for k in k_samples]   # nothing to check if no sample
        else:
            values = [evaluate(sigma, None)]
        for val in values:
            n = valencies[v]
            if val.shape != (n, n):
                raise ValueError(f"Vertex {v} has {n} ports but its scattering matrix has "
                                 f"shape {val.shape}.")
            if not np.all(np.isfinite(val)):
                raise ValueError(f"Vertex {v}: scattering matrix contains NaN or inf.")
            if np.linalg.norm(val.conj().T @ val - np.eye(n)) > unitarity_tol:
                lossy.append(v)
                break
    if lossy:
        warnings.warn(f"Vertex scattering matrices of vertices {sorted(set(lossy))} are not "
                      "unitary (lossy or gain vertices). Intended?", RuntimeWarning, stacklevel=3)


def resolve_vertex_matrices(vertex_matrices, n_vertices):
    """Normalise the ``vertex_matrices`` argument of the base graph classes.

    Accepts a dict {vertex: σ}, a list (one σ per vertex), a single array (same
    σ for every vertex) or a single callable σ(k). Returns a dict.
    """
    if isinstance(vertex_matrices, dict):
        missing = [v for v in range(n_vertices) if v not in vertex_matrices]
        if missing:
            raise ValueError(f"vertex_matrices has no matrix for vertices {missing}.")
        return {v: vertex_matrices[v] for v in range(n_vertices)}
    if isinstance(vertex_matrices, np.ndarray) and vertex_matrices.ndim == 2:
        return {v: vertex_matrices for v in range(n_vertices)}
    if isinstance(vertex_matrices, (list, tuple)):
        if len(vertex_matrices) != n_vertices:
            raise ValueError(f"vertex_matrices has {len(vertex_matrices)} entries for "
                             f"{n_vertices} vertices.")
        return {v: vertex_matrices[v] for v in range(n_vertices)}
    if callable(vertex_matrices):
        return {v: vertex_matrices for v in range(n_vertices)}
    raise ValueError("vertex_matrices must be a dict, a list, a 2D array or a callable sigma(k).")
