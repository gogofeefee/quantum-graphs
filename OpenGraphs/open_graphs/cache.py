#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-matrix cache utilities shared by plot_S_components and plot_S_complex_plane.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def sanitize_float(x: float) -> str:
    return f"{x:.12g}".replace(".", "p").replace("+", "").replace("-", "m")


def comp_tag(comps):
    return "_".join([f"S{i}{j}" for i, j in comps])


def cache_filename_full(lmin: float, lmax: float, N: int, kim: float) -> str:
    tag = (
        f"cache_Smatrix_full"
        f"__lam{sanitize_float(lmin)}-{sanitize_float(lmax)}"
        f"__N{int(N)}"
        f"__kIm{sanitize_float(kim)}"
    )
    return tag + ".npz"


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def interp_complex(lam_src, y_src, lam_tgt):
    re = np.interp(lam_tgt, lam_src, np.real(y_src))
    im = np.interp(lam_tgt, lam_src, np.imag(y_src))
    return re + 1j * im


def interp_S_cube(lam_src, S_src, lam_tgt):
    """
    Interpolate a full S cube:
        S_src shape = (N_src, nrow, ncol)
    onto lam_tgt, returning:
        S_tgt shape = (N_tgt, nrow, ncol)
    """
    S_src = np.asarray(S_src)
    if S_src.ndim != 3:
        raise ValueError("S_src must have shape (N, nrow, ncol).")
    N_tgt = len(lam_tgt)
    nrow  = S_src.shape[1]
    ncol  = S_src.shape[2]
    S_tgt = np.empty((N_tgt, nrow, ncol), dtype=np.complex128)
    for i in range(nrow):
        for j in range(ncol):
            S_tgt[:, i, j] = interp_complex(lam_src, S_src[:, i, j], lam_tgt)
    return S_tgt


# ---------------------------------------------------------------------------
# Progress printing helpers
# ---------------------------------------------------------------------------

def progress_print(idx, total, last_pct, verbose=True, prefix="Computing S..."):
    if not verbose:
        return last_pct
    pct = int(100 * (idx + 1) / total)
    if pct != last_pct:
        print(f"\r{prefix} {pct:3d}%", end="", flush=True)
        return pct
    return last_pct


def finish_progress(verbose=True, prefix="Computing S..."):
    if verbose:
        print(f"\r{prefix} 100%")


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_full_cache(path, lambda_values, S_full, k_imag=0.0):
    payload = {
        "lambda_values": np.asarray(lambda_values, dtype=float),
        "S_full":        np.asarray(S_full, dtype=np.complex128),
        "meta_lambda_min": float(np.min(lambda_values)),
        "meta_lambda_max": float(np.max(lambda_values)),
        "meta_n_points":   int(len(lambda_values)),
        "meta_k_imag":     float(k_imag),
    }
    np.savez_compressed(path, **payload)


def load_full_cache_if_exact(path, lambda_req_exact):
    """
    Load a cached S-matrix if it matches lambda_req_exact exactly.
    Returns the S_full array (shape (N, nrow, ncol)) or None.
    """
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    if "lambda_values" not in data or "S_full" not in data:
        return None
    lam_cached   = np.asarray(data["lambda_values"])
    S_full_cached = np.asarray(data["S_full"])
    if lam_cached.shape != lambda_req_exact.shape:
        return None
    if not np.allclose(lam_cached, lambda_req_exact, rtol=0, atol=0):
        return None
    if S_full_cached.ndim != 3 or S_full_cached.shape[0] != len(lambda_req_exact):
        return None
    return S_full_cached


def find_covering_cache(workdir, kim_str, lambda_min, lambda_max):
    """
    Search for the smallest full-S cache that covers [lambda_min, lambda_max].
    Returns (path, lam_src, S_src) or (None, None, None).
    """
    prefix = "cache_Smatrix_full__"
    best_path = None
    best_span = None
    best_lam  = None
    best_S    = None

    for fn in os.listdir(workdir):
        if not (fn.startswith(prefix) and fn.endswith(".npz") and
                (f"__kIm{kim_str}.npz" in fn)):
            continue
        full_path = os.path.join(workdir, fn)
        try:
            data = np.load(full_path, allow_pickle=False)
        except Exception:
            continue
        if "lambda_values" not in data or "S_full" not in data:
            continue
        lam_src = np.asarray(data["lambda_values"])
        S_src   = np.asarray(data["S_full"])
        if lam_src.ndim != 1 or lam_src.size < 2:
            continue
        if S_src.ndim != 3 or S_src.shape[0] != lam_src.size:
            continue
        if lam_src[0] > lam_src[-1]:
            lam_src = lam_src[::-1]
            S_src   = S_src[::-1, :, :]
        lam_lo, lam_hi = float(lam_src[0]), float(lam_src[-1])
        if not (lam_lo <= lambda_min and lam_hi >= lambda_max):
            continue
        span = lam_hi - lam_lo
        if best_span is None or span < best_span:
            best_span = span
            best_path = full_path
            best_lam  = lam_src
            best_S    = S_src

    return best_path, best_lam, best_S


# ---------------------------------------------------------------------------
# High-level: get or compute the full S-matrix on a wavelength grid
# ---------------------------------------------------------------------------

def get_S_full(graph, lambda_min, lambda_max, n_points, k_imag,
               reuse_if_included=True, verbose=True):
    """
    Return (lambda_req, real_k_req, delta_k, S_full_req) where
    S_full_req has shape (n_points, n_leads, n_leads), complex128.

    Loads from cache if possible; otherwise computes and saves.
    """
    k_min = 2.0 * np.pi / lambda_max
    k_max = 2.0 * np.pi / lambda_min
    real_k_req = np.linspace(k_min, k_max, int(n_points))
    lambda_req = 2.0 * np.pi / real_k_req
    k_req      = real_k_req + 1j * float(k_imag)
    delta_k    = real_k_req[1] - real_k_req[0]

    workdir = getattr(graph, "save_path", None)
    if workdir is None:
        workdir = os.getcwd()
    os.makedirs(workdir, exist_ok=True)

    cache_fname = cache_filename_full(lambda_min, lambda_max, n_points, k_imag)
    cache_path  = os.path.join(workdir, cache_fname)

    S_full_req = None

    if reuse_if_included:
        # 1) exact cache
        S_full_req = load_full_cache_if_exact(cache_path, lambda_req)
        if S_full_req is not None and verbose:
            print(f"✓ Loaded exact cached full S(λ) from: {cache_path}")

        # 2) broader covering cache
        if S_full_req is None:
            kim_str = sanitize_float(k_imag)
            best_path, lam_src, S_src = find_covering_cache(
                workdir, kim_str, lambda_min, lambda_max
            )
            if best_path is not None:
                if verbose:
                    print(f"✓ Reusing cached full S(λ) via interpolation from: {best_path}")
                S_full_req = interp_S_cube(lam_src, S_src, lambda_req)
                save_full_cache(cache_path, lambda_req, S_full_req, k_imag)
                if verbose:
                    print(f"✓ Cached interpolated full S(λ) to: {cache_path}")

        # 3) compute
        if S_full_req is None:
            if verbose:
                print("⚠ No covering full-S cache found → computing full S(λ) on requested grid.")
            S_full_req = _compute_S_full(graph, k_req, cache_path, lambda_req, k_imag, verbose)

    else:
        S_full_req = load_full_cache_if_exact(cache_path, lambda_req)
        if S_full_req is not None and verbose:
            print(f"✓ Loaded exact cached full S(λ) from: {cache_path}")
        if S_full_req is None:
            if verbose:
                print("✓ Computing full S(λ) on requested grid (reuse_if_included=False).")
            S_full_req = _compute_S_full(graph, k_req, cache_path, lambda_req, k_imag, verbose)

    return lambda_req, real_k_req, delta_k, S_full_req


def _compute_S_full(graph, k_req, cache_path, lambda_req, k_imag, verbose):
    S0 = np.asarray(graph.compute_S_matrix(k_req[0]), dtype=np.complex128)
    if S0.ndim != 2 or S0.shape[0] != S0.shape[1]:
        raise ValueError("compute_S_matrix(k) must return a square 2D array.")
    n_ports    = S0.shape[0]
    S_full_req = np.empty((len(k_req), n_ports, n_ports), dtype=np.complex128)
    S_full_req[0] = S0
    last_pct = -1
    total    = len(k_req)
    last_pct = progress_print(0, total, last_pct, verbose)
    for idx_k in range(1, total):
        S_full_req[idx_k] = np.asarray(graph.compute_S_matrix(k_req[idx_k]), dtype=np.complex128)
        last_pct = progress_print(idx_k, total, last_pct, verbose)
    finish_progress(verbose)
    save_full_cache(cache_path, lambda_req, S_full_req, k_imag)
    if verbose:
        print(f"✓ Cached full S(λ) to: {cache_path}")
    return S_full_req
