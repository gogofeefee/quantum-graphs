"""
Bond scattering model: propagation D(k), vertex blocks Σ(k), U(k), S(k).

Notation (Kottos-Smilansky / Gnutzmann-Smilansky)
--------------------------------------------------
* α = internal directed bonds (2B of them), β = external leads.
* ``D(k)`` diagonal, ``D_bb = exp(i k n_eff(Re k) L_b)``.
* ``Σ_αα[b', b]`` = σ_v[port where b' leaves v, port where b arrives at v].
  ``Σ_αβ`` (lead -> bond), ``Σ_βα`` (bond -> lead), ``Σ_ββ`` (lead -> lead)
  are the other blocks of the vertex matrices.
* ``U(k) = D(k) Σ_αα(k)``; closed-graph eigenvalues solve det(I − U) = 0.
* ``S(k) = Σ_ββ + Σ_βα (I − D Σ_αα)⁻¹ D Σ_αβ``; ``S[j, i]`` is the amplitude
  leaving lead j for unit amplitude injected in lead i. Leads are numbered in
  the order of the ``None`` entries of ``port_map``.

Eigenvector convention: a solution of ``a = U a`` gives, for each directed bond
u → v, the amplitude of the wave *arriving* at v.

The constant parts of Σ are computed once; ``sigma_min_many`` and ``S_many``
evaluate many wavenumbers at once with batched LAPACK calls.
"""

import numpy as np

from .vertices import evaluate


def index_function(f, x):
    """Evaluate a constant or a function of k on an array of real k.

    Tries a vectorised call first and falls back on a loop if the function
    does not broadcast.
    """
    x = np.asarray(x, dtype=float)
    if not callable(f):
        return np.full(x.shape, float(f))
    try:
        y = np.asarray(f(x), dtype=float)
        if y.shape == x.shape:
            return y
        if y.ndim == 0:
            return np.full(x.shape, float(y))
    except (TypeError, ValueError):
        pass
    return np.array([float(f(float(xi))) for xi in x.ravel()]).reshape(x.shape)


def _map_chunks(work, ks, chunk):
    """Apply ``work`` to consecutive chunks of ``ks`` (in threads if the
    ``n_workers`` option is > 1) and concatenate the results in order."""
    from concurrent.futures import ThreadPoolExecutor
    from .config import get_option
    parts = [ks[s:s + chunk] for s in range(0, ks.size, chunk)] or [ks]
    workers = min(get_option("n_workers"), len(parts))
    if workers <= 1:
        return np.concatenate([work(p) for p in parts])
    with ThreadPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(work, parts)))


class ScatteringModel:
    """Σ blocks and derived operators of a graph (see module docstring)."""

    def __init__(self, topology, vertex_matrices, neff):
        t = topology
        self.n = t.n_directed_bonds
        self.n_leads = t.n_leads
        self.lengths = t.directed_length_array()
        self.neff = neff
        self.vertex_matrices = dict(vertex_matrices)
        self.constant = not any(callable(s) for s in self.vertex_matrices.values())
        idx = t.directed_bond_index
        lead_idx = {lead: i for i, lead in enumerate(t.leads)}
        vpm = t.virtual_port_map

        aa, ba = [], []
        for db_in, i_in in idx.items():
            v, p_in = db_in[1]
            for p_out, conn in enumerate(vpm[v]):
                if conn is None:
                    ba.append((lead_idx[(v, p_out)], i_in, v, p_out, p_in))
                else:
                    aa.append((idx[((v, p_out), conn)], i_in, v, p_out, p_in))
        ab, bb = [], []
        for i_in, (v, p_in) in enumerate(t.leads):
            for p_out, conn in enumerate(vpm[v]):
                if conn is None:
                    bb.append((lead_idx[(v, p_out)], i_in, v, p_out, p_in))
                else:
                    ab.append((idx[((v, p_out), conn)], i_in, v, p_out, p_in))
        self._entries = {"aa": (aa, (self.n, self.n)), "ab": (ab, (self.n, self.n_leads)),
                         "ba": (ba, (self.n_leads, self.n)), "bb": (bb, (self.n_leads, self.n_leads))}
        self._const_blocks = {}

    # -- vertex blocks ------------------------------------------------------

    def block(self, name, k=None):
        """Σ block ``name`` in {'aa', 'ab', 'ba', 'bb'} at wavenumber k."""
        if self.constant and name in self._const_blocks:
            return self._const_blocks[name]
        entries, shape = self._entries[name]
        values = {}
        M = np.zeros(shape, dtype=complex)
        for row, col, v, p_out, p_in in entries:
            if v not in values:
                values[v] = evaluate(self.vertex_matrices[v], k)
            M[row, col] = values[v][p_out, p_in]
        if self.constant:
            M.setflags(write=False)
            self._const_blocks[name] = M
        return M

    # -- propagation ----------------------------------------------------------

    def phases(self, k):
        """exp(i k n_eff(Re k) L_b) for scalar k (shape (n,)) or an array of k (shape (m, n))."""
        k = np.asarray(k, dtype=complex)
        n_eff = index_function(self.neff, np.real(k))
        return np.exp(1j * (k * n_eff)[..., None] * self.lengths)

    def D(self, k):
        return np.diag(self.phases(k))

    def U(self, k):
        return self.phases(k)[:, None] * self.block("aa", k)

    # -- many wavenumbers -------------------------------------------------------

    def _U_many(self, ks):
        d = self.phases(ks)                                   # (m, n)
        if self.constant:
            return d[:, :, None] * self.block("aa")[None, :, :]
        return np.stack([d[i][:, None] * self.block("aa", k) for i, k in enumerate(ks)])

    def sigma_min(self, k):
        """Smallest singular value of I − U(k) (inf if the SVD fails)."""
        return float(self.sigma_min_many(np.atleast_1d(k))[0])

    def sigma_min_many(self, ks, chunk=2048):
        """σ_min(I − U(k)) for an array of (real or complex) k."""
        ks = np.atleast_1d(np.asarray(ks, dtype=complex))
        eye = np.eye(self.n)

        def work(kc):
            A = eye - self._U_many(kc)
            try:
                return np.linalg.svd(A, compute_uv=False)[:, -1]
            except np.linalg.LinAlgError:
                out = np.empty(len(kc))
                for j, a in enumerate(A):
                    try:
                        out[j] = np.linalg.svd(a, compute_uv=False)[-1]
                    except np.linalg.LinAlgError:
                        out[j] = np.inf
                return out

        out = _map_chunks(work, ks, chunk)
        out[~np.isfinite(out)] = np.inf
        return out

    def S(self, k):
        return self.S_many(np.atleast_1d(k))[0]

    def S_many(self, ks, chunk=2048):
        """S(k) for an array of k, shape (m, n_leads, n_leads)."""
        ks = np.atleast_1d(np.asarray(ks, dtype=complex))
        eye = np.eye(self.n)
        if self.constant:                    # build the constant blocks once, outside the threads
            blocks = tuple(self.block(b) for b in ("aa", "ab", "ba", "bb"))

        def work(kc):
            d = self.phases(kc)
            if self.constant:
                aa, ab, ba, bb = blocks
                X = np.linalg.solve(eye - d[:, :, None] * aa[None], d[:, :, None] * ab[None])
                return bb[None] + ba[None] @ X
            out = np.empty((len(kc), self.n_leads, self.n_leads), dtype=complex)
            for j, k in enumerate(kc):
                aa, ab, ba, bb = (self.block(b, k) for b in ("aa", "ab", "ba", "bb"))
                X = np.linalg.solve(eye - d[j][:, None] * aa, d[j][:, None] * ab)
                out[j] = bb + ba @ X
            return out

        return _map_chunks(work, ks, chunk)

    def internal_amplitudes(self, k, lead_amplitudes):
        """Amplitudes on the directed bonds (arriving convention) for given incoming
        lead amplitudes: a = (I − D Σ_αα)⁻¹ D Σ_αβ e."""
        d = self.phases(k)
        A = np.eye(self.n) - d[:, None] * self.block("aa", k)
        return np.linalg.solve(A, d * (self.block("ab", k) @ np.asarray(lead_amplitudes)))
