"""
Bond scattering model: propagation D(k), vertex blocks Σ(k), U(k), S(k).

Notation (Kottos-Smilansky / Gnutzmann-Smilansky)
--------------------------------------------------
* α = directed bonds (2B of them, numbered as in :mod:`.topology`), β = leads.
* ``D(k)`` diagonal, ``D_bb = exp(i k n_eff(Re k) L_b)``.
* ``Σ_αα[b', b]`` = σ_v[port where b' leaves v, port where b arrives at v].
  ``Σ_αβ`` (lead -> bond), ``Σ_βα`` (bond -> lead), ``Σ_ββ`` (lead -> lead)
  are the other blocks of the vertex matrices.
* ``U(k) = D(k) Σ_αα(k)``; closed-graph eigenvalues solve det(I − U) = 0.
* ``S(k) = Σ_ββ + Σ_βα (I − D Σ_αα)⁻¹ D Σ_αβ``; ``S[j, i]`` is the amplitude
  leaving lead j for unit amplitude injected in lead i. Leads are numbered in
  the order of the graph's ``leads``.

Eigenvector convention: a solution of ``a = U a`` gives, for each directed bond
u → v, the amplitude of the wave *arriving* at v.

The constant parts of Σ are computed once; ``sigma_min_many``, ``log_det_many``
and ``S_many`` evaluate many wavenumbers at once with batched LAPACK calls.
``model.sector(directed)`` restricts U to one sector (see :mod:`.sectors`).

Multiplicity of a level k*: the number of singular values of I − U(k*) below
``NULLITY_TOL`` (one threshold for the spectra, the resonances and the
wavefunctions).
"""

import numpy as np

from .vertices import evaluate

#: singular values of I − U(k) at or below this value count as zero
NULLITY_TOL = 1e-4


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
        self.n = t.n_directed
        self.n_leads = t.n_leads
        self.lengths = np.asarray(t.directed_length, dtype=float)
        self.neff = neff
        self.vertex_matrices = dict(vertex_matrices)
        self.constant = not any(callable(s) for s in self.vertex_matrices.values())

        def outputs(v):
            """(port, 'a' or 'b', row index) for every port of v."""
            for p_out in range(t.valency[v]):
                d = t.leaving(v, p_out)
                yield (p_out, "a", d) if d is not None else (p_out, "b", t.lead_index(v, p_out))

        aa, ab, ba, bb = [], [], [], []
        for d_in in range(self.n):                      # wave arriving at (v, p_in) on d_in
            v, p_in = int(t.dst_vertex[d_in]), int(t.dst_port[d_in])
            for p_out, kind, row in outputs(v):
                (aa if kind == "a" else ba).append((row, d_in, v, p_out, p_in))
        for l_in, (v, p_in) in enumerate(t.leads):      # wave injected by lead l_in
            for p_out, kind, row in outputs(v):
                (ab if kind == "a" else bb).append((row, l_in, v, p_out, p_in))
        self._entries = {"aa": (aa, (self.n, self.n)), "ab": (ab, (self.n, self.n_leads)),
                         "ba": (ba, (self.n_leads, self.n)), "bb": (bb, (self.n_leads, self.n_leads))}
        self._const_blocks = {}

    def sector(self, directed):
        """The model restricted to the directed bonds ``directed`` (U_s =
        U[directed][:, directed]); the model itself if they are all the bonds."""
        directed = np.asarray(directed, dtype=int)
        if np.array_equal(directed, np.arange(self.n)):
            return self
        return _SectorModel(self, directed)

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

    def singular_values(self, k):
        """Singular values of I − U(k), in decreasing order."""
        return np.linalg.svd(np.eye(self.n) - self.U(k), compute_uv=False)

    def nullity(self, k, tol=NULLITY_TOL):
        """Number of singular values of I − U(k) at or below ``tol``."""
        return int(np.sum(self.singular_values(k) <= tol))

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

    def log_det_many(self, ks, derivative=False, chunk=2048):
        """log det(I − U(k)) = log|det| + i arg det (arg in (−π, π]) for an
        array of (real or complex) k; −inf where det = 0.

        With ``derivative``, returns an array of shape (m, 2) whose second
        column is d/dk log det(I − U) = −tr((I − U)⁻¹ U'), U' = i n_g L U with
        n_g = d(k n_eff)/dk at Re k (central differences); the k-dependence of
        the vertex matrices is neglected (inf where I − U is singular)."""
        ks = np.atleast_1d(np.asarray(ks, dtype=complex))
        eye = np.eye(self.n)

        def work(kc):
            U = self._U_many(kc)
            sign, logabs = np.linalg.slogdet(eye - U)
            out = logabs + 1j * np.angle(sign)
            if not derivative:
                return out
            d = np.full(kc.size, np.inf, dtype=complex)
            ok = sign != 0
            try:
                X = np.linalg.solve(eye - U[ok], U[ok])
                d[ok] = -1j * self._phase_rate(np.real(kc[ok])) * (
                    np.einsum("mdd,d->m", X, self.lengths))
            except np.linalg.LinAlgError:
                pass
            return np.stack([out, d], axis=-1)

        return _map_chunks(work, ks, chunk)

    def _phase_rate(self, k):
        """n_g(k) = d(k n_eff)/dk at real k (exact for a constant index)."""
        k = np.asarray(k, dtype=float)
        if not callable(self.neff):
            return np.full(k.shape, float(self.neff))
        h = 1e-6 * np.maximum(np.abs(k), 1.0)
        return ((k + h) * index_function(self.neff, k + h)
                - (k - h) * index_function(self.neff, k - h)) / (2 * h)

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


class _SectorModel(ScatteringModel):
    """U restricted to the directed bonds of one sector (bond block only)."""

    def __init__(self, parent, directed):
        self._parent = parent
        self._index = directed
        self.n = directed.size
        self.n_leads = 0
        self.lengths = parent.lengths[directed]
        self.neff = parent.neff
        self.constant = parent.constant
        self._const_blocks = {}

    def block(self, name, k=None):
        if name != "aa":
            raise ValueError("A sector model only has the bond block 'aa'.")
        if self.constant and name in self._const_blocks:
            return self._const_blocks[name]
        M = self._parent.block("aa", k)[np.ix_(self._index, self._index)]
        if self.constant:
            M.setflags(write=False)
            self._const_blocks[name] = M
        return M
