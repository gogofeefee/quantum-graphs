"""
Sectors of the bond dynamics: independent blocks of U(k) and their partners.

A wave on directed bond ``d`` can only scatter into the directed bonds ``d'``
with ``Σ_αα[d', d] ≠ 0``. The strongly connected components of this
transition pattern are the **sectors**: U(k) is block triangular in them
(block diagonal when U is unitary, i.e. for a closed lossless graph), so

    det(I − U(k)) = Π_s det(I − U_s(k)),   U_s = U restricted to sector s,

and the spectrum (or the resonances) is the union of the sector spectra.
A directed bond that cannot return to itself carries no level and belongs
to no sector (open graphs only).

Reversed sectors
----------------
The reverse of sector ``s`` is the set of reversed bonds ``d + B``. When the
vertex matrices restricted to the sector are symmetric (σ_v[p, q] =
σ_v[q, p], reciprocity), ``U`` on the reversed bonds is the transpose of
``U_s`` up to a permutation and a similarity by D(k), so the two sectors have
the same spectrum:

* ``reverse(s) == s``: a self-reverse sector with time-reversal symmetry
  (GOE statistics expected for a chaotic graph);
* ``reverse(s) != s``: two partner sectors, every level is doubly degenerate
  and each sector alone has no time-reversal symmetry (GUE expected). The
  double ring and every ring of directional couplers are of this type.

Without reciprocity the sectors are independent (GUE expected).

Weyl law
--------
Each sector has a mean counting function N_s(k) = (L_s / 2π) ∫ n_g dk with
L_s the sum of the lengths of its directed bonds (2 L for a graph that is a
single sector, L for each of two partner sectors).

The pattern is sampled at a few wavenumbers so that a k-dependent vertex
matrix that vanishes at isolated k does not split a sector.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

#: |Σ_αα| entries below this value (relative to the largest) are treated as zero
PATTERN_TOL = 1e-12
#: tolerance of the reciprocity test Σ_αα[r(d), r(d')] == Σ_αα[d', d]
RECIPROCITY_TOL = 1e-12


@dataclass(frozen=True)
class Sector:
    """One sector of the bond dynamics (see the module docstring).

    Attributes
    ----------
    index : int
        Position in :class:`Sectors`.
    directed : ndarray of int
        Its directed bonds, sorted.
    length : float
        Sum of their lengths L_s (µm): N_s(k) = (L_s / 2π) ∫ n_g dk.
    reverse : int or None
        Sector made of the reversed bonds (``index`` for a self-reverse
        sector; None if the reversed bonds do not form a sector).
    representative : int
        Sector whose spectrum this one shares: ``min(index, reverse)`` for
        reciprocal partner sectors, ``index`` otherwise.
    time_reversal : bool
        Self-reverse with symmetric vertex matrices.
    """
    index: int
    directed: np.ndarray
    length: float
    reverse: object
    representative: int
    time_reversal: bool

    @property
    def ensemble(self):
        """Random-matrix ensemble expected for a chaotic sector: "GOE" or "GUE"."""
        return "GOE" if self.time_reversal else "GUE"

    def __repr__(self):
        kind = ("self-reverse" if self.reverse == self.index
                else f"partner of {self.reverse}" if self.reverse is not None else "no partner")
        return (f"Sector({self.index}: {self.directed.size} directed bonds, "
                f"L_s = {self.length:.3f} µm, {kind}, {self.ensemble})")


class Sectors(tuple):
    """The sectors of a graph (a tuple of :class:`Sector`) and the independent
    spectra they carry."""

    @property
    def independent(self):
        """The representative sectors: one per distinct spectrum."""
        return tuple(s for s in self if s.representative == s.index)

    def multiplicity(self, index):
        """Number of sectors that share the spectrum of sector ``index``
        (2 for reciprocal partner sectors, 1 otherwise)."""
        rep = self[index].representative
        return sum(s.representative == rep for s in self)

    def class_of(self, index):
        """Representative of sector ``index``."""
        return self[index].representative

    def __repr__(self):
        return "Sectors(\n" + "".join(f"  {s!r},\n" for s in self) + ")"


def _sample_blocks(model, k_samples):
    """Σ_αα at the sampled wavenumbers (once if the vertex matrices are constant)."""
    ks = list(k_samples)
    if model.constant or not ks:
        ks = ks[:1] or [None]
    return [model.block("aa", k) for k in ks]


def transition_pattern(blocks):
    """Boolean matrix P[d', d] = (Σ_αα[d', d] ≠ 0 in one of ``blocks``)."""
    pattern = np.zeros(blocks[0].shape, dtype=bool)
    for S in blocks:
        A = np.abs(S)
        if A.size:
            pattern |= A > PATTERN_TOL * max(float(A.max()), np.finfo(float).tiny)
    return pattern


def find_sectors(topology, model, k_samples):
    """Sectors of a graph (see the module docstring).

    Parameters
    ----------
    topology : :class:`~quantum_graphs.core.topology.Topology`
    model : :class:`~quantum_graphs.core.scattering.ScatteringModel`
    k_samples : list of float
        Wavenumbers at which the transition pattern and the reciprocity are
        sampled (one is enough for constant vertex matrices).
    """
    blocks = _sample_blocks(model, k_samples)
    pattern = transition_pattern(blocks)
    n_comp, label = connected_components(csr_matrix(pattern), directed=True,
                                         connection="strong")
    groups = [np.flatnonzero(label == c) for c in range(n_comp)]
    # a singleton without a self-transition carries no level
    groups = [g for g in groups if g.size > 1 or pattern[g[0], g[0]]]
    groups.sort(key=lambda g: g[0])
    index_of = {int(d): i for i, g in enumerate(groups) for d in g}

    sectors = []
    for i, g in enumerate(groups):
        rev = topology.reverse(g)
        r = index_of.get(int(rev[0]))
        if r is not None and not np.array_equal(np.sort(rev), groups[r]):
            r = None
        reciprocal = r is not None and all(
            np.allclose(S[np.ix_(rev, rev)], S[np.ix_(g, g)].T, rtol=0, atol=RECIPROCITY_TOL)
            for S in blocks)
        sectors.append(Sector(index=i, directed=_readonly(g),
                              length=float(topology.directed_length[g].sum()),
                              reverse=r, representative=min(i, r) if reciprocal else i,
                              time_reversal=reciprocal and r == i))
    return Sectors(sectors)


def _readonly(a):
    a = np.asarray(a, dtype=int)
    a.setflags(write=False)
    return a
