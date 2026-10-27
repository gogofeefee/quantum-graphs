"""
Localisation of a wavefunction on the bonds of a graph, as pure functions.

* :func:`eigenspace_intensity` — intensity per directed bond of one
  wavefunction, or of a degenerate eigenspace (basis-independent);
* :func:`physical_bond_intensity` — intensity per physical bond;
* :func:`localization_measures` — Shannon entropy and inverse participation
  ratio (IPR) of the bond distribution.

Conventions (docs/scientific_notes.md §5):

* The distribution is p_b = I_b / Σ I_b over the N_b **physical** bonds (the
  bonds of ``port_map``), with I_b = |a_{u→v}|² + |a_{v→u}|² the intensity of
  the two waves arriving at the real end vertices of bond b = (u, v). Virtual
  vertices do not change the result: a split bond counts once, with the
  amplitudes it would carry if it had not been split. This is an intensity,
  not an energy (which would be ∝ L_b I_b).
* A degenerate level of multiplicity g has no preferred eigenvector: its
  intensity is the diagonal of the projector onto the eigenspace divided by g,
  (1/g) Σ_m |a_b^(m)|² over any orthonormal basis, i.e. the maximally mixed
  state of the level. For g = 1 this is |a_b|². Because the entropy is
  concave, the entropy of this average is ≥ the average entropy of the basis
  vectors: degenerate levels look at least as delocalised as any basis
  choice would make them.
"""

import numpy as np


def eigenspace_intensity(vecs):
    """Intensity per directed bond, (1/g) Σ_m |a^(m)|².

    ``vecs`` is one vector (shape ``(n,)``) or the g orthonormal rows of a
    basis of an eigenspace (shape ``(g, n)``, as returned by
    ``compute_wavefunction``). The result does not depend on the choice of
    that basis.
    """
    vecs = np.atleast_2d(np.asarray(vecs))
    return np.sum(np.abs(vecs) ** 2, axis=0) / vecs.shape[0]


def physical_bond_intensity(directed_intensity, arrivals):
    """Intensity per physical bond, I_b = I[d_fwd] + I[d_bwd].

    ``arrivals`` is the graph's ``physical_bond_arrivals``: for each physical
    bond, the two directed bonds arriving at its real end vertices.
    """
    intensity = np.asarray(directed_intensity, dtype=float)
    arrivals = np.asarray(arrivals, dtype=int)
    return intensity[arrivals[:, 0]] + intensity[arrivals[:, 1]]


def localization_measures(bond_intensity):
    """Shannon entropy and IPR of p_b = I_b / Σ I_b.

    Returns
    -------
    S : float
        -Σ p_b ln p_b, in [0, ln N_b].
    S_norm : float
        S / ln N_b, in [0, 1]: 1 for a uniform distribution, 0 on one bond.
    IPR : float
        N_b Σ p_b², in [1, N_b]: 1 for a uniform distribution, N_b on one
        bond. Σ p_b² (in [1/N_b, 1]) is IPR / N_b.

    S_norm is NaN for a single bond, and all three are NaN for a zero
    distribution.
    """
    intensity = np.asarray(bond_intensity, dtype=float)
    n_b = intensity.size
    total = float(np.sum(intensity))
    if n_b == 0 or total <= 0:
        return np.nan, np.nan, np.nan
    p = intensity / total
    m = p > 0
    S = float(0.0 - np.sum(p[m] * np.log(p[m])))     # 0.0, not -0.0, on one bond
    S_norm = S / np.log(n_b) if n_b > 1 else np.nan
    return S, S_norm, float(n_b * np.sum(p ** 2))
