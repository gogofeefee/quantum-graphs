"""
ScatteringMixin — Σ blocks, S(k), U(k) and σ_min for open graphs.

All matrices come from the shared :class:`quantum_graphs.core.scattering.ScatteringModel`
(``self.model``); see its docstring for the conventions. ``S[j, i]`` is the
amplitude leaving lead j for a unit amplitude injected in lead i, leads being
numbered in the order of the ``None`` entries of ``port_map`` (``self.leads``).
"""

import numpy as np


class ScatteringMixin:

    def _build_propagation_matrix(self, k):
        """Diagonal propagation matrix D(k) for all directed bonds."""
        return self.model.D(k)

    def _build_sigma_alpha_alpha(self, k):
        """Internal scattering block Σ_αα."""
        return np.array(self.model.block("aa", k))

    def _build_sigma_alpha_betahat(self, k):
        """Lead -> internal block Σ_αβ."""
        return np.array(self.model.block("ab", k))

    def _build_sigma_beta_alpha(self, k):
        """Internal -> lead block Σ_βα."""
        return np.array(self.model.block("ba", k))

    def _build_sigma_beta_betahat(self, k):
        """Direct lead -> lead block Σ_ββ."""
        return np.array(self.model.block("bb", k))

    def compute_S_matrix(self, k):
        """Scattering matrix S(k), shape (n_leads, n_leads); k may be complex."""
        return self.model.S(k)

    def compute_S_matrices(self, ks):
        """S(k) for an array of wavenumbers, shape (len(ks), n_leads, n_leads)."""
        return self.model.S_many(ks)

    def _U(self, k):
        """U(k) = D(k) Σ_αα(k), shape (2B, 2B)."""
        return self.model.U(complex(k))

    def _sigma_min(self, k):
        """g(k) = σ_min(I − U(k)) (inf if the SVD fails)."""
        return self.model.sigma_min(complex(k))
