"""
ScatteringMixin — S(k) for open graphs.

All matrices come from the shared :class:`quantum_graphs.core.scattering.ScatteringModel`
(``self.model``); see its docstring for the conventions. ``S[j, i]`` is the
amplitude leaving lead j for a unit amplitude injected in lead i, leads being
numbered in the order of ``topology.leads``. U(k), σ_min(k) and the secular
determinant are common to every graph (``GraphBase``).
"""

import numpy as np


class ScatteringMixin:
    """S(k) of an open graph."""

    def S(self, k):
        """Scattering matrix S(k): shape (n_leads, n_leads) for one k, (m, n_leads,
        n_leads) for an array of m wavenumbers; k may be complex."""
        if np.ndim(k) == 0:
            return self.model.S(k)
        return self.model.S_many(np.asarray(k))
