"""
CoreMixin — construction of a closed graph and its parameter report.

Topology handling (validation, directed bonds) is shared with the other
subpackages through :class:`quantum_graphs.core.base.GraphBase`.
"""

from ..core.base import GraphBase
from ..core.optics import group_index_function
from ..core.persistence import topology_hash
from ..core.vertices import resolve_vertex_matrices


class CoreMixin(GraphBase):
    """Graph construction, save path and parameter report of a closed graph."""

    _parameters_title = "Closed Quantum Graph Parameters"

    def __init__(self, vertex_positions, bonds, *,
                 vertex_matrices,
                 wavelength_range,
                 neff=1.0,
                 ng=None,
                 bond_spacing=None,
                 name="MyClosedGraph",
                 save_path=None,
                 base_dir=None):
        """
        Initialize a closed quantum graph (no external leads).

        ``bonds``: ``[(i, p, j, q[, L]), ...]`` (see :mod:`quantum_graphs.core.topology`);
        ``bond_spacing``: drawing of parallel bonds (see
        :class:`quantum_graphs.plotting.drawing.BondPaths`).

        Raises ValueError for inconsistent bonds (a port used twice or left
        unconnected, a non-positive length) or vertex matrices whose size
        differs from the valency.
        """
        self.wavelength_range = self._check_wavelength_range(wavelength_range)
        self.neff = neff
        self.ng = ng if ng is not None else self._compute_group_index()

        self._setup_topology(vertex_positions, bonds)
        self.bond_spacing = bond_spacing
        self._setup_vertex_matrices(resolve_vertex_matrices(vertex_matrices, self.n_vertices),
                                    self._k_samples())

        leaf = self._build_save_dir_name(getattr(self, "coupling_info", None))
        self._setup_save_path(name, leaf, save_path, base_dir)

    # =========================================================================
    # Group index
    # =========================================================================

    def _compute_group_index(self):
        """ng(k) = neff(k) + k dneff/dk (float if neff is constant)."""
        return group_index_function(self.neff)

    # =========================================================================
    # Save-path helpers
    # =========================================================================

    def _build_save_dir_name(self, coupling_info=None):
        """Deterministic save directory leaf name (historical format)."""
        info_str = f"Nv={self.n_vertices}_Nb={self.n_bonds}"
        if callable(self.neff):
            neff_str = getattr(self.neff, "__name__", "neff_func")
        else:
            neff_str = f"neff={float(self.neff):.5f}"
        lamb_min, lamb_max = self.wavelength_range
        lamb_str = f"lambda={lamb_min:.3f}-{lamb_max:.3f}"
        hash_digest = topology_hash(self.topology)
        if coupling_info:
            return f"{info_str}_{neff_str}_{lamb_str}_{coupling_info}_{hash_digest}"
        return f"{info_str}_{neff_str}_{lamb_str}_{hash_digest}"
