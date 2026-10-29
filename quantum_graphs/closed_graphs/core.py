"""
CoreMixin — construction of a closed graph and its parameter report.

Topology handling (validation, directed bonds) is shared with the other
subpackages through :class:`quantum_graphs.core.base.GraphBase`.
"""

import datetime
import inspect
import os

import numpy as np

from ..core.base import GraphBase
from ..core.optics import group_index_function
from ..core.persistence import topology_hash
from ..core.vertices import resolve_vertex_matrices


class CoreMixin(GraphBase):
    """Graph construction, save path and parameter report of a closed graph."""

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

    def _save_parameters(self):
        """Write a human-readable description of the graph to graph_parameters.txt."""
        filepath = os.path.join(self.save_path, "graph_parameters.txt")
        with open(filepath, "w") as f:
            f.write("### Closed Quantum Graph Parameters ###\n")
            f.write(f"Graph Name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n")
            f.write(f"Physics fingerprint: {self.fingerprint}\n")
            f.write(f"Vertices: {self.n_vertices}\n")
            f.write(f"Bonds: {self.n_bonds}\n")
            f.write(f"Total length: {self.total_length:.6f} µm\n")
            f.write(f"Wavelength range (vacuum): {self.wavelength_range[0]:.3f} – "
                    f"{self.wavelength_range[1]:.3f} µm\n\n")

            if callable(self.neff):
                f.write("Effective index: function of k\n")
                f.write(_source_or_placeholder(self.neff))
            else:
                f.write(f"Effective index: {float(self.neff):.4f}\n")

            if callable(self.ng):
                f.write("\nGroup index: function of k\n")
            else:
                f.write(f"\nGroup index: {float(self.ng):.4f}\n")

            f.write("\n### Vertex Scattering Matrices ###\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v}: function of k\n")
                    f.write(_source_or_placeholder(getattr(sigma, "coupling", sigma)))
                else:
                    f.write(f"Vertex {v} (valency {self.topology.valency[v]}):\n")
                    np.savetxt(f, np.asarray(sigma), fmt="%-8.4f", delimiter=" ", newline="\n")
                    f.write("\n")

            f.write("\n### Vertex positions (µm) ###\n")
            for v, (x, y) in enumerate(self.topology.positions):
                f.write(f"Vertex {v}: ({x:.6f}, {y:.6f})\n")

            f.write("\n### Bonds (i, p, j, q, L in µm) ###\n")
            for b, (i, p, j, q, L) in enumerate(self.topology.bonds):
                f.write(f"{b}: ({i}, {p}) — ({j}, {q}) : {L:.6f}\n")

        self._log(f"Graph parameters saved to {filepath}")


def _source_or_placeholder(func):
    try:
        return inspect.getsource(func) + "\n"
    except (OSError, TypeError):
        return "(source code unavailable)\n"
