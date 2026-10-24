"""
CoreMixin — construction of a closed graph and its parameter report.

Topology handling (validation, virtual vertices, directed bonds) is shared with
the other subpackages through :class:`quantum_graphs.core.base.GraphBase`.
"""

import datetime
import inspect
import os

import numpy as np

from ..core.base import GraphBase
from ..core.optics import group_index_function
from ..core.persistence import legacy_config_hash
from ..core.vertices import resolve_vertex_matrices


class CoreMixin(GraphBase):
    """Graph construction, save path and parameter report of a closed graph."""

    def __init__(self, vertex_positions, port_map,
                 lengths=None,
                 vertex_matrices=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 name="MyClosedGraph",
                 save_path=None,
                 base_dir=None):
        """
        Initialize a closed quantum graph (no external leads).

        Raises ValueError for an inconsistent ``port_map``, missing / unknown
        bond lengths, or vertex matrices whose size differs from the valency.
        """
        self.wavelength_range = wavelength_range
        self.neff = neff
        self.ng = ng if ng is not None else self._compute_group_index()

        self._setup_topology(vertex_positions, port_map, lengths, virtual_vertex_spacing)
        if vertex_matrices is None:
            raise ValueError("vertex_matrices is required (dict, list, array or sigma(k)).")
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
        info_str = f"Nv={self.n_vertices_total}_Nb={self.n_internal_bonds}"
        if callable(self.neff):
            neff_str = getattr(self.neff, "__name__", "neff_func")
        else:
            neff_str = f"neff={float(self.neff):.5f}"
        lamb_min, lamb_max = self.wavelength_range
        lamb_str = f"lambda={lamb_min:.3f}-{lamb_max:.3f}"
        hash_digest = legacy_config_hash(self.port_map, self.lengths)
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
            f.write(f"Total vertices: {self.n_vertices_total} "
                    f"(virtual: {self.n_virtual_vertices})\n")
            f.write(f"Internal bonds: {self.n_internal_bonds}\n")
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
                    f.write(f"Vertex {v} (valency {self.valencies[v]}):\n")
                    np.savetxt(f, np.asarray(sigma), fmt="%-8.4f", delimiter=" ", newline="\n")
                    f.write("\n")

            f.write("\n### Port Map (raw, before virtualization) ###\n")
            for i, ports in enumerate(self.port_map):
                f.write(f"Vertex {i} ports: {ports}\n")

            f.write("\n### Bond Lengths (raw, before virtualization) ###\n")
            for (a, b), L in self.lengths.items():
                f.write(f"{a} — {b} : {L:.6f} µm\n")

            f.write("\n### Virtualization Info ###\n")
            f.write(f"Virtual vertices: {self.virtual_vertices}\n")
            f.write(f"Updated bonds (after virtualization): {self.updated_bonds}\n")

        self._log(f"Graph parameters saved to {filepath}")


def _source_or_placeholder(func):
    try:
        return inspect.getsource(func) + "\n"
    except (OSError, TypeError):
        return "(source code unavailable)\n"
