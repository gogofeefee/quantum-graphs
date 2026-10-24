"""Shared foundations: topology, vertex matrices, scattering model, persistence."""

from .config import get_option, set_options, options
from .topology import build_topology, canonical_edge, Topology
from .vertices import (PASS_THROUGH, neumann_matrix, directional_coupler_matrix,
                       couplings_for_vertices)
from .scattering import ScatteringModel, index_function
from .optics import group_index_function, wavelength_to_k, k_to_wavelength

__all__ = ["get_option", "set_options", "options", "build_topology", "canonical_edge",
           "Topology", "PASS_THROUGH", "neumann_matrix", "directional_coupler_matrix",
           "couplings_for_vertices", "ScatteringModel", "index_function",
           "group_index_function", "wavelength_to_k", "k_to_wavelength"]
