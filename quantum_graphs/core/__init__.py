"""Shared foundations: topology, vertex matrices, scattering model, sectors, persistence."""

from .config import get_option, set_options, options
from .topology import build_topology, Topology
from .vertices import (PASS_THROUGH, neumann_matrix, directional_coupler_matrix,
                       couplings_for_vertices)
from .scattering import NULLITY_TOL, ScatteringModel, index_function
from .sectors import Sector, Sectors, find_sectors
from .optics import group_index_function, wavelength_to_k, k_to_wavelength

__all__ = ["get_option", "set_options", "options", "build_topology",
           "Topology", "PASS_THROUGH", "neumann_matrix", "directional_coupler_matrix",
           "couplings_for_vertices", "NULLITY_TOL", "ScatteringModel", "index_function",
           "Sector", "Sectors", "find_sectors",
           "group_index_function", "wavelength_to_k", "k_to_wavelength"]
