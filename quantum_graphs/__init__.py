"""
quantum_graphs — quantum graphs of photonic waveguide networks.

One graph, four points of view::

    import quantum_graphs as qg

    layout = qg.layouts.flower_graph()                 # vertex_positions, port_map, lengths
    closed = qg.ClosedDirectionalGraphs(**layout, vertex_couplings=0.5, neff=3.97,
                                        wavelength_range=(1.50, 1.53))
    spectrum, degeneracies = closed.compute_spectrum(N_scan=150_000)

Subpackages
-----------
closed_graphs     spectra and eigenvectors of closed graphs, spectral statistics
open_graphs       S-matrix, resonances and transmission of graphs with leads
classical_graphs  Perron-Frobenius (ray) dynamics
periodic_orbits   periodic-orbit enumeration for directional-coupler graphs
stats             unfolding, NNSD, Σ², Δ₃ and random-matrix reference curves
layouts           ready-made geometries and graph transformations
core              topology, vertex matrices, bond scattering model, caches, options

Options: ``set_options(verbose=False, show=False)`` (or the ``options``
context manager) silences messages and ``plt.show()``.
"""

__version__ = "1.0.0.dev0"

from .core.config import get_option, options, set_options
from .core.vertices import directional_coupler_matrix, neumann_matrix, PASS_THROUGH
from . import layouts, stats
from .closed_graphs import ClosedGraphs
from .closed_graphs import DirectionalGraphs as ClosedDirectionalGraphs
from .open_graphs import OpenGraphs
from .open_graphs import DirectionalGraphs as OpenDirectionalGraphs
from .classical_graphs import ClassicalGraphs, DirectionalClassicalGraphs
from .periodic_orbits import PeriodicOrbits
from .stats import nnsd, sigma2_rmt, delta3_rmt

__all__ = [
    "__version__", "get_option", "options", "set_options",
    "directional_coupler_matrix", "neumann_matrix", "PASS_THROUGH",
    "layouts", "stats",
    "ClosedGraphs", "ClosedDirectionalGraphs", "OpenGraphs", "OpenDirectionalGraphs",
    "ClassicalGraphs", "DirectionalClassicalGraphs", "PeriodicOrbits",
    "nnsd", "sigma2_rmt", "delta3_rmt",
]
