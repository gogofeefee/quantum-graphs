#!/usr/bin/env python3
"""
Template for a new study — copy this file into your own folder and edit it.

1. Describe the graph once (section 1): vertex positions (drawing only),
   port connections and bond lengths in µm.
2. Choose the physics (section 2): effective index, couplers, window.
3. Run any of the analyses below; every expensive result is cached next to
   this script (``<name>/<parameters>/``) together with a ``.meta.json``
   record of the parameters, and recomputed automatically if they change.

Requires ``pip install -e /path/to/quantum-graphs`` (or put the repository
root on ``sys.path``).
"""

import numpy as np

import quantum_graphs as qg

# qg.set_options(show=False)          # no plt.show() (batch runs)
# qg.set_options(verbose=False)       # no progress messages

# ============================================================================
#  1. Graph
# ============================================================================
#  port_map[i][p] = (j, q): port p of vertex i <-> port q of vertex j (both ways).
#  Directional couplers: ports (0, 1) = one arm, (2, 3) = the other arm.
#  None = external lead (open graphs only). Lengths: one entry per bond, µm.
#  Multiple bonds and self-loops are allowed.

layout = dict(
    vertex_positions=[(0.0, 0.0), (100.0, 0.0), (50.0, 80.0)],
    port_map=[
        [(1, 0), (2, 0), (1, 2), (2, 2)],      # vertex 0
        [(0, 0), (2, 1), (0, 2), (2, 3)],      # vertex 1
        [(0, 1), (1, 1), (0, 3), (1, 3)],      # vertex 2
    ],
    lengths={((0, 0), (1, 0)): 103.0, ((0, 1), (2, 0)): 131.0, ((0, 2), (1, 2)): 187.0,
             ((0, 3), (2, 2)): 211.0, ((1, 1), (2, 1)): 97.0, ((1, 3), (2, 3)): 163.0},
)
# Ready-made alternatives: qg.layouts.flower_graph(), qg.layouts.bow_tie_graph(open=True), ...

# ============================================================================
#  2. Physics
# ============================================================================

physics = dict(
    neff=3.97,                       # float, or a function neff(k); see qg.dispersion
    wavelength_range=(1.50, 1.56),   # vacuum wavelength window (µm)
)
couplings = 0.5                      # float, list, dict {vertex: C} or function C(k)

# ============================================================================
#  3. Closed graph: spectrum and statistics
# ============================================================================

if True:
    g = qg.ClosedDirectionalGraphs(**layout, vertex_couplings=couplings, **physics,
                                   name="my_graph_closed")
    g.plot_graph(show_lengths=True, show_port_markers=True, save_fig=True)
    spectrum, degeneracies = g.compute_spectrum(N_scan=200_000)
    print(f"{spectrum.size} levels, Weyl estimate {g.weyl_count():.1f}")
    g.plot_wavefunction(a_in=g.compute_wavefunction(spectrum[0]), k=spectrum[0], save_fig=True)
    g.short_range_correlations(plot_wigner=["goe", "gue", "poissonian"], save_fig=True)
    g.long_range_correlations(L_max=5, L_step=0.1, plot_wigner=["goe", "poissonian"],
                              save_fig=True)

# ============================================================================
#  4. Open graph: make vertex 0 an access coupler with two leads
# ============================================================================

if False:
    open_layout = dict(layout)
    open_layout["port_map"] = [list(ports) for ports in layout["port_map"]]
    # ... replace two connections by None (and remove their lengths) ...
    o = qg.OpenDirectionalGraphs(**open_layout, internal_coupling=0.5, external_coupling=0.3,
                                 **physics, name="my_graph_open")
    resonances, _ = o.find_resonances(N_re=200_000, N_im=15, im_min=-1e-4, im_max=0.0)
    o.plot_S_amplitude(*physics["wavelength_range"], n_points=200_000, components=(1, 0),
                       spectrum=o.best_spectrum, save=True)

# ============================================================================
#  5. Classical dynamics and periodic orbits on the same graph
# ============================================================================

if True:
    c = qg.DirectionalClassicalGraphs(**layout, vertex_couplings=couplings, name="my_graph_cl")
    print("spectral gap:", c.spectral_gap(), " mixing:", c.is_mixing())

    po = qg.PeriodicOrbits(**layout, ng=physics["neff"], name="my_graph_orbits")
    orbits = po.find_orbits_up_to_length(1000.0)
    po.export_orbits_csv(which="all")
    print(f"{len(orbits)} periodic orbits up to 1000 µm")

# ============================================================================
#  6. Your own analysis: the building blocks
# ============================================================================
#  g.U(k)                      quantum map U(k) = D(k) Σ (2B × 2B)
#  g.sigma_min(k)              σ_min(I − U(k)) for one k or an array
#  g.model.S_many(ks)          (open graphs) S(k) for many k
#  qg.stats.unfold(...), qg.stats.number_variance(...), qg.stats.spectral_rigidity(...)
#  qg.nnsd(s, "GOE"), qg.sigma2_rmt(L, "GUE"), qg.delta3_rmt(L, "GOE")

print("Done.")
