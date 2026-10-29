#!/usr/bin/env python3
"""
Generate the numerical reference (non-regression) data set.

Usage
-----
    python tests/reference/generate_reference.py OUT.npz

Computes a broad set of physical quantities with ``quantum_graphs`` and stores
them in one ``.npz`` file. ``tests/reference/compare_reference.py`` compares
two such files with physically meaningful tolerances.

``reference_v0.npz`` was produced with the code of commit 83ca561 (the state
published before the 2026-10 refactor), ``reference_v1``-``v3`` with the v1 API
(``port_map`` + virtual vertices, tag v1.0.0), ``reference_v4`` with
the bond-list API (no virtual vertices), ``reference_v5`` and later with the
sector analysis (statistics on the desymmetrised spectrum).

Only quantities that do not depend on internal ordering conventions are
stored as raw arrays (spectra, S-matrix entries, eigenvalues, sorted bond
lengths...). Quantities indexed by directed bonds are stored together with
their bond labels.
"""

import os
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometries as G                                         # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
plt.show = lambda *a, **k: None

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from quantum_graphs.closed_graphs import ClosedGraphs, DirectionalGraphs as ClosedDG  # noqa: E402
from quantum_graphs.open_graphs import DirectionalGraphs as OpenDG                    # noqa: E402
from quantum_graphs.classical_graphs import ClassicalGraphs                          # noqa: E402
from quantum_graphs.periodic_orbits import PeriodicOrbits                             # noqa: E402
import quantum_graphs                                                                 # noqa: E402
quantum_graphs.set_options(verbose=False, show=False)


OUT = {}
TMP = tempfile.mkdtemp(prefix="qg_reference_")


def tmp(name):
    return os.path.join(TMP, name)


def store(prefix, **arrays):
    for key, val in arrays.items():
        OUT[f"{prefix}/{key}"] = np.asarray(val)


def bond_labels(graph):
    """Directed-bond labels as an (n, 4) int array: src vertex, src port, dst vertex, dst port."""
    t = graph.topology
    return np.stack([t.src_vertex, t.src_port, t.dst_vertex, t.dst_port], axis=1).astype(int)


def sectors(prefix, graph):
    """Sector structure (reference_v5 and later): lengths, partners, ensembles."""
    secs = graph.sectors
    store(prefix, sector_lengths=np.sort([s.length for s in secs]),
          sector_multiplicities=np.sort([secs.multiplicity(s.index) for s in secs]),
          sector_n_goe=sum(s.ensemble == "GOE" for s in secs))


def topology(prefix, graph):
    t = graph.topology
    store(prefix,
          n_bonds=t.n_bonds,
          total_length=graph.total_length,
          sorted_bond_lengths=np.sort(t.length))


# ---------------------------------------------------------------------------
# Closed graphs
# ---------------------------------------------------------------------------

def closed_case(prefix, graph, k_probe, n_scan, wf_levels=(0,), stats=True):
    topology(prefix, graph)
    U = [graph.U(k) for k in k_probe]
    # eigenvalues of U are basis-independent
    store(prefix, U_eigvals=np.array([np.sort_complex(np.linalg.eigvals(u)) for u in U]),
          secular_det=np.array([graph.secular_det(k) for k in k_probe]))
    spectrum, deg = graph.compute_spectrum(N_scan=n_scan, save_spectrum=False)
    store(prefix, spectrum=spectrum, degeneracies=deg)
    labels = bond_labels(graph)
    for i in wf_levels:
        a = graph.compute_wavefunction(spectrum[i])
        U_k = graph.U(spectrum[i])
        resid = max(np.linalg.norm(v - U_k @ v) for v in a)
        # intensity per directed bond, keyed by bond label: diagonal of the projector
        # onto the eigenspace (sum over its orthonormal basis), basis-independent
        # also for degenerate levels (reference_v1 stored |a[0]|^2 only)
        store(f"{prefix}/wf{i}", intensity=np.sum(np.abs(a) ** 2, axis=0), labels=labels,
              residual=resid, nmodes=a.shape[0])
        if hasattr(graph, "localization_measures"):        # reference_v3 and later
            S, S_norm, ipr = graph.localization_measures(a)
            store(f"{prefix}/wf{i}", loc_entropy=S, loc_S_norm=S_norm, loc_ipr=ipr)
    if hasattr(graph, "localization_measures"):
        # entropy / IPR of every level (per bond, eigenspace projector)
        _, _, S_norm, ipr = graph._compute_mode_quantities()
        store(prefix, loc_S_norm_levels=S_norm, loc_ipr_levels=ipr)
    sectors(prefix, graph)
    if stats and spectrum.size > 20:
        # desymmetrised spectrum (one sector per pair of partners), reference_v5 and later
        unfolded = graph.unfolded_spectrum()
        Ls, s2 = graph.sigma2(5.0, 0.25)
        Ld, d3 = graph.delta3(5.0, 0.25)
        store(prefix, spacings=np.diff(unfolded), sigma2=s2, sigma2_L=Ls,
              delta3=d3, delta3_L=Ld,
              weyl=graph.weyl_staircase(np.array(k_probe), sector="all"))


def run_closed():
    k_probe = [4.0, 4.05, 4.1003]

    g = ClosedGraphs(**G.SINGLE_RING, vertex_matrices={0: G.neumann(2)}, neff=2.41,
                     wavelength_range=(1.50, 1.60), name="ref_single_ring",
                     save_path=tmp("single"))
    closed_case("closed/single_ring", g, k_probe, 20_000)

    g = ClosedDG(**G.DOUBLE_RING, couplings=0.5, neff=1.0,
                 wavelength_range=(1.48, 1.64), name="ref_double_ring", save_path=tmp("dr"))
    closed_case("closed/double_ring", g, k_probe, 60_000)

    g = ClosedDG(**G.DOUBLE_RING, couplings=G.coupling_dispersive, neff=G.neff_dispersive,
                 wavelength_range=(1.48, 1.64), name="ref_double_ring_disp", save_path=tmp("drd"))
    closed_case("closed/double_ring_dispersive", g, k_probe, 60_000)
    store("closed/double_ring_dispersive", ng=np.array([g.ng(k) for k in k_probe]))

    g = ClosedGraphs(**G.TETRAHEDRON, vertex_matrices={v: G.neumann(3) for v in range(4)},
                     neff=2.41, wavelength_range=(1.50, 1.56), name="ref_tetra", save_path=tmp("tet"))
    closed_case("closed/tetrahedron", g, k_probe, 60_000)

    for name, geom in (("flower", G.FG_CLOSED), ("bowtie", G.BTG_CLOSED)):
        g = ClosedDG(**geom, couplings=0.5, neff=3.97,
                     wavelength_range=(1.50, 1.53), name=f"ref_{name}", save_path=tmp(name))
        closed_case(f"closed/{name}", g, k_probe, 150_000, wf_levels=(0, 50))

    # dict couplings
    g = ClosedDG(**G.FG_CLOSED, couplings={0: 0.5, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.5},
                 neff=3.97, wavelength_range=(1.55, 1.56), name="ref_flower_dict", save_path=tmp("fgd"))
    closed_case("closed/flower_dict_couplings", g, k_probe, 60_000, stats=False)


# ---------------------------------------------------------------------------
# Open graphs
# ---------------------------------------------------------------------------

def open_case(prefix, graph, k_probe, resonances_kw, exp_kw=None):
    topology(prefix, graph)
    S = np.array([graph.S(k) for k in k_probe])
    store(prefix, S=S, n_leads=graph.n_leads)
    spectrum, deg = graph.find_resonances(**resonances_kw, plot=False, verbose=False)
    store(prefix, resonances=spectrum, degeneracies=deg)
    sectors(prefix, graph)
    if spectrum.size:
        vecs = graph.resonance_vectors(0)
        U = graph.U(spectrum[0])
        store(prefix, res_wf_residual=np.linalg.norm(vecs[0] - U @ vecs[0]))
        if hasattr(graph, "localization_measures"):        # reference_v3 and later
            # multiplicity of find_resonances (reference_v5; up to v4 a relative 1e-3 threshold)
            _, S_norm, ipr = graph._compute_all_resonance_quantities()
            store(prefix, loc_S_norm_resonances=S_norm, loc_ipr_resonances=ipr)
    if exp_kw is not None:
        exp = graph.find_experimental_resonances(**exp_kw, plot=False, cache=False,
                                                 verbose=False)
        store(prefix, exp_resonances=exp)


def run_open():
    k_probe = [4.0 - 1e-5j, 4.05 + 0.0j, 4.1003 - 3e-6j]
    res_small = dict(N_re=40_000, N_im=8, im_min=-1e-4, im_max=0.0)

    g = OpenDG(**G.RING_1ACCESS, couplings=0.5,
               access_couplings=0.3, neff=2.41, wavelength_range=(1.54, 1.56),
               lead_orientations={(0, 0): -150.0, (0, 2): -30.0},
               name="ref_ring1", save_path=tmp("ring1"))
    open_case("open/ring_1access", g, k_probe, res_small,
              dict(n_points=20_000, component=(1, 0), k_imag=2e-7, amp_threshold=0))

    g = OpenDG(**G.RING_2ACCESS, couplings=0.5, access_couplings=0.3, neff=2.41,
               wavelength_range=(1.54, 1.56),
               lead_orientations={(0, 0): -150.0, (0, 2): -30.0, (1, 0): 150.0, (1, 2): 30.0},
               name="ref_ring2", save_path=tmp("ring2"))
    open_case("open/ring_2access", g, k_probe, res_small)

    for name, geom in (("flower", G.FG_OPEN), ("bowtie", G.BTG_OPEN)):
        g = OpenDG(**geom, couplings=0.5, access_couplings=0.3, neff=3.97,
                   wavelength_range=(1.550, 1.552), name=f"ref_open_{name}", save_path=tmp("o" + name))
        open_case(f"open/{name}", g, k_probe, dict(N_re=8_000, N_im=8, im_min=-1e-4, im_max=0.0))

    g = OpenDG(**G.FG_OPEN, couplings=G.coupling_dispersive, access_couplings=0.3,
               neff=G.neff_dispersive, wavelength_range=(1.550, 1.552), name="ref_open_flower_disp",
               save_path=tmp("ofd"))
    store("open/flower_dispersive", S=np.array([g.S(k) for k in k_probe]))


# ---------------------------------------------------------------------------
# Classical dynamics
# ---------------------------------------------------------------------------

def run_classical():
    for name, geom in (("flower", G.FG_CLOSED), ("bowtie", G.BTG_CLOSED),
                       ("double_ring", G.DOUBLE_RING)):
        g = ClassicalGraphs(**geom, vertex_matrices=G.directional_coupler(0.5),
                            name=f"ref_cl_{name}", save_path=tmp("cl" + name))
        P = g.perron_frobenius()
        summary = g.classical_summary()
        rho = g.evolve_rho(g.basis_state(index=0), n_steps=7)
        store(f"classical/{name}", eigvals=np.sort_complex(summary["eigenvalues"]),
              spectral_gap=summary["spectral_gap"], is_ergodic=summary["is_ergodic"],
              is_mixing=summary["is_mixing"], rho7_sorted=np.sort(rho),
              invariant_sorted=np.sort(summary["invariant_measure"]),
              P_col_sums=P.sum(axis=0))


# ---------------------------------------------------------------------------
# Periodic orbits
# ---------------------------------------------------------------------------

def run_orbits():
    for name, geom in (("flower", G.FG_CLOSED), ("bowtie", G.BTG_CLOSED)):
        g = PeriodicOrbits(**geom, ng=3.97, name=f"ref_po_{name}", save_path=tmp("po" + name))
        irr = g.find_irreducible_orbits(verbose=False)
        allo = g.find_orbits_up_to_length(1500.0, max_orbits=10**7, max_search_nodes=10**9,
                                          verbose=False)
        store(f"orbits/{name}",
              irreducible_lengths=np.sort([r["physical_length_um"] for r in irr.values()]),
              irreducible_names=np.array(sorted(irr)),
              all_lengths=np.sort([r["physical_length_um"] for r in allo.values()]),
              n_all=len(allo))


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "reference_new.npz")
    for step in (run_closed, run_open, run_classical, run_orbits):
        print(f"[reference] {step.__name__} ...", flush=True)
        step()
    np.savez_compressed(out_path, **OUT)
    print(f"[reference] {len(OUT)} arrays written to {out_path}")
