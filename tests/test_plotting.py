"""Smoke tests: every plotting method runs and writes its file (small graphs)."""

import os

import numpy as np

import quantum_graphs as qg


def n_files(path, ext):
    return sum(f.endswith(ext) for f in os.listdir(path))


def test_closed_plots(save):
    g = qg.ClosedDirectionalGraphs(**qg.layouts.flower_graph(), couplings=0.5, neff=3.97,
                                   wavelength_range=(1.50, 1.51), bond_spacing=[40, -40],
                                   **save)
    g.plot_graph(show_lengths=True, show_port_markers=True, save_fig=True, file_format="png")
    spectrum, _ = g.compute_spectrum(N_scan=40_000)
    g.plot_zeta_B(wavelength_range=(1.50, 1.502), N_plot=2000, spectrum=spectrum, save_fig=True)
    g.plot_wavefunction(a_in=g.compute_wavefunction(spectrum[0]), k=spectrum[0], save_fig=True)
    g.plot_entropy(save_fig=True)
    g.plot_ipr(save_fig=True)
    g.plot_staircase_function(plot_weyl=True, weyl_diff=True, save_fig=True)
    g.short_range_correlations(plot_wigner=["goe", "gue", "poissonian"], cumulative_spacing=True,
                               save_fig=True)
    g.long_range_correlations(L_max=4, L_step=0.5, plot_wigner=["goe", "poissonian"], save_fig=True)
    g.plot_length_spectrum(L_max=3000, n_points=5000, save_fig=True)
    assert n_files(save["save_path"], ".pdf") >= 10


def test_base_closed_graph_plot(save):
    g = qg.ClosedGraphs(**qg.layouts.tetrahedron(), vertex_matrices=qg.neumann_matrix(3),
                        neff=2.41, wavelength_range=(1.50, 1.51), **save)
    g.plot_graph(show_lengths=True, save_fig=True)


def test_open_plots(save):
    g = qg.OpenDirectionalGraphs(**qg.layouts.bow_tie_graph(open=True), couplings=0.5,
                                 access_couplings=0.3, neff=3.97, wavelength_range=(1.550, 1.553),
                                 **save)
    g.plot_graph(show_lengths=True, show_port_markers=True, save_fig=True)
    g.find_resonances(N_re=6_000, N_im=6, im_min=-1e-4, im_max=0.0, plot=True)
    g.find_experimental_resonances(n_points=20_000, k_imag=2e-7, amp_threshold=0, plot=True)
    kw = dict(lambda_min=1.550, lambda_max=1.553, n_points=3000, components=(1, 0), k_imag=2e-6,
              save_fig=True)
    g.plot_S_amplitude(spectrum=g.best_spectrum, x_axis="frequency", **kw)
    g.plot_S_phase(**kw)
    g.plot_S_amplitude_phase(spectrum=g.best_spectrum, **kw)
    g.plot_S_fourier(xlim_um=(0, 3000), **kw)
    g.plot_S_complex_plane(spectrum=g.best_spectrum, **kw)
    g.plot_energy_distribution(k=float(np.real(g.best_spectrum[0])) + 2e-6j, injection_port=0,
                               save_fig=True)
    g.plot_resonance_wavefunction(index=0, save_fig=True)
    g.plot_entropy(save_fig=True)
    g.plot_ipr(save_fig=True)
    g.plot_staircase_function(save_fig=True)
    g.short_range_correlations(save_fig=True)
    g.long_range_correlations(L_max=3, L_step=0.5, save_fig=True)
    g.plot_length_spectrum(L_max=3000, n_points=5000, save_fig=True)
    assert n_files(save["save_path"], ".pdf") >= 12


def test_classical_plots(save):
    g = qg.DirectionalClassicalGraphs(**qg.layouts.bow_tie_graph(), couplings=0.5, **save)
    g.plot_graph(show_lengths=True, show_port_markers=True, save_fig=True)
    rho0 = g.basis_state(index=0)
    g.plot_rho_after_steps(rho0=rho0, n_steps=3, save_fig=True, filename="rho3.pdf")
    g.animate_rho(rho0=rho0, n_steps=2, fps=5, n_subframes_move=2, n_subframes_fade=1,
                  save_animation=True, writer="gif", filename="anim.gif")
    assert os.path.exists(os.path.join(save["save_path"], "anim.gif"))


def test_orbit_plots(save):
    po = qg.PeriodicOrbits(**qg.layouts.bow_tie_graph(), ng=3.97, **save)
    po.plot_graph(show_lengths=True, save_fig=True)
    irr = po.find_irreducible_orbits()
    po.plot_orbit(sorted(irr)[0], save_fig=True)
    po.export_orbits_csv(which="irreducible")
    assert n_files(save["save_path"], ".csv") == 1
