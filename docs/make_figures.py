#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures of the documentation (README and docs/), written to docs/images/.

    python docs/make_figures.py            # every figure (~10 s)
    python docs/make_figures.py hero open  # only some of them

Every number shown is computed here with the public API of the package; the
graphs are the bow-tie graph (``layouts.bow_tie_graph``, a fabricated device)
and the complete pentagon K5 of ``examples/closed_graphs/Pentagon``. Cached
results go to a temporary folder, nothing is written next to this script
except the PNG files.
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Use the installed package (pip install -e .) or, failing that, this checkout.
try:
    import quantum_graphs as qg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import quantum_graphs as qg
from quantum_graphs.plotting.drawing import (
    draw_bond_values, draw_bonds, draw_vertices, finish_axes, size_scale)

OUT = Path(__file__).resolve().parent / "images"
WORK = tempfile.mkdtemp(prefix="qg_docs_")

# Colours: data in ink, one hue per random-matrix ensemble, sectors blue / orange
INK, MUTED, GRID = "#1b1b1b", "#8a8a85", "#d9d8d4"
GOE, GUE, POISSON = "#2a78d6", "#eb6834", "#8a8a85"
SECTOR_COLORS = ("#2a78d6", "#eb6834")
ORBIT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.fontsize": 10,
    "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "figure.facecolor": "white",
})
qg.set_options(show=False, verbose=False)

# Pentagon K5 of examples/closed_graphs/Pentagon (lengths drawn in [80, 120] µm)
PENTAGON_BONDS = [
    (0, 0, 4, 2, 110.958242), (0, 1, 1, 3, 97.555138), (0, 2, 2, 0, 114.343917),
    (0, 3, 3, 0, 107.894721), (1, 0, 3, 2, 83.767094), (1, 1, 2, 3, 119.024894),
    (1, 2, 4, 1, 110.445588), (2, 1, 3, 3, 111.442572), (2, 2, 4, 0, 85.124545),
    (3, 1, 4, 3, 98.015438),
]
PENTAGON_POSITIONS = [(-100 * np.sin(a), 100 * np.cos(a))
                      for a in 2 * np.pi * np.arange(5) / 5]

NEFF_BTG = 3.97   # constant index of the bow-tie examples


# ============================================================================
#  Graphs
# ============================================================================

def closed_bow_tie(wavelength_range=(1.40, 1.70)):
    return qg.ClosedDirectionalGraphs(
        **qg.layouts.bow_tie_graph(), couplings=0.5, neff=NEFF_BTG,
        wavelength_range=wavelength_range, name="BowTie", base_dir=WORK)


def open_bow_tie():
    return qg.OpenDirectionalGraphs(
        **qg.layouts.bow_tie_graph(open=True), couplings=0.5, access_couplings=0.3,
        neff=NEFF_BTG, wavelength_range=(1.545, 1.555), name="BowTieOpen", base_dir=WORK)


def pentagon():
    return qg.ClosedDirectionalGraphs(
        vertex_positions=PENTAGON_POSITIONS, bonds=PENTAGON_BONDS, couplings=0.5,
        neff=2.45, wavelength_range=(0.8, 2.0), name="Pentagon", base_dir=WORK)


# ============================================================================
#  Drawing helpers
# ============================================================================

def draw_mode(ax, graph, vec):
    """Bonds coloured by the intensity of one eigenvector (normalised to its max)."""
    rho = graph.bond_intensities(vec)
    paths = graph.bond_paths
    lc = draw_bond_values(ax, paths, rho / rho.max(), width_min=2.5, width_gain=5.0)
    draw_vertices(ax, graph.topology, 0.07 * size_scale(graph.topology), fontsize=9)
    finish_axes(ax, paths, margin=0.04)
    return lc


def draw_directed(ax, graph, d, color, length=0.44, offset=0.07, lw=2.0, alpha=1.0):
    """Arrow along directed bond ``d`` at mid-length, beside its bond (on the
    right of the direction of travel, so that the two directions do not
    overlap). ``length`` and ``offset`` are fractions of the shortest bond."""
    paths = graph.bond_paths
    scale = size_scale(graph.topology)
    p = paths.directed_point(d, np.array([0.5]))[0]
    u = paths.directed_tangent(d, np.array([0.5]))[0]
    p = p + offset * scale * np.array([u[1], -u[0]])
    half = 0.5 * length * scale * u
    ax.annotate("", xy=p + half, xytext=p - half, zorder=6,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25",
                                color=color, alpha=alpha, lw=lw, shrinkA=0, shrinkB=0))


def draw_plain_graph(ax, graph, color=INK, linewidth=2.0, fontsize=9, margin=0.04):
    draw_bonds(ax, graph.bond_paths, color=color, linewidth=linewidth)
    draw_vertices(ax, graph.topology, 0.07 * size_scale(graph.topology), fontsize=fontsize)
    finish_axes(ax, graph.bond_paths, margin=margin)


def draw_density(ax, graph, rho, color="#6250d6"):
    """Classical density on the directed bonds: one arrow per occupied directed
    bond, width and opacity growing with rho."""
    draw_plain_graph(ax, graph, color=GRID)
    rho = np.asarray(rho, dtype=float) / max(float(np.max(rho)), 1e-300)
    for d in np.flatnonzero(rho > 1e-3):
        draw_directed(ax, graph, d, color, lw=1.0 + 3.5 * rho[d], alpha=0.35 + 0.65 * rho[d])


def draw_orbit(ax, graph, record, color):
    """A periodic orbit: its directed bonds as arrows over the light-grey graph."""
    draw_plain_graph(ax, graph, color=GRID)
    paths = graph.bond_paths
    for d in record["path"]:
        s = np.linspace(0.2, 0.8, 60)
        xy = paths.directed_point(d, s)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=3.0, solid_capstyle="round", zorder=4)
        ax.annotate("", xy=xy[-1], xytext=xy[-4], zorder=6,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.4",
                                    color=color, lw=0, shrinkA=0, shrinkB=0))


def style_axes(ax):
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote docs/images/{name}.png")


def most_structured_mode(graph, k_levels):
    """Level of ``k_levels`` whose (non-degenerate) mode has the lowest entropy."""
    best = None
    for k in k_levels:
        vec = graph.compute_wavefunction(k)
        S_norm = graph.localization_measures(vec)[1]
        if best is None or S_norm < best[1]:
            best = (k, S_norm, vec)
    return best


# ============================================================================
#  Figures
# ============================================================================

def fig_hero():
    """One graph, four points of view (bow-tie graph)."""
    closed = closed_bow_tie(wavelength_range=(1.545, 1.555))
    spectrum, _ = closed.compute_spectrum()
    k_mode, S_norm, vec = most_structured_mode(closed, spectrum[:40])

    opened = open_bow_tie()
    lam = np.linspace(1.5495, 1.5515, 4000)
    T = np.abs(opened.S(2 * np.pi / lam)[:, 1, 0]) ** 2

    classical = qg.DirectionalClassicalGraphs(
        **qg.layouts.bow_tie_graph(), couplings=0.5, name="BowTieCl", base_dir=WORK)
    rho3 = classical.evolve_rho(classical.basis_state(index=0), n_steps=3)

    orbits = qg.PeriodicOrbits(**qg.layouts.bow_tie_graph(), ng=NEFF_BTG,
                               name="BowTieOrbits", base_dir=WORK)
    irreducible = orbits.find_irreducible_orbits()
    orbit = sorted(irreducible.values(), key=lambda r: r["physical_length_um"])[2]

    fig = plt.figure(figsize=(15, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.95, 1, 1], wspace=0.34)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    draw_mode(axes[0], closed, vec)
    axes[0].set_title(f"Closed graph: eigenmode\n"
                      f"λ = {2 * np.pi / k_mode * 1e3:.2f} nm, S$_{{norm}}$ = {S_norm:.2f}")

    ax = axes[1]
    ax.plot(lam * 1e3, T, color=INK, lw=1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$|S_{21}|^2$")
    ax.set_xlim(lam[0] * 1e3, lam[-1] * 1e3)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1550.0, 1551.0])
    ax.set_box_aspect(0.68)
    style_axes(ax)
    ax.set_title("Open graph: transmission\nthrough the access coupler")

    draw_density(axes[2], classical, rho3)
    axes[2].set_title("Classical dynamics: ray density\n3 steps after one directed bond")

    draw_orbit(axes[3], orbits, orbit, GUE)
    axes[3].set_title(f"Periodic orbit {orbit['name']}\n"
                      f"length {orbit['physical_length_um']:.1f} µm")
    save(fig, "hero")


def fig_statistics():
    """Spectral statistics: one self-reverse sector (GOE) vs two partner sectors (GUE)."""
    rows = [("Bow-tie graph", closed_bow_tie(), GOE, "GOE"),
            ("Pentagon K$_5$", pentagon(), GUE, "GUE")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1], hspace=0.38,
                                              wspace=0.28))
    s = np.linspace(0, 3, 400)
    L = np.linspace(0, 10, 400)
    for r, (title, graph, color, ens) in enumerate(rows):
        graph.compute_spectrum()
        ax = axes[r, 0]
        sectors = graph.sectors
        if len(sectors) == 1:
            draw_plain_graph(ax, graph)
            text = f"{title}\none sector, its own reverse\n→ time-reversal symmetry"
        else:
            draw_plain_graph(ax, graph, color=GRID)
            for sec, c in zip(sectors, SECTOR_COLORS):
                for d in sec.directed:
                    draw_directed(ax, graph, d, c, lw=2.5)
            text = (f"{title}\ntwo partner sectors (blue / orange)\n"
                    "→ no time-reversal symmetry")
        ax.set_title(text, fontsize=11)

        x = graph.unfolded_spectrum()
        n = len(x)
        ax = axes[r, 1]
        ax.hist(np.diff(x), bins=np.arange(0, 3.01, 0.1), density=True,
                histtype="stepfilled", color="#e9e8e4", edgecolor=INK, lw=1.0)
        ax.plot(s, qg.nnsd(s, "poisson"), color=POISSON, ls="--", lw=1.6, label="Poisson")
        ax.plot(s, qg.nnsd(s, "goe"), color=GOE, lw=2.0, label="GOE")
        ax.plot(s, qg.nnsd(s, "gue"), color=GUE, lw=2.0, label="GUE")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Spacing $s$")
        ax.set_ylabel("$P(s)$")
        ax.set_title(f"Nearest-neighbour spacings\n{n} levels, expected {ens}", fontsize=11)
        if r == 0:
            ax.legend(loc="upper right")

        for c, (fn, rmt, label) in enumerate(((graph.sigma2, qg.sigma2_rmt, r"$\Sigma^2(L)$"),
                                              (graph.delta3, qg.delta3_rmt, r"$\Delta_3(L)$"))):
            ax = axes[r, 2 + c]
            Lx, y = fn(L_max=10, L_step=0.2)
            ax.plot(L, rmt(L, "poisson"), color=POISSON, ls="--", lw=1.6)
            ax.plot(L, rmt(L, "goe"), color=GOE, lw=2.0)
            ax.plot(L, rmt(L, "gue"), color=GUE, lw=2.0)
            ax.plot(Lx, y, "o", ms=4, mfc="white", mec=INK, mew=1.0, label="graph")
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1.1 * max(rmt(L, "goe").max(), y.max()))
            ax.set_xlabel("$L$")
            ax.set_ylabel(label)
            ax.set_title("Number variance" if c == 0 else "Spectral rigidity", fontsize=11)
        for ax in axes[r, 1:]:
            style_axes(ax)
    save(fig, "spectral_statistics")


def fig_closed():
    """Closed graph: eigenphases and levels, staircase vs Weyl law, length spectrum."""
    graph = closed_bow_tie()
    spectrum, _ = graph.compute_spectrum()
    k = np.sort(np.real(spectrum))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw=dict(wspace=0.3))

    ax = axes[0]
    lam = np.linspace(1.5500, 1.5515, 3000)
    phases = np.angle(np.linalg.eigvals(graph.U(2 * np.pi / lam)))
    levels = 2 * np.pi / k
    shown = levels[(levels > lam[0]) & (levels < lam[-1])]
    for l in shown:
        ax.axvline(l * 1e3, color=GUE, lw=1.2, alpha=0.7, zorder=1)
    ax.plot(np.repeat(lam[:, None] * 1e3, phases.shape[1], 1), phases, ".", ms=0.6,
            color=INK, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(lam[0] * 1e3, lam[-1] * 1e3)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_xticks([1550.25, 1550.75, 1551.25])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Eigenphases $\theta_j$ of $U(k)$")
    ax.set_title("Levels (orange) = eigenphases crossing 0", fontsize=11)

    ax = axes[1]
    n = np.arange(1, k.size + 1)
    offset = np.mean(n - 0.5 - graph.weyl_staircase(k))     # one constant fitted
    sel = slice(800, 840)
    kk = np.linspace(k[sel][0] - 0.0005, k[sel][-1] + 0.0005, 500)
    ax.step(np.r_[kk[0], k[sel], kk[-1]], np.r_[n[sel][0] - 1, n[sel], n[sel][-1]],
            where="post", color=INK, lw=1.4, label=r"$N(k)$")
    ax.plot(kk, graph.weyl_staircase(kk) + offset, color=GOE, lw=2.0,
            label=r"Weyl law $\frac{L_s}{2\pi}\int n_g\,dk$")
    ax.set_xlim(kk[0], kk[-1])
    ax.set_xlabel(r"$k$ (µm$^{-1}$)")
    ax.set_ylabel("Number of levels")
    ax.legend(loc="upper left")
    ax.set_title(f"Staircase and Weyl law (40 of {k.size} levels)", fontsize=11)

    ax = axes[2]
    ell = np.linspace(0, 1200, 24000)
    rho = np.abs(np.exp(-1j * np.outer(ell, k * NEFF_BTG)).sum(axis=1))
    ax.plot(ell, rho, color=INK, lw=0.8)
    orbits = qg.PeriodicOrbits(**qg.layouts.bow_tie_graph(), ng=NEFF_BTG,
                               name="BowTieOrbits", base_dir=WORK)
    irreducible = sorted(orbits.find_irreducible_orbits().values(),
                         key=lambda r: r["physical_length_um"])
    top = rho[ell > 100].max()
    for rec, dy in zip(irreducible, (0.0, 0.11, 0.0, 0.0, 0.0)):
        lo = rec["physical_length_um"]
        peak = rho[np.abs(ell - lo) < 3].max()
        ax.annotate(rec["name"], xy=(lo, peak), xytext=(lo, peak + (0.08 + dy) * top),
                    ha="center", fontsize=9, color=GUE,
                    arrowprops=dict(arrowstyle="-", color=GUE, lw=0.8))
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, top * 1.3)
    ax.set_xlabel(r"Length $\ell$ (µm)")
    ax.set_ylabel(r"$|\hat\rho(\ell)|$")
    ax.set_title("Length spectrum: peaks at the periodic orbits", fontsize=11)
    for ax in axes:
        style_axes(ax)
    save(fig, "closed_graph")


def fig_open():
    """Open graph: transmission, resonances in the complex plane, field of one resonance."""
    graph = open_bow_tie()
    res, _ = graph.find_resonances(im_min=-1e-4, im_max=0.0)
    lam = np.linspace(1.548, 1.552, 6000)
    T = np.abs(graph.S(2 * np.pi / lam)[:, 1, 0]) ** 2
    keep = (2 * np.pi / np.real(res) > lam[0]) & (2 * np.pi / np.real(res) < lam[-1])
    res = res[keep]

    fig = plt.figure(figsize=(15, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 2.2, 0.9], height_ratios=[1.3, 1],
                          hspace=0.08, wspace=0.25)
    ax_g = fig.add_subplot(gs[:, 0])
    graph.plot_graph(ax=ax_g, vertex_fontsize=9, bond_linewidth=2.0)
    ax_g.set_title("Bow-tie graph + access coupler A\n(leads dashed)", fontsize=11)

    ax_t = fig.add_subplot(gs[0, 1])
    ax_t.plot(lam * 1e3, T, color=INK, lw=0.8)
    ax_t.set_ylabel(r"$|S_{21}|^2$")
    ax_t.set_ylim(0, 1.05)
    ax_t.tick_params(labelbottom=False)
    ax_t.set_title(f"Transmission $|S_{{21}}|^2$ and the {len(res)} resonances (poles of S)",
                   fontsize=11)
    ax_r = fig.add_subplot(gs[1, 1], sharex=ax_t)
    ax_r.plot(2 * np.pi / np.real(res) * 1e3, np.imag(res) * 1e6, "o", ms=4, mfc=GUE,
              mec="white", mew=0.5)
    ax_r.set_xlabel("Wavelength (nm)")
    ax_r.set_ylabel(r"Im $k$ (10$^{-6}$ µm$^{-1}$)")
    ax_r.set_xlim(lam[0] * 1e3, lam[-1] * 1e3)
    for ax in (ax_t, ax_r):
        style_axes(ax)

    ax_w = fig.add_subplot(gs[:, 2])
    i = int(np.argmax(np.imag(res)))          # narrowest resonance
    vec = graph.resonance_vectors(i)[0]
    draw_mode(ax_w, graph, vec)
    ax_w.set_title(f"Narrowest resonance\nQ = {np.real(res[i]) / (-2 * np.imag(res[i])):.0f}",
                   fontsize=11)
    save(fig, "open_graph")


def fig_classical():
    """Classical dynamics: ρ after n steps and the Perron-Frobenius spectrum."""
    graph = qg.DirectionalClassicalGraphs(
        **qg.layouts.bow_tie_graph(), couplings=0.5, name="BowTieCl", base_dir=WORK)
    rho0 = graph.basis_state(index=0)
    steps = (0, 1, 4, 30)
    fig = plt.figure(figsize=(15, 3.7))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.12, 1.0], wspace=0.10)
    for c, n in enumerate(steps):
        ax = fig.add_subplot(gs[c])
        rho = graph.evolve_rho(rho0, n_steps=n)
        draw_density(ax, graph, rho)
        ax.set_title(f"n = {n}", fontsize=11)
    ax = fig.add_subplot(gs[5])
    ev, _ = graph.perron_frobenius_spectrum()
    t = np.linspace(0, 2 * np.pi, 400)
    gap = graph.spectral_gap()
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.0)
    ax.plot((1 - gap) * np.cos(t), (1 - gap) * np.sin(t), color=GOE, lw=1.0, ls="--")
    ax.plot(ev.real, ev.imag, "o", ms=4.5, mfc=INK, mec="white", mew=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"Re $\mu$")
    ax.set_ylabel(r"Im $\mu$")
    ax.set_title(f"Perron-Frobenius spectrum\ngap 1 − |μ₂| = {gap:.3f}", fontsize=11)
    style_axes(ax)
    save(fig, "classical_dynamics")


def fig_orbits():
    """Periodic orbits: the shortest irreducible orbits of the bow-tie graph."""
    graph = qg.PeriodicOrbits(**qg.layouts.bow_tie_graph(), ng=NEFF_BTG,
                              name="BowTieOrbits", base_dir=WORK)
    orbits = sorted(graph.find_irreducible_orbits().values(),
                    key=lambda r: r["physical_length_um"])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), gridspec_kw=dict(wspace=0.05))
    for ax, rec, color in zip(axes, orbits, ORBIT_COLORS * 2):
        draw_orbit(ax, graph, rec, color)
        ax.set_title(f"{rec['name']}\n{rec['physical_length_um']:.1f} µm", fontsize=11)
    save(fig, "periodic_orbits")


FIGURES = {"hero": fig_hero, "statistics": fig_statistics, "closed": fig_closed,
           "open": fig_open, "classical": fig_classical, "orbits": fig_orbits}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        print(f"{name}:")
        FIGURES[name]()
