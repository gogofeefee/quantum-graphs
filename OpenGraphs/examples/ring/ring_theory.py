#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ring_theory.py — Analytical ring resonator response
====================================================

Computes the through-port (B) and cross-port (A) S-matrix components of an
open ring resonator analytically, for comparison with DirectionalGraphs
simulation results.

Geometry
--------
Single access coupler (1-access) or two identical couplers (2-access).
Coupling constant c = power cross-coupling fraction.

S-matrix components (1-access ring, port conventions matching config files)
    B(k) : through-port   (same-coupler, same-side output)
    A(k) : cross-port     (same-coupler, cross-side output — intra-ring field)

Plots produced
--------------
    1. |S|² vs frequency (primary) / wavelength (secondary)
    2. arg(S) vs frequency (primary) / wavelength (secondary)
    3. Combined two-panel figure

Physical units: wavelength in µm, frequency in THz.
"""

# ============================================================================
#  Imports
# ============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator, MaxNLocator, NullLocator,
    FixedLocator, FixedFormatter,
)

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _argv0 = sys.argv[0] if sys.argv else ""
    _here  = (os.path.dirname(os.path.abspath(_argv0))
              if os.path.isfile(_argv0) else os.getcwd())

# ============================================================================
#  Style constants  (mirrors open_graphs/plotting/style.py)
# ============================================================================

C_UM_THz = 299.792458   # c in µm·THz

MAJOR_TICK_LEN = 10
MINOR_TICK_LEN = 5
MAJOR_TICK_W   = 2
MINOR_TICK_W   = 1.5


def _lambda_to_freq(lam):
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return C_UM_THz / lam


def _freq_to_lambda(f):
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return C_UM_THz / f


def _regular_locator():
    return MaxNLocator(nbins=6, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])


def _configure_x(ax, xlabel=None, show_top=False, labelbottom=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=24)
    ax.xaxis.set_major_locator(_regular_locator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", bottom=True, top=show_top,
                   direction="in", length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=22, labelbottom=labelbottom)
    ax.tick_params(axis="x", which="minor", bottom=True, top=show_top,
                   direction="in", length=MINOR_TICK_LEN, width=MINOR_TICK_W)


def _configure_y(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=24)
    ax.yaxis.set_major_locator(_regular_locator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", left=True, right=True,
                   direction="in", length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=22)
    ax.tick_params(axis="y", which="minor", left=True, right=True,
                   direction="in", length=MINOR_TICK_LEN, width=MINOR_TICK_W)


def _configure_phase_y(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=24)
    ticks  = [-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi]
    labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-1.05 * np.pi, 1.05 * np.pi)
    ax.tick_params(axis="y", which="major", left=True, right=True,
                   direction="in", length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=22)


def _add_secondary_top(ax, x_axis):
    if x_axis == "frequency":
        secax = ax.secondary_xaxis("top", functions=(_freq_to_lambda, _lambda_to_freq))
        secax.set_xlabel("Wavelength (µm)", fontsize=22, labelpad=14)
    else:
        secax = ax.secondary_xaxis("top", functions=(_lambda_to_freq, _freq_to_lambda))
        secax.set_xlabel("Frequency (THz)", fontsize=22, labelpad=14)
    secax.xaxis.set_major_locator(_regular_locator())
    secax.xaxis.set_minor_locator(NullLocator())
    secax.tick_params(axis="x", which="major", top=True, bottom=False,
                      direction="in", length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                      labelsize=18)
    return secax


def _break_phase(x, phi, threshold=np.pi):
    x_out, phi_out = [x[0]], [phi[0]]
    for i in range(1, len(x)):
        if np.abs(phi[i] - phi[i-1]) > threshold:
            x_out.append(np.nan)
            phi_out.append(np.nan)
        x_out.append(x[i])
        phi_out.append(phi[i])
    return np.asarray(x_out), np.asarray(phi_out)


# ============================================================================
#  Physical parameters
# ============================================================================

neff             = 2.41
R                = 100.0          # ring radius (µm)
L_half           = np.pi * R      # half-ring length (µm)
c                = 0.3            # power cross-coupling fraction

wavelength_range = (1.48, 1.64)   # µm
N_pts            = 2_000_000
x_axis           = "frequency"    # "wavelength" or "frequency"

save_figs   = True
file_format = "pdf"

lambda_min, lambda_max = wavelength_range
lambdas = np.linspace(lambda_min, lambda_max, N_pts)
k_real  = 2.0 * np.pi / lambdas   # vacuum wavenumber (µm⁻¹)

print(f"Ring geometry : R = {R:.1f} µm,  L_half = {L_half:.3f} µm,  "
      f"L_ring = {2*L_half:.3f} µm")
print(f"Coupling      : c = {c:.3f}")
print(f"neff          : {neff:.5f}")


# ============================================================================
#  Loss model  (same formulae as config files)
# ============================================================================

alpha_dB_per_cm   = 1.0
coupler_loss_frac = 0.05
n_vertices        = 1          # 1-access ring → 1 coupler
total_length      = 2 * L_half

k_mid    = 0.5 * (2.0*np.pi/lambda_max + 2.0*np.pi/lambda_min)
neff_mid = float(neff)

k_imag_abs = np.log(10) / (1e4 * 20 * neff_mid) * alpha_dB_per_cm
k_imag_cpl = (
    -n_vertices * np.log(1 - coupler_loss_frac)
    / (2 * neff_mid * total_length)
)
k_imag = k_imag_abs + k_imag_cpl

print("\n=== Loss Contributions ===")
print(f"  k_mid                          = {k_mid:.4f} µm⁻¹")
print(f"  neff                           = {neff_mid:.5f}")
print(f"  Material absorption  k_imag_abs = {k_imag_abs:.3e} µm⁻¹")
print(f"  Coupler loss         k_imag_cpl = {k_imag_cpl:.3e} µm⁻¹")
print(f"  Total                k_imag     = {k_imag:.3e} µm⁻¹\n")


# ============================================================================
#  Complex wavenumber  k_eff = neff * (k_real + i * k_imag)
# ============================================================================

k_complex = neff * (k_real + 1j * k_imag)


# ============================================================================
#  Analytical S-matrix  (1-access ring, 4-port directional coupler)
#
#  Coupling constant c = power cross-coupling fraction.
#  z = exp(i k_eff L_half)  — phase / decay over one half-ring.
#
#  B(k) : through-port   — (1,0) in port indexing of config files
#  A(k) : cross-port     — (3,2) in port indexing of config files (intra-ring)
# ============================================================================

def B_of_k(k_eff, c):
    """Through-port: same-coupler, same-side output."""
    z   = np.exp(1j * k_eff * L_half)
    num = np.sqrt(1 - c) * (1 - z**2)
    den = 1 - (1 - c) * z**2
    return num / den


def A_of_k(k_eff, c):
    """Cross-port: same-coupler, cross-side output (intra-ring field)."""
    z   = np.exp(1j * k_eff * L_half)
    den = 1 - (1 - c) * z**2
    return -c * z / den


B = B_of_k(k_complex, c)
A = A_of_k(k_complex, c)

# ============================================================================
#  Display axis
# ============================================================================

if x_axis == "frequency":
    x_vals  = _lambda_to_freq(lambdas)
    x_min   = _lambda_to_freq(lambda_max)
    x_max   = _lambda_to_freq(lambda_min)
    x_label = "Frequency (THz)"
else:
    x_vals  = lambdas
    x_min   = lambda_min
    x_max   = lambda_max
    x_label = "Wavelength (µm)"

# Sort along display axis (frequency is inverted relative to wavelength)
order  = np.argsort(x_vals)
x_plot = x_vals[order]
B_plot = B[order]
A_plot = A[order]

# Save directory (next to this script)
save_dir = os.path.join(
    _here,
    f"ring_theory_R{R:.0f}_neff{neff:.3f}_c{c:.2f}",
)
if save_figs:
    os.makedirs(save_dir, exist_ok=True)


# ============================================================================
#  Helper: save figure
# ============================================================================

def _save(fig, name):
    if not save_figs:
        return
    path = os.path.join(save_dir, f"{name}.{file_format}")
    fig.savefig(path, format=file_format, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")


# ============================================================================
#  Plot 1 — Through-port |B|² + arg(B)  (two-panel)
# ============================================================================

fig1, (ax_amp, ax_phi) = plt.subplots(
    2, 1, figsize=(10, 10), sharex=True, layout="constrained",
)

# Amplitude panel
ax_amp.plot(x_plot, np.abs(B_plot)**2, color="black", linewidth=1.2)
configure_y_axis = _configure_y     # local alias
_configure_y(ax_amp, ylabel=r"$|B(k)|^2$")
ax_amp.tick_params(axis="x", which="both", labelbottom=False)
ax_amp.tick_params(axis="x", which="major",
                   bottom=True, top=False, direction="in",
                   length=MAJOR_TICK_LEN, width=MAJOR_TICK_W)
ax_amp.tick_params(axis="x", which="minor",
                   bottom=True, top=False, direction="in",
                   length=MINOR_TICK_LEN, width=MINOR_TICK_W)
ax_amp.set_xlim(x_min, x_max)
_add_secondary_top(ax_amp, x_axis)

# Phase panel
phi_B        = np.angle(B_plot)
x_brk, p_brk = _break_phase(x_plot, phi_B)
ax_phi.plot(x_brk, p_brk, color="black", linewidth=1.2)
_configure_phase_y(ax_phi, ylabel=r"$\arg(B(k))$")
_configure_x(ax_phi, xlabel=x_label)
ax_phi.set_xlim(x_min, x_max)

_save(fig1, f"ring_theory_B_amp_phase_vs{x_axis}")
plt.show()


# ============================================================================
#  Plot 2 — Cross-port |A|² + arg(A)  (two-panel)
# ============================================================================

fig2, (ax_amp2, ax_phi2) = plt.subplots(
    2, 1, figsize=(10, 10), sharex=True, layout="constrained",
)

ax_amp2.plot(x_plot, np.abs(A_plot)**2, color="black", linewidth=1.2)
_configure_y(ax_amp2, ylabel=r"$|A(k)|^2$")
ax_amp2.tick_params(axis="x", which="both", labelbottom=False)
ax_amp2.tick_params(axis="x", which="major",
                    bottom=True, top=False, direction="in",
                    length=MAJOR_TICK_LEN, width=MAJOR_TICK_W)
ax_amp2.tick_params(axis="x", which="minor",
                    bottom=True, top=False, direction="in",
                    length=MINOR_TICK_LEN, width=MINOR_TICK_W)
ax_amp2.set_xlim(x_min, x_max)
_add_secondary_top(ax_amp2, x_axis)

phi_A         = np.angle(A_plot)
x_brk2, p_brk2 = _break_phase(x_plot, phi_A)
ax_phi2.plot(x_brk2, p_brk2, color="black", linewidth=1.2)
_configure_phase_y(ax_phi2, ylabel=r"$\arg(A(k))$")
_configure_x(ax_phi2, xlabel=x_label)
ax_phi2.set_xlim(x_min, x_max)

_save(fig2, f"ring_theory_A_amp_phase_vs{x_axis}")
plt.show()


# ============================================================================
#  Plot 3 — Both components |B|² and |A|² zoomed over a few FSR
#
#  The FSR is FSR_THz = c / (neff * L_ring) in frequency.
#  zoom_n_fsr controls how many resonances are shown.
#  Labels are placed directly on the curves (no legend box).
# ============================================================================

zoom_n_fsr   = 4           # number of FSR to show in the zoom window
zoom_center  = None        # THz or µm — set to None to use range centre

FSR_THz = C_UM_THz / (neff * 2 * L_half)   # free spectral range in THz
FSR_um  = (np.mean([lambda_min, lambda_max]))**2 / (neff * 2 * L_half)

print(f"FSR = {FSR_THz:.4f} THz  ({FSR_um*1e3:.3f} nm)")

if x_axis == "frequency":
    x_ctr = 0.5 * (x_min + x_max) if zoom_center is None else float(zoom_center)
    half_w = 0.5 * zoom_n_fsr * FSR_THz
    x_zoom_min = x_ctr - half_w
    x_zoom_max = x_ctr + half_w
else:
    x_ctr = 0.5 * (lambda_min + lambda_max) if zoom_center is None else float(zoom_center)
    half_w = 0.5 * zoom_n_fsr * FSR_um
    x_zoom_min = x_ctr - half_w
    x_zoom_max = x_ctr + half_w

# Restrict arrays to the zoom window
mask   = (x_plot >= x_zoom_min) & (x_plot <= x_zoom_max)
x_zm   = x_plot[mask]
B_zm   = np.abs(B_plot[mask])**2
A_zm   = np.abs(A_plot[mask])**2

fig3, ax3 = plt.subplots(figsize=(10, 8), layout="constrained")

ax3.plot(x_zm, B_zm, color="black",   linewidth=1.5)
ax3.plot(x_zm, A_zm, color="#d62728", linewidth=1.5, linestyle="--")

_configure_y(ax3, ylabel=r"$|S(k)|^2$")
_configure_x(ax3, xlabel=x_label)
ax3.set_xlim(x_zoom_min, x_zoom_max)
_add_secondary_top(ax3, x_axis)


_save(fig3, f"ring_theory_BA_amplitude_zoom{zoom_n_fsr}FSR_vs{x_axis}")
plt.show()

# ============================================================================
#  Energy conservation check
# ============================================================================

residual = np.max(np.abs(np.abs(B)**2 + np.abs(A)**2 - 1.0))
print(f"Energy conservation check: max | |B|² + |A|² − 1 | = {residual:.2e}")
if residual > 1e-6:
    print("  ⚠  Non-trivial residual — losses are included (k_imag ≠ 0)")
else:
    print("  ✓  Lossless limit satisfied")
