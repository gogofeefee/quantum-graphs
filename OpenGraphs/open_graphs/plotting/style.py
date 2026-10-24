#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style constants and axis-configuration helpers shared across plotting methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
    MultipleLocator,
    NullLocator,
    FixedLocator,
    FixedFormatter,
)

# ---------------------------------------------------------------------------
# Physical / display constants
# ---------------------------------------------------------------------------
C_UM_THz = 299.792458          # c so that f[THz] = C_UM_THz / lambda[µm]

MAJOR_TICK_LEN = 10
MINOR_TICK_LEN = 5
MAJOR_TICK_W   = 2
MINOR_TICK_W   = 1.5


# ---------------------------------------------------------------------------
# Wavelength ↔ frequency converters
# ---------------------------------------------------------------------------

def lambda_to_freq(lam_um):
    """λ (µm) → frequency (THz).  Returns inf where lam_um == 0."""
    lam_um = np.asarray(lam_um, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return C_UM_THz / lam_um


def freq_to_lambda(freq_thz):
    """Frequency (THz) → λ (µm).  Returns inf where freq_thz == 0."""
    freq_thz = np.asarray(freq_thz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return C_UM_THz / freq_thz


# ---------------------------------------------------------------------------
# Tick / axis helpers
# ---------------------------------------------------------------------------

def regular_major_locator():
    return MaxNLocator(nbins=6, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])


def set_regular_major_minor(axis_obj, with_minor=True):
    axis_obj.set_major_locator(regular_major_locator())
    if with_minor:
        axis_obj.set_minor_locator(AutoMinorLocator(2))
    else:
        axis_obj.set_minor_locator(NullLocator())


def configure_y_axis(ax, ylabel=None, labelsize=22):
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=24)
    set_regular_major_minor(ax.yaxis, with_minor=True)
    ax.tick_params(
        axis="y", which="major",
        left=True, right=True, direction="in",
        length=MAJOR_TICK_LEN, width=MAJOR_TICK_W, labelsize=labelsize,
    )
    ax.tick_params(
        axis="y", which="minor",
        left=True, right=True, direction="in",
        length=MINOR_TICK_LEN, width=MINOR_TICK_W,
    )


def configure_phase_y_axis_pi(ax, ylabel=None, labelsize=22):
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=24)
    phase_ticks  = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    phase_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    ax.yaxis.set_major_locator(FixedLocator(phase_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(phase_labels))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-1.05 * np.pi, 1.05 * np.pi)
    ax.tick_params(
        axis="y", which="major",
        left=True, right=True, direction="in",
        length=MAJOR_TICK_LEN, width=MAJOR_TICK_W, labelsize=labelsize,
    )
    ax.tick_params(axis="y", which="minor", left=False, right=False)


def configure_primary_x_axis(
    ax, show_bottom=True, show_top=False,
    labelbottom=True, xlabel=None, labelsize=22,
):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=24)
    set_regular_major_minor(ax.xaxis, with_minor=True)
    ax.tick_params(
        axis="x", which="major",
        bottom=show_bottom, top=show_top, direction="in",
        length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
        labelsize=labelsize, labelbottom=labelbottom, labeltop=False,
    )
    ax.tick_params(
        axis="x", which="minor",
        bottom=show_bottom, top=show_top, direction="in",
        length=MINOR_TICK_LEN, width=MINOR_TICK_W,
    )


def add_secondary_top_axis(ax, choice, set_label=True, labelpad=14):
    """Add a secondary x-axis (wavelength ↔ frequency)."""
    if choice == "wavelength":
        secax = ax.secondary_xaxis("top", functions=(lambda_to_freq, freq_to_lambda))
    else:
        secax = ax.secondary_xaxis("top", functions=(freq_to_lambda, lambda_to_freq))
    if set_label:
        top_label = "Frequency (THz)" if choice == "wavelength" else "Wavelength (µm)"
        secax.set_xlabel(top_label, fontsize=22, labelpad=labelpad)
    secax.xaxis.set_major_locator(regular_major_locator())
    secax.xaxis.set_minor_locator(NullLocator())
    secax.tick_params(
        axis="x", which="major",
        top=True, bottom=False, direction="in",
        length=MAJOR_TICK_LEN, width=MAJOR_TICK_W, labelsize=18, labeltop=True,
    )
    secax.tick_params(axis="x", which="minor", top=False, bottom=False)
    return secax


def set_x_limits(ax, choice, lmin, lmax):
    if choice == "wavelength":
        ax.set_xlim(lmin, lmax)
    else:
        ax.set_xlim(lambda_to_freq(lmax), lambda_to_freq(lmin))


def break_wrapped_phase_curve(x, phi, jump_threshold=np.pi):
    x   = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if x.size <= 1:
        return x, phi
    x_out   = [x[0]]
    phi_out = [phi[0]]
    for i in range(1, len(x)):
        if np.abs(phi[i] - phi[i - 1]) > jump_threshold:
            x_out.append(np.nan)
            phi_out.append(np.nan)
        x_out.append(x[i])
        phi_out.append(phi[i])
    return np.asarray(x_out), np.asarray(phi_out)


def format_bottom_xlabel(choice):
    return "Wavelength (µm)" if choice == "wavelength" else "Frequency (THz)"


def validate_x_axis(choice):
    choice = str(choice).lower()
    if choice not in ("wavelength", "frequency"):
        raise ValueError("`x_axis` must be either 'wavelength' or 'frequency'.")
    return choice
