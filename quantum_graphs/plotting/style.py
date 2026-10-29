#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style constants, axis-configuration helpers and figure output shared across
plotting methods.

Every plotting method ends the same way: ``save_figure`` if asked, then
``show_figure(fig, show)`` (from :mod:`quantum_graphs.core.config`) when it
created the figure, and returns the figure.
"""

import os

import numpy as np
from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
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
# Figure output
# ---------------------------------------------------------------------------

def save_figure(fig, folder, filename, what="Figure", verbose=None, **savefig_kw):
    """Save ``fig`` as ``folder/filename`` (format from the extension, 300 dpi,
    tight bounding box unless overridden) and report it; returns the path."""
    from ..core.config import log
    path = os.path.join(folder, filename)
    savefig_kw.setdefault("dpi", 300)
    savefig_kw.setdefault("bbox_inches", "tight")
    fig.savefig(path, **savefig_kw)
    log(f"✓ {what} saved to {path}", verbose=verbose)
    return path


def box_ticks(ax, labelsize=22):
    """Inward major and minor ticks on the four sides of ``ax``."""
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in",
                   length=MAJOR_TICK_LEN, width=MAJOR_TICK_W,
                   labelsize=labelsize, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=MINOR_TICK_LEN, width=MINOR_TICK_W,
                   top=True, right=True)


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
    """4–6 major ticks at 1, 2, 2.5 or 5 × 10ⁿ."""
    return MaxNLocator(nbins=6, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])


def set_regular_major_minor(axis_obj, with_minor=True):
    """Regular major ticks on ``axis_obj`` and one minor tick between them."""
    axis_obj.set_major_locator(regular_major_locator())
    if with_minor:
        axis_obj.set_minor_locator(AutoMinorLocator(2))
    else:
        axis_obj.set_minor_locator(NullLocator())


def configure_y_axis(ax, ylabel=None, labelsize=22):
    """Label and inward ticks (both sides) of a linear y axis."""
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
    """y axis of a phase: ticks at multiples of π/2 on [−π, π]."""
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
    """Label and inward ticks of the bottom x axis (optionally mirrored on top)."""
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
    """x limits of the window (lmin, lmax) in µm, on a wavelength or frequency axis."""
    if choice == "wavelength":
        ax.set_xlim(lmin, lmax)
    else:
        ax.set_xlim(lambda_to_freq(lmax), lambda_to_freq(lmin))


def break_wrapped_phase_curve(x, phi, jump_threshold=np.pi):
    """Insert NaN between samples whose phase jumps by more than ``jump_threshold``
    (the 2π wraps), so that a wrapped phase is drawn without vertical lines."""
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
    """Bottom x label of a wavelength or frequency axis."""
    return "Wavelength (µm)" if choice == "wavelength" else "Frequency (THz)"


def validate_x_axis(choice):
    """'wavelength' or 'frequency' (case-insensitive), else ValueError."""
    choice = str(choice).lower()
    if choice not in ("wavelength", "frequency"):
        raise ValueError("`x_axis` must be either 'wavelength' or 'frequency'.")
    return choice
