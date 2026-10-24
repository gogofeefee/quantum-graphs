"""Spectral statistics and random-matrix-theory reference curves."""

from .spectral import (
    weyl_staircase, unfold, spacings, mean_spacing, number_variance, spectral_rigidity,
)
from .rmt import (
    nnsd, sigma2_rmt, delta3_rmt, sigma2_dyson_mehta, delta3_dyson_mehta,
    normalize_ensemble, table_range,
)

__all__ = [
    "weyl_staircase", "unfold", "spacings", "mean_spacing", "number_variance",
    "spectral_rigidity",
    "nnsd", "sigma2_rmt", "delta3_rmt", "sigma2_dyson_mehta",
    "delta3_dyson_mehta", "normalize_ensemble", "table_range",
]
