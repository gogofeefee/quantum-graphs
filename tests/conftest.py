"""Shared fixtures: quiet, non-interactive runs and temporary save paths."""

import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import quantum_graphs as qg  # noqa: E402

qg.set_options(verbose=False, show=False)


@pytest.fixture
def save(tmp_path):
    """Keyword arguments sending every output of a graph to a temporary folder."""
    return dict(save_path=str(tmp_path))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
