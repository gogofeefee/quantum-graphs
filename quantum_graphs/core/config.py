"""
Package-wide options: verbosity and interactive display of figures.

    import quantum_graphs as qg
    qg.set_options(verbose=False, show=False)     # quiet, no plt.show()
    with qg.options(show=False):
        graph.plot_graph(save_fig=True)

Defaults reproduce the historical behaviour (messages printed, ``plt.show()``
called after each figure). They can also be set with the environment
variables ``QUANTUM_GRAPHS_VERBOSE=0`` and ``QUANTUM_GRAPHS_SHOW=0``.
``n_workers`` (default: half the CPU cores, at most 8; environment variable
``QUANTUM_GRAPHS_WORKERS``) is the number of threads used for the batched
SVDs / solves of spectrum and resonance scans; results are identical for any
value.
Every plotting method also accepts ``show=`` and most computations
``verbose=``; an explicit argument always wins over the global option.
"""

import contextlib
import os


def _default_workers():
    env = os.environ.get("QUANTUM_GRAPHS_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _env_flag(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


_OPTIONS = {
    "verbose": _env_flag("QUANTUM_GRAPHS_VERBOSE", True),
    "show": _env_flag("QUANTUM_GRAPHS_SHOW", True),
    # threads used for batched linear algebra (LAPACK releases the GIL;
    # results do not depend on this number)
    "n_workers": _default_workers(),
}


def get_option(name):
    return _OPTIONS[name]


def set_options(**kwargs):
    """Set global options (``verbose``, ``show``)."""
    for key, val in kwargs.items():
        if key not in _OPTIONS:
            raise KeyError(f"Unknown option {key!r}; available: {sorted(_OPTIONS)}.")
        _OPTIONS[key] = max(1, int(val)) if key == "n_workers" else bool(val)


@contextlib.contextmanager
def options(**kwargs):
    """Temporarily change global options."""
    old = dict(_OPTIONS)
    set_options(**kwargs)
    try:
        yield
    finally:
        _OPTIONS.update(old)


def resolve(name, value):
    """Explicit argument if given, global option otherwise."""
    return _OPTIONS[name] if value is None else bool(value)


def log(*args, verbose=None, **kwargs):
    """``print`` that respects the ``verbose`` option."""
    if resolve("verbose", verbose):
        print(*args, **kwargs)


def show_figure(show=None):
    """Call ``plt.show()`` unless display is disabled."""
    if resolve("show", show):
        import matplotlib.pyplot as plt
        plt.show()
