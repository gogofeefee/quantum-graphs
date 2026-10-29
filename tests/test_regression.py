"""Non-regression against the stored reference data set (slow, ~1 min)."""

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")
sys.path.insert(0, REF)
from compare_reference import compare  # noqa: E402

# reference_v0 -> reference_v1: intentional changes (see CHANGELOG.md)
V0_TO_V1 = {"orbits/flower/all_lengths", "orbits/flower/n_all",
            "orbits/bowtie/all_lengths", "orbits/bowtie/n_all"}
# reference_v1 -> reference_v2: degenerate eigenvectors stored as the diagonal of
# the eigenspace projector instead of one arbitrary (LAPACK-dependent) basis vector
V1_TO_V2 = {"closed/single_ring/wf0/intensity", "closed/double_ring/wf0/intensity",
            "closed/double_ring_dispersive/wf0/intensity"}
# reference_v2 -> reference_v3: entropy / IPR (per physical bond, eigenspace projector)
# added; no stored quantity changes
V2_TO_V3_ADDED = 49


def v3_to_v4_is_documented(line):
    """reference_v3 -> reference_v4: bond lists without virtual vertices (2.0).

    Spectra, degeneracies, S-matrices, resonances, secular determinants, spectral
    statistics, entropy / IPR of the stored wavefunctions and orbit catalogues
    are unchanged. What changes: the topology keys (physical bonds instead of
    virtualised directed bonds); the size of U and of the classical operator;
    the indexing of the directed-bond intensities (unchanged port by port); the
    classical dynamics (one step = one bond traversal); residuals |a - U a| of
    other-sized matrices; and the per-level entropy / IPR of the bow tie at the
    1e-6 level (wavefunctions at refined roots, |dk| ~ 1e-9)."""
    kind, key = (x.strip() for x in line.split(":")[:2])
    if key.startswith("classical/"):
        return True
    if key.rsplit("/", 1)[-1] in ("n_directed_bonds", "n_virtual_vertices", "sorted_directed_lengths"):
        return kind == "MISSING in new"
    if key.rsplit("/", 1)[-1] in ("n_bonds", "sorted_bond_lengths"):
        return kind == "EXTRA in new"
    if key.endswith("/U_eigvals"):
        return kind == "SHAPE"
    if key.endswith("/intensity"):
        return kind == "LABELS differ"
    if key.endswith("/residual"):
        return True
    return key in ("closed/bowtie/loc_S_norm_levels", "closed/bowtie/loc_ipr_levels")


@pytest.mark.slow
def test_reference_data_set_is_reproduced(tmp_path):
    out = tmp_path / "reference_new.npz"
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run([sys.executable, os.path.join(REF, "generate_reference.py"), str(out)],
                   check=True, env=env, capture_output=True)
    problems = compare(np.load(os.path.join(REF, "reference_v4.npz")), np.load(out))
    assert not problems, "\n".join(problems)


def test_v0_to_v1_differences_are_the_documented_ones():
    problems = compare(np.load(os.path.join(REF, "reference_v0.npz")),
                       np.load(os.path.join(REF, "reference_v1.npz")))
    changed = {line.split(":")[1].strip() for line in problems}
    assert changed == V0_TO_V1


def test_v1_to_v2_differences_are_the_documented_ones():
    problems = compare(np.load(os.path.join(REF, "reference_v1.npz")),
                       np.load(os.path.join(REF, "reference_v2.npz")))
    changed = {line.split(":")[1].strip() for line in problems}
    assert changed == V1_TO_V2


def test_v2_to_v3_differences_are_the_documented_ones():
    problems = compare(np.load(os.path.join(REF, "reference_v2.npz")),
                       np.load(os.path.join(REF, "reference_v3.npz")))
    assert all(line.startswith("EXTRA") and "/loc_" in line for line in problems), problems
    assert len(problems) == V2_TO_V3_ADDED


def test_v3_to_v4_differences_are_the_documented_ones():
    problems = compare(np.load(os.path.join(REF, "reference_v3.npz")),
                       np.load(os.path.join(REF, "reference_v4.npz")))
    undocumented = [line for line in problems if not v3_to_v4_is_documented(line)]
    assert not undocumented, "\n".join(undocumented)
    # the physics did not move: spectra, S, resonances, orbits are not in the list
    assert not any(k in line for line in problems
                   for k in ("/spectrum", "/S:", "/resonances", "orbits/", "/sigma2", "/delta3"))
