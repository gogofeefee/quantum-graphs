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


@pytest.mark.slow
def test_reference_data_set_is_reproduced(tmp_path):
    out = tmp_path / "reference_new.npz"
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run([sys.executable, os.path.join(REF, "generate_reference.py"), str(out)],
                   check=True, env=env, capture_output=True)
    problems = compare(np.load(os.path.join(REF, "reference_v3.npz")), np.load(out))
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
