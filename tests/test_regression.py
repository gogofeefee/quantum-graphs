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
    problems = compare(np.load(os.path.join(REF, "reference_v7.npz")), np.load(out))
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


def test_v4_to_v5_differences_are_the_documented_ones():
    """reference_v4 -> reference_v5: sector analysis. Spectra are computed per
    independent sector and statistics use the desymmetrised spectrum with the
    Weyl law of the sector; resonance wavefunctions use the multiplicity found by
    find_resonances. Nothing stored changes; the sector structure is added."""
    problems = compare(np.load(os.path.join(REF, "reference_v4.npz")),
                       np.load(os.path.join(REF, "reference_v5.npz")))
    assert all(line.startswith("EXTRA") and "/sector_" in line for line in problems), problems
    assert len(problems) == 33


def test_v5_to_v6_differences_are_the_documented_ones():
    """reference_v5 -> reference_v6: bounded resonance refinement. The
    Nelder-Mead stage of find_resonances was unbounded: on the rings (all
    resonances below im_min = -1e-4 of v5) it ran away to Re k = 3.456 and
    -0.610, outside the window, and the resonance at Re k = 4.06639 was
    missed. The rings are now searched down to im_min = -3e-4 and their 13
    resonances are the analytic ones (|dk| < 1e-15). The argument-principle
    count of each open case is added (equal to the summed multiplicities)."""
    v5 = np.load(os.path.join(REF, "reference_v5.npz"))
    v6 = np.load(os.path.join(REF, "reference_v6.npz"))
    problems = compare(v5, v6)
    changed = {line.split(":")[1].strip() for line in problems if line.startswith("VALUE")}
    added = {line.split(":")[1].strip() for line in problems if line.startswith("EXTRA")}
    assert changed == {"open/ring_1access/resonances", "open/ring_2access/resonances"}
    assert added == {f"open/{c}/resonance_count"
                     for c in ("ring_1access", "ring_2access", "flower", "bowtie")}
    assert len(problems) == len(changed) + len(added)
    for case in ("ring_1access", "ring_2access", "flower", "bowtie"):
        assert v6[f"open/{case}/resonance_count"] == v6[f"open/{case}/degeneracies"].sum()


def test_v6_to_v7_differences_are_the_documented_ones():
    """reference_v6 -> reference_v7: Newton resonance search and first-order
    continuation of a dispersive index off the real axis. The resonances of the
    four open cases (constant index) are unchanged (|dk| < 1e-15). With a
    dispersive index, U(k) at complex k used n_eff(Re k); it now uses
    n_eff(x) + iy n_eff'(x), so S at the complex probe wavenumbers of the
    dispersive flower graph changes (3e-4) and its resonances, added, have the
    widths of the group index (n_eff(Re k) gave Im k larger by n_g / n_eff)."""
    v6 = np.load(os.path.join(REF, "reference_v6.npz"))
    v7 = np.load(os.path.join(REF, "reference_v7.npz"))
    problems = compare(v6, v7)
    changed = {line.split(":")[1].strip() for line in problems if line.startswith("VALUE")}
    added = {line.split(":")[1].strip() for line in problems if line.startswith("EXTRA")}
    assert changed == {"open/flower_dispersive/S"}
    assert added == {"open/flower_dispersive/resonances", "open/flower_dispersive/degeneracies"}
    assert len(problems) == len(changed) + len(added)
    # the real-axis probe of S is unchanged
    assert np.allclose(v6["open/flower_dispersive/S"][1], v7["open/flower_dispersive/S"][1],
                       atol=1e-13)
    for case in ("ring_1access", "ring_2access", "flower", "bowtie"):
        assert np.max(np.abs(v6[f"open/{case}/resonances"] - v7[f"open/{case}/resonances"])) < 1e-15
