#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PeriodicOrbits — Periodic-orbit enumeration on closed directional-coupler graphs.

A graph is defined, as everywhere in the package, by ``vertex_positions`` and
``bonds`` = ``[(i, p, j, q[, L]), ...]``; self-loops and multiple bonds are
ordinary bonds.

Scope
-----
Unlike `ClassicalGraphs`/`ClosedGraphs`, this module does NOT accept an
arbitrary vertex scattering matrix. It assumes every vertex is a lossless
4-port directional coupler with zero self/back reflection:

    ports (0, 1) <-> ports (2, 3)

i.e. entering via port 0 or 1 always exits via port 2 or 3, and vice versa.
This is exactly the vertex model of the `DirectionalGraphs` classes of this
package (direct back-reflection is zero), so periodic orbits can be
enumerated from the topology alone, without coupling-constant values.

What a "periodic orbit" is here
--------------------------------
Following Kottos & Smilansky (Ann. Phys. 274 (1999) 76-124): an orbit is an
itinerary of successively connected directed bonds, periodic with period n
if it returns to its starting directed bond after n steps.

- Irreducible orbit: no vertex appears more than once in the code (hence no
  bond is traversed twice either). There are finitely many of these, and
  every longer orbit can be built by concatenating them.
- Primitive orbit: the code is not the repetition of a shorter code.
- Reducible / non-primitive orbits: everything else. Their number grows
  exponentially with allowed length, since every vertex offers 2 valid
  continuations (the two ports on the opposite arm).

Naming
------
Orbits are named by their canonical vertex sequence (vertices numbered from
1 in index order), made unique under the two operations that leave a
periodic orbit unchanged: cyclic rotation of the code, and traversal in the
reverse direction. The lexicographically smallest such representative is
used, e.g. "1-4-5". If two physically distinct orbits (using different
parallel bonds between the same vertices) share the same vertex-sequence
name, the name is disambiguated with a bracket suffix, in the order in which
the search finds them: "1-4[a]", "1-4[b]", ...

Records
-------
``path`` is the tuple of directed-bond indices of the orbit (numbering of
:mod:`quantum_graphs.core.topology`), ``path_ports`` the same as
``(i, p, j, q)`` tuples. Lengths are in micrometers; an optional group index
``ng`` converts physical lengths to optical lengths (L_opt = ng * L_phys).
"""

# ======================================================================
# Imports
# ======================================================================

from ..core.config import log
from ..core.config import resolve
from ..core.base import GraphBase
from ..core.persistence import topology_hash
from ..plotting.topology_plot import TopologyPlotMixin
import os
import warnings
import csv
import datetime

from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath


# %% ======================================================================
# Helper utilities (general-purpose, no class yet)
# ======================================================================

def _as_path(path_like):
    """Convert input to pathlib.Path."""
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _format_float_for_name(x, ndigits=3):
    """Format a float in a deterministic compact way for folder/file names."""
    return f"{float(x):.{ndigits}f}"


# %% ======================================================================
# PeriodicOrbits class — geometry, transition graph,
# orbit enumeration, naming, export, and plotting
# ======================================================================

class PeriodicOrbits(GraphBase, TopologyPlotMixin):
    """
    Periodic-orbit enumeration on a closed graph of 4-port directional
    couplers (see module docstring for the exact vertex model).
    """

    # ==================================================================
    # Construction
    # ==================================================================
    def __init__(
        self,
        vertex_positions,
        bonds,
        ng=None,
        bond_spacing=None,
        name="MyPeriodicOrbitGraph",
        save_path=None,
        base_dir=None,
    ):
        """
        Parameters
        ----------
        vertex_positions : list of (float, float)
        bonds : list of (i, p, j, q) or (i, p, j, q, L) — L in um, Euclidean if omitted.
        ng : float or None — group index used to convert physical lengths
            (um) to optical lengths (um). Defaults to 1.0 (optical == physical)
            if not given.
        bond_spacing : float | sequence | None — drawing of parallel bonds.
        name : str
        save_path : str | Path | None
        base_dir : str | Path | None — base directory for the default save
            path. Priority: ``save_path`` > ``base_dir`` > directory of the
            running script > ``cwd``.
        """
        self.ng = float(ng) if ng is not None else 1.0
        self.name = str(name)
        self._setup_topology(vertex_positions, bonds)
        self.bond_spacing = bond_spacing
        for i in range(self.n_vertices):
            if self.topology.valency[i] != 4:
                raise ValueError(
                    "PeriodicOrbits only supports 4-port directional-coupler "
                    f"vertices. Vertex {i} has valency {self.topology.valency[i]} "
                    "(expected 4)."
                )
        self._build_bond_transitions()

        # Orbit registries
        self.orbit_registry = {}       # name -> record (union of all searches)
        self.irreducible_orbits = {}   # name -> record (last find_irreducible_orbits() call)
        self.orbits = {}               # name -> record (last find_orbits_up_to_length() call)
        self._orbit_dbond_keys = {}    # canonical dbond-path key -> name
        self._ambiguous_base_names = set()

        self._setup_save_path(self.name, self._build_save_dir_name(), save_path, base_dir)

    # ==================================================================
    # Transition graph (directional-coupler rule)
    # ==================================================================
    @staticmethod
    def _allowed_exit_ports(entry_port):
        """
        Ports reachable when arriving at a directional coupler via
        `entry_port`: ports (0, 1) form one arm and ports (2, 3) the other;
        entering on one arm always exits on the other (zero back-reflection).
        """
        return [2, 3] if entry_port in (0, 1) else [0, 1]

    def _build_bond_transitions(self):
        """
        Build self.bond_transitions: directed bond d -> list of valid next
        directed bonds (those leaving the vertex where d arrives).
        """
        t = self.topology
        self.bond_transitions = []
        for d in range(t.n_directed):
            v, p = int(t.dst_vertex[d]), int(t.dst_port[d])
            nxts = []
            for q in self._allowed_exit_ports(p):
                nxt = t.leaving(v, q)
                if nxt is None:
                    raise RuntimeError(
                        f"Open port at vertex {v}, port {q}: PeriodicOrbits "
                        "requires a fully closed graph."
                    )
                nxts.append(nxt)
            self.bond_transitions.append(nxts)

    def directed_bond_ports(self, d):
        """``(i, p, j, q)``: directed bond ``d`` goes from port (i, p) to port (j, q)."""
        t = self.topology
        return (int(t.src_vertex[d]), int(t.src_port[d]), int(t.dst_vertex[d]), int(t.dst_port[d]))

    # ==================================================================
    # Saving helpers
    # ==================================================================
    def _build_save_dir_name(self):
        info_str = f"Nv={self.n_vertices}_Nb={self.n_bonds}"
        bs = self.bond_spacing
        if bs is None:
            spacing_str = "bs=None"
        elif np.isscalar(bs):
            spacing_str = f"bs={_format_float_for_name(bs, ndigits=3)}"
        else:
            spacing_str = "bs=seq"
        ng_str = f"ng={_format_float_for_name(self.ng, ndigits=3)}"
        return f"{info_str}_{spacing_str}_{ng_str}_{topology_hash(self.topology)}"

    def _save_parameters(self):
        filepath = self.save_path / "graph_parameters.txt"
        t = self.topology
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("### PeriodicOrbits Graph Parameters ###\n")
            f.write(f"Graph name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n")
            f.write("Vertex model: directional coupler only (ports 0,1 <-> ports 2,3)\n\n")
            f.write(f"Group index ng: {self.ng}\n\n")

            f.write("### Counts ###\n")
            f.write(f"Vertices: {self.n_vertices}\n")
            f.write(f"Bonds: {self.n_bonds}\n")
            f.write(f"Directed bonds: {t.n_directed}\n")
            f.write(f"Total physical length: {self.total_length:.6f}\n\n")

            f.write("### Vertex positions ###\n")
            for i, pos in enumerate(t.positions):
                f.write(f"{i}: ({pos[0]:.6f}, {pos[1]:.6f})\n")

            f.write("\n### Bonds (i, p, j, q, L) ###\n")
            for b, bond in enumerate(t.bonds):
                f.write(f"{b}: {bond}\n")

    # ==================================================================
    # Graph plotting
    # ==================================================================
    def plot_graph(
        self,
        ax=None,
        show_lengths: bool = False,
        show_port_markers: bool = True,
        xl: bool = False,
        save_fig: bool = False,
        file_format: str = "svg",
        title=None,
        **kwargs,
    ):
        """
        Plot the graph (light-gray bonds, so that orbits stand out) with the
        port markers of the shared ``plot_graph`` (blue: ports 0, 1; orange:
        ports 2, 3).

        xl : bool
            If True, scale up vertices, edges, fonts, and port markers
            (everything thicker/bigger) for figures that will be shrunk
            very small in a paper. Off by default.

        Returns
        -------
        fig, ax
        """
        scale = 1.8 if xl else 1.0
        suffix = "XL" if xl else None
        return TopologyPlotMixin.plot_graph(
            self, show_lengths=show_lengths, show_port_markers=show_port_markers,
            vertex_radius_frac=0.065 * scale, marker_radius_frac=0.03 * scale,
            vertex_fontsize=20 * scale, length_fontsize=14 * scale,
            bond_linewidth=1.5 * scale, bond_color="lightgray", ax=ax, title=title,
            save_fig=save_fig, file_format=file_format, filename_suffix=suffix, **kwargs)

    # ==================================================================
    # Canonicalization (rotation + reversal invariance)
    # ==================================================================
    @staticmethod
    def _bracelet_min(seq, reverse_fn):
        """
        Lexicographically smallest representative of a cyclic sequence,
        under both rotation and reversal (via reverse_fn applied element-wise
        to the reversed sequence).
        """
        seq = list(seq)
        n = len(seq)
        if n == 0:
            return tuple()

        best = None
        for i in range(n):
            rot = tuple(seq[i:] + seq[:i])
            if best is None or rot < best:
                best = rot

        rev = [reverse_fn(x) for x in reversed(seq)]
        for i in range(n):
            rot = tuple(rev[i:] + rev[:i])
            if rot < best:
                best = rot

        return best

    @staticmethod
    def _primitive_info(seq):
        """
        Return (is_primitive, minimal_period, multiplicity) for a cyclic
        sequence `seq` of length n: the smallest d | n such that seq is the
        r = n/d-fold repetition of seq[:d].
        """
        n = len(seq)
        for d in range(1, n):
            if n % d != 0:
                continue
            if all(seq[i] == seq[i % d] for i in range(n)):
                return False, d, n // d
        return True, n, 1

    def _assign_name(self, base_name):
        """
        Return a unique orbit name for `base_name`, adding a bracket suffix
        ([a], [b], ...) if `base_name` is already used by a physically
        different orbit (e.g. a different parallel bond between the same
        two vertices). Retroactively renames the first colliding orbit too.
        """
        if base_name not in self.orbit_registry:
            return base_name

        if base_name not in self._ambiguous_base_names:
            self._ambiguous_base_names.add(base_name)
            old_record = self.orbit_registry.pop(base_name)
            old_new_name = f"{base_name}[a]"
            old_record["name"] = old_new_name
            self.orbit_registry[old_new_name] = old_record
            for k, v in list(self._orbit_dbond_keys.items()):
                if v == base_name:
                    self._orbit_dbond_keys[k] = old_new_name
            next_index = 1  # 'b'
        else:
            existing_variants = [nm for nm in self.orbit_registry if nm.startswith(base_name + "[")]
            next_index = len(existing_variants)

        return f"{base_name}[{chr(ord('a') + next_index)}]"

    def _register_orbit(self, path, physical_length, registry):
        """
        Register a closed directed-bond walk `path` (tuple of directed-bond
        indices d1..dn, with a valid transition from dn back to d1) as a periodic
        orbit, deduplicating against rotations/reversal of already-known
        orbits. Returns the (possibly pre-existing) orbit name.
        """
        path = tuple(int(d) for d in path)
        dbond_key = self._bracelet_min(path, self.topology.reverse)

        if dbond_key in self._orbit_dbond_keys:
            name = self._orbit_dbond_keys[dbond_key]
            registry[name] = self.orbit_registry[name]
            return name

        vertex_seq_0idx = tuple(int(self.topology.src_vertex[d]) for d in path)
        canon_vseq = self._bracelet_min(vertex_seq_0idx, lambda x: x)
        base_name = "-".join(str(v + 1) for v in canon_vseq)
        name = self._assign_name(base_name)

        is_irreducible = len(set(vertex_seq_0idx)) == len(vertex_seq_0idx)
        is_primitive, period_d, multiplicity = self._primitive_info(list(canon_vseq))

        primitive_name = None
        if not is_primitive:
            prim_seq = self._bracelet_min(canon_vseq[:period_d], lambda x: x)
            primitive_name = "-".join(str(v + 1) for v in prim_seq)

        optical_length = physical_length * self.ng

        record = {
            "name": name,
            "vertex_sequence": tuple(int(v + 1) for v in vertex_seq_0idx),
            "canonical_vertex_sequence": tuple(int(v + 1) for v in canon_vseq),
            "path": path,
            "path_ports": tuple(self.directed_bond_ports(d) for d in path),
            "n_real_hops": len(vertex_seq_0idx),
            "n_bonds": len(path),
            "physical_length_um": float(physical_length),
            "optical_length_um": float(optical_length),
            "is_irreducible": bool(is_irreducible),
            "is_primitive": bool(is_primitive),
            "multiplicity": int(multiplicity),
            "primitive_name": primitive_name,
        }

        self.orbit_registry[name] = record
        self._orbit_dbond_keys[dbond_key] = name
        registry[name] = record
        return name

    # ==================================================================
    # Irreducible orbits (unbounded length; finite in number)
    # ==================================================================
    def find_irreducible_orbits(self, max_orbits=200_000, verbose=None):
        """
        Enumerate ALL irreducible periodic orbits (no vertex repeated in
        the code). These are finite in number (period <= total vertex
        count) and every other periodic orbit can be built by concatenating
        them.

        Returns
        -------
        dict : name -> orbit record
        """
        verbose = resolve("verbose", verbose)
        registry = {}
        t = self.topology
        for start_dbond in range(t.n_directed):
            visited = {int(t.src_vertex[start_dbond])}
            path = [start_dbond]
            self._dfs_irreducible(start_dbond, start_dbond, path, visited, registry, max_orbits)
            if len(registry) >= max_orbits:
                if verbose:
                    print(f"[find_irreducible_orbits] max_orbits={max_orbits} reached; stopping early.")
                break

        # Same re-keying as find_orbits_up_to_length(); see the comment there.
        registry = {r["name"]: r for r in registry.values()}
        self.irreducible_orbits = registry
        if len(registry) >= max_orbits:
            warnings.warn(f"find_irreducible_orbits stopped at max_orbits={max_orbits}: the list "
                          "is INCOMPLETE.", RuntimeWarning, stacklevel=2)
        if verbose:
            print(f"Found {len(registry)} irreducible periodic orbits.")
        return registry

    def _dfs_irreducible(self, start_dbond, current_dbond, path, visited, registry, max_orbits):
        if len(registry) >= max_orbits:
            return
        for nxt in self.bond_transitions[current_dbond]:
            if nxt == start_dbond:
                physical_length = float(sum(self.topology.directed_length[b] for b in path))
                self._register_orbit(list(path), physical_length, registry)
                if len(registry) >= max_orbits:
                    return
                continue

            nxt_vertex = int(self.topology.src_vertex[nxt])
            if nxt_vertex in visited:
                continue

            visited.add(nxt_vertex)
            path.append(nxt)
            self._dfs_irreducible(start_dbond, nxt, path, visited, registry, max_orbits)
            path.pop()
            visited.discard(nxt_vertex)

    # ==================================================================
    # All periodic orbits (primitive or not) up to a physical length
    # ==================================================================
    def find_orbits_up_to_length(
        self,
        L_max_um,
        max_orbits=20_000,
        max_search_nodes=5_000_000,
        verbose=None,
    ):
        """
        Enumerate ALL periodic orbits (irreducible or not, primitive or
        not) whose total PHYSICAL length is <= L_max_um.

        WARNING: the number of periodic orbits grows exponentially with
        L_max as soon as the graph has branching (any real vertex), since
        every real-vertex hop offers 2 valid continuations. Two safety
        valves are provided:

        - max_orbits: stop once this many distinct orbits have been found
          (result is then INCOMPLETE — lower L_max or raise this instead).
        - max_search_nodes: hard cap on the number of DFS steps explored,
          protecting against runaway search even before max_orbits is hit.

        Returns
        -------
        dict : name -> orbit record
        """
        verbose = resolve("verbose", verbose)
        registry = {}
        counters = {"nodes": 0}

        for start_dbond in range(self.topology.n_directed):
            start_length = float(self.topology.directed_length[start_dbond])
            if start_length > L_max_um:
                continue
            path = [start_dbond]
            self._dfs_bounded(
                start_dbond, start_dbond, path, start_length,
                L_max_um, registry, max_orbits, counters, max_search_nodes,
            )
            if len(registry) >= max_orbits or counters["nodes"] >= max_search_nodes:
                break

        # Re-key by each record's CURRENT name.  A retroactive rename in
        # _assign_name (X registered as "B", then a collision renames it to
        # "B[a]") fixes self.orbit_registry but cannot reach this local
        # registry, which was handed the record under the OLD key earlier in
        # the same search.  self.orbits then holds the same record object twice,
        # under "B" and "B[a]", and every count downstream is inflated: the
        # flower's catalogue reported 187 orbits where it has 167, and cluster
        # multiplicities were over-stated by the same mechanism.  Found
        # 2026-08-25.  Records are unique per name, so this cannot merge two
        # genuinely distinct orbits.
        registry = {r["name"]: r for r in registry.values()}
        self.orbits = registry
        self.orbits_complete = (len(registry) < max_orbits
                                and counters["nodes"] < max_search_nodes)
        if not self.orbits_complete:
            warnings.warn(
                f"find_orbits_up_to_length({L_max_um}) stopped at a safety cap "
                f"(max_orbits={max_orbits}, max_search_nodes={max_search_nodes}): the orbit list "
                "is INCOMPLETE. Lower L_max_um or raise the caps.", RuntimeWarning, stacklevel=2)

        if verbose:
            print(
                f"Found {len(registry)} periodic orbits with physical length "
                f"<= {L_max_um} um (search nodes explored: {counters['nodes']})."
            )
            if len(registry) >= max_orbits:
                print(
                    f"  NOTE: max_orbits={max_orbits} cap reached -> enumeration "
                    "is INCOMPLETE. Lower L_max or raise max_orbits."
                )
            if counters["nodes"] >= max_search_nodes:
                print(
                    f"  NOTE: max_search_nodes={max_search_nodes} safety cap reached "
                    "-> search was truncated. Results may be incomplete."
                )
        return registry

    def _dfs_bounded(
        self, start_dbond, current_dbond, path, cum_length, L_max,
        registry, max_orbits, counters, max_search_nodes,
    ):
        if len(registry) >= max_orbits or counters["nodes"] >= max_search_nodes:
            return

        for nxt in self.bond_transitions[current_dbond]:
            # Closing the orbit does not add length: test closure BEFORE the
            # length bound (the reverse order missed every orbit longer than
            # L_max minus its shortest bond).
            if nxt == start_dbond:
                self._register_orbit(list(path), cum_length, registry)
                if len(registry) >= max_orbits:
                    return

            new_length = cum_length + float(self.topology.directed_length[nxt])
            if new_length > L_max:
                continue

            counters["nodes"] += 1
            if counters["nodes"] >= max_search_nodes:
                return

            path.append(nxt)
            self._dfs_bounded(
                start_dbond, nxt, path, new_length, L_max,
                registry, max_orbits, counters, max_search_nodes,
            )
            path.pop()

    # ==================================================================
    # Export
    # ==================================================================
    def export_orbits_csv(self, path=None, which="all"):
        """
        Export orbit records to a CSV file.

        Parameters
        ----------
        path : str | Path | None
            Output file. Defaults to self.save_path / "{name}_{which}_orbits.csv".
        which : "all" | "irreducible"
            Selects self.orbits or self.irreducible_orbits as the source.

        Returns
        -------
        Path
        """
        if which not in ("all", "irreducible"):
            raise ValueError("which must be 'all' or 'irreducible'.")

        source = self.orbits if which == "all" else self.irreducible_orbits
        if not source:
            raise ValueError(
                f"No orbits found for which='{which}'. Run "
                f"{'find_orbits_up_to_length()' if which == 'all' else 'find_irreducible_orbits()'} first."
            )

        if path is None:
            path = self.save_path / f"{self.name}_{which}_orbits.csv"
        path = _as_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = sorted(source.values(), key=lambda r: r["physical_length_um"])
        fieldnames = [
            "name", "vertex_sequence", "n_real_hops", "n_bonds",
            "physical_length_um", "optical_length_um",
            "is_irreducible", "is_primitive", "multiplicity", "primitive_name",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "name": r["name"],
                    "vertex_sequence": "-".join(str(v) for v in r["canonical_vertex_sequence"]),
                    "n_real_hops": r["n_real_hops"],
                    "n_bonds": r["n_bonds"],
                    "physical_length_um": f"{r['physical_length_um']:.6f}",
                    "optical_length_um": f"{r['optical_length_um']:.6f}",
                    "is_irreducible": r["is_irreducible"],
                    "is_primitive": r["is_primitive"],
                    "multiplicity": r["multiplicity"],
                    "primitive_name": r["primitive_name"] if r["primitive_name"] is not None else "",
                })

        log(f"Exported {len(rows)} orbits to {path}")
        return path

    # ==================================================================
    # Orbit plotting
    # ==================================================================
    def plot_orbit(
        self,
        name,
        color=None,
        ax=None,
        save_fig=False,
        file_format="svg",
        show_lengths=False,
        show_port_markers=True,
        linewidth=6.5,
        offset_ratio=0.0,
        shrink_frac=0.18,
        xl=False,
        title=None,
    ):
        """
        Plot the graph topology with a periodic orbit overlaid as directed
        arrows along the bonds it traverses (following curved bonds). Each
        arrow is centered on its bond, shortened symmetrically at both ends
        by `shrink_frac` of that bond's drawn length, so it never reaches the vertex disks
        (or port-marker disks) at either end. When an orbit uses the same
        physical bond in both directions, the two arrows overlap and read
        as a single "double-headed" arrow.

        Parameters
        ----------
        name : str
            Orbit name, as produced by find_irreducible_orbits() /
            find_orbits_up_to_length() / export_orbits_csv().
        color : str or None
            Arrow color. Defaults to a dark red ("#7a1f1f").
        ax : matplotlib.axes.Axes or None
        save_fig : bool
        file_format : str
        show_lengths : bool
        show_port_markers : bool
            Whether to draw the blue/orange port-marker disks (as in
            plot_graph). save_fig filenames are tagged accordingly so both
            variants can be saved side by side.
        linewidth : float
        offset_ratio : float
            Lateral offset of the orbit arrows relative to the underlying
            bond, as a fraction of the median bond length. 0.0 (default)
            draws arrows directly on the bond; increase only if you want
            forward/backward passes drawn on separate sides instead of
            overlapping into a double arrow.
        shrink_frac : float
            Fraction of each bond's own length trimmed off at each end
            (same fraction at tail and head), so the arrow is shorter than
            the link and centered on it.
        xl : bool
            If True, scale up everything (vertices, edges, fonts, port
            markers, and the orbit arrows themselves) for figures that
            will be shrunk very small in a paper. Off by default.
        title : str or None

        Returns
        -------
        fig, ax
        """
        if name not in self.orbit_registry:
            raise KeyError(
                f"Unknown orbit name '{name}'. Run find_irreducible_orbits() and/or "
                "find_orbits_up_to_length() first."
            )
        record = self.orbit_registry[name]
        color = color if color is not None else "#7a1f1f"
        scale = 1.8 if xl else 1.0

        fig, ax = self.plot_graph(
            ax=ax, show_lengths=show_lengths,
            show_port_markers=show_port_markers, xl=xl, save_fig=False,
        )
        paths = self.bond_paths
        t = self.topology
        offset_mag = float(offset_ratio) * float(np.median(t.length))

        # Group traversals by bond (regardless of direction): if a bond is
        # used in both directions somewhere in the orbit, draw ONE
        # double-headed arrow instead of two overlapping opposite arrows.
        bonds_seen = OrderedDict()
        for d in record["path"]:
            bonds_seen.setdefault(t.bond_of(d), set()).add(d)

        for b, dirs in bonds_seen.items():
            both_directions = len(dirs) == 2
            d = min(dirs)
            s_arc = np.linspace(float(shrink_frac), 1.0 - float(shrink_frac), 60)
            pts = paths.directed_point(d, s_arc)
            if offset_mag:
                tang = paths.directed_tangent(d, s_arc)
                pts = pts + offset_mag * np.stack([-tang[:, 1], tang[:, 0]], axis=1)

            arrowstyle = "<|-|>" if both_directions else "-|>"

            patch = mpatches.FancyArrowPatch(
                path=MplPath(pts),
                arrowstyle=arrowstyle, mutation_scale=38.0 * scale,
                linewidth=linewidth * scale,
                color=color, zorder=8,
                capstyle="round", joinstyle="round",
            )
            ax.add_patch(patch)

        if title is not None:
            ax.set_title(title, fontsize=14)
            fig.tight_layout()

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            safe_name = record["name"].replace("/", "_")
            marker_tag = "port_markers" if show_port_markers else "no_markers"
            xl_tag = "_XL" if xl else ""
            filename = f"{self.name}_orbit_{safe_name}_{marker_tag}{xl_tag}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight", pad_inches=0)
            log(f"Orbit plot saved to {filepath}")

        return fig, ax
