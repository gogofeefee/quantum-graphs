#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed Quantum Graphs — Numerical modeling of wave propagation on closed quantum graphs.

This module defines the `ClosedGraphs` class to construct and analyze open quantum graphs
as described in the vertex scattering matrix formalism [Smilansky, Gnutzmann].

Each graph is initialized with:
- vertex positions (embedded geometry),
- a port_map to define connectivity (without open leads as `None`),
- bond lengths,
- vertex scattering matrices (static or k-dependent),
- effective index (constant or k-dependent),
- and a working wavelength range in vacuum.

Special preprocessing transforms the graph into a *simple* graph by removing:
- multiple bonds (replaced by added vertices with Neumann BC),
- self-loops (broken into chains of Neumann BC vertices).

Author: Hugo Girin
Date: July 2025
"""

# ========================================================================
#                         Imports and Path Setup
# ========================================================================

import os
import sys
import copy
import datetime
import inspect
import hashlib
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numdifftools as nd
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
from pathlib import Path


# ---------------------------------------------------------------------
# Extend sys.path to include the custom module directory containing 
# spectral statistics tools like sigma2_RMT and delta3_RMT.
# This allows importing your custom RMT functions.
# ---------------------------------------------------------------------
module_path = os.path.expanduser('~/Documents/Research/Simul/Graphs/wigner')
if module_path not in sys.path:
    sys.path.append(module_path)

# ---------------------------------------------------------------------
# Attempt to import the long-range spectral statistics functions.
# These provide GOE/GUE/GSE predictions for σ² and Δ₃ correlations.
# ---------------------------------------------------------------------
try:
    from sigma2_RMT import sigma2_RMT
    print("Successfully imported sigma2 from sigma2_RMT.")
except ImportError as e:
    print(f"Error importing sigma2: {e}")

try:
    from delta3_RMT import delta3_RMT
    print("Successfully imported delta3 from delta3_RMT.")
except ImportError as e:
    print(f"Error importing delta3: {e}")

# ========================================================================
#                 Nearest-Neighbor Spacing Distributions
# ========================================================================

def NNSD_func(s, wigner="GOE"):
    """
    Computes the Nearest-Neighbor Spacing Distribution (NNSD) based on the Wigner surmise.
    
    Parameters
    ----------
    s : float or array-like
        The normalized level spacing.
    wigner : str
        The ensemble type: "poissonian", "GOE", "GUE", or "GSE" (case-insensitive).
    
    Returns
    -------
    float or array-like
        The value of the spacing distribution at s.
    """
    wigner = wigner.lower()
    
    if wigner == "poissonian":
        return np.exp(-s)
    elif wigner == "goe":
        return (np.pi / 2) * s * np.exp(-np.pi / 4 * s**2)
    elif wigner == "gue":
        return (32 / np.pi**2) * s**2 * np.exp(-4 / np.pi * s**2)
    elif wigner == "gse":
        return (2**18 / 3**6 / np.pi**3) * s**4 * np.exp(-64 / (9 * np.pi) * s**2)
    else:
        raise ValueError("Invalid wigner type. Choose from 'poissonian', 'GOE', 'GUE', or 'GSE'.")
        
# ========================================================================
#                 OpenGraphs Class Starts
# ========================================================================

class ClosedGraphs:
    """
    Construct a closed quantum graph in the vertex–scattering formalism.
    
    A closed graph has no external leads: every port is internally connected.  
    During initialization, the geometry, bond lengths, and scattering matrices
    are set up, while self-loops and multiple bonds are automatically resolved
    using virtual vertices with Neumann boundary conditions.
    
    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of the vertices in µm.
    
    port_map : list of lists
        Connectivity specification. For vertex i and port p:
            port_map[i][p] = (j, q)
        meaning port p of vertex i connects to port q of vertex j.
        No `None` entries are allowed in ClosedGraphs.
    
    lengths : dict or None, optional
        Explicit dictionary mapping bonds to physical lengths:
            {((i, p), (j, q)) : length_in_um}
        If None, lengths are inferred from Euclidean distances between
        vertex positions.
    
    vertex_matrices : dict, list, ndarray, or callable, optional
        Scattering matrices σ_v at each vertex.
        - dict/list: one entry per vertex.
        - ndarray: same σ for all vertices.
        - callable: returns σ_v(k) as a function of wavenumber.
    
    neff : float or callable, optional
        Effective refractive index. May be constant or k-dependent.
    
    ng : float or callable, optional
        Group index. If None, it is derived numerically from neff(k).
    
    wavelength_range : tuple of (float, float), optional
        Vacuum wavelength range (µm) defining the working spectral window.
    
    virtual_vertex_spacing : float, sequence, or None, optional
        Spacing used when splitting multiple bonds. If a sequence is given,
        values are cycled across multi-bond groups.
    
    name : str, optional
        Label for the graph (used in plots and default save paths).
    
    save_path : str, pathlib.Path, or None, optional
        Directory where results will be stored.
        - If None (default), results are saved under:
              {Path.cwd()}/{name}/{_build_save_dir_name()}
        - If provided, this path is used directly (overrides `name`).
    """
    
    def __init__(self, vertex_positions, port_map,
                 lengths=None,
                 vertex_matrices=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 name="MyClosedGraph",
                 save_path=None):
        """
        Initialize a closed quantum graph (no external leads).  
        Sets geometry, bond lengths, and vertex scattering matrices,  
        while resolving loops and multiple bonds with virtual vertices.  
        """
    
        # --- Basic attributes ---
        self.vertex_positions = list(vertex_positions)
        self.port_map = port_map
        self.virtual_port_map = copy.deepcopy(port_map)  # mutated by virtual splitting
        self.lengths = lengths                           # ORIGINAL (immutable after init)
        self.n_vertices = len(vertex_positions)
        self.wavelength_range = wavelength_range
        self.neff = neff
        self.ng = ng if ng is not None else self._compute_group_index()
    
        # Normalize & prepare spacing input (supports float OR cycling list)
        self._init_spacing_sequence(virtual_vertex_spacing)
        self.virtual_vertex_spacing = virtual_vertex_spacing  # keep original for metadata
    
        # --- Compute ORIGINAL lengths if not provided ---
        if self.lengths is None:
            self.lengths = {}
            for v_idx, ports in enumerate(self.port_map):
                for p_idx, bond in enumerate(ports):
                    j, p2 = bond
                    if v_idx < j:  # avoid double-counting
                        x1, y1 = self.vertex_positions[v_idx]
                        x2, y2 = self.vertex_positions[j]
                        dist = np.hypot(x2 - x1, y2 - y1)
                        self.lengths[((v_idx, p_idx), (j, p2))] = dist
    
        # Keep a pristine scalar of original total length
        self.total_length = sum(self.lengths.values())
    
        # Create the VIRTUAL lengths dict that WILL be modified by splitting
        self.virtual_lengths = copy.deepcopy(self.lengths)

        # --- STEP 1: Build bond list from ORIGINAL port_map (for grouping) ---
        self.bonds = []       # (i, pi, j, pj)
        self.valencies = []   # ports per vertex

        for i, ports in enumerate(self.port_map):
            self.valencies.append(len(ports))
            for pi, conn in enumerate(ports):
                j, pj = conn
                if (j, pj, i, pi) not in self.bonds:
                    self.bonds.append((i, pi, j, pj))

        self.n_bonds = len(self.bonds)

        # --- STEP 2: Group bonds by vertex pair ---
        vertex_pair_bonds = defaultdict(list)
        for (i, pi, j, pj) in self.bonds:
            vpair = tuple(sorted([i, j]))
            vertex_pair_bonds[vpair].append(((i, pi), (j, pj)))

        # --- STEP 3: Break multiple bonds / loops (mutates virtual_* only) ---
        self.virtual_vertices = []
        self.updated_bonds = []  # list of ((i,pi),(j,pj)) AFTER virtualization

        for (i, j), bond_list in vertex_pair_bonds.items():
            if i == j:
                # Loops unaffected by multi-bond spacing cycling
                for bond in bond_list:
                    self._break_complex_bond(bond, loop=True, index=0, n_total=len(bond_list))
            elif len(bond_list) == 1:
                self.updated_bonds.append(bond_list[0])    # raw append is fine
            else:
                # Choose ONE spacing for this entire multi-bond group (cycled if a list was given)
                group_spacing = self._next_group_spacing()
                for k, bond in enumerate(bond_list):
                    self._break_complex_bond(
                        bond,
                        loop=False,
                        index=k,
                        n_total=len(bond_list),
                        group_spacing=group_spacing
                    )

        # Canonicalize and order everything consistently (uses virtual_lengths)
        self._order_bonds_and_lengths()

        # --- STEP 4: Build networkx graph using VIRTUAL geometry ---
        self.G = nx.Graph()
        self.vertex_scattering_matrix = {}

        for i, pos in enumerate(self.vertex_positions):
            self.G.add_node(i, pos=pos)

        for (a, pa), (b, pb) in self.updated_bonds:
            key = tuple(sorted([(a, pa), (b, pb)]))
            self.G.add_edge(a, b, length=self.virtual_lengths[key])

        # --- STEP 5: Assign vertex scattering matrices ---
        for v in range(len(self.vertex_positions)):
            if v in self.virtual_vertices:
                # Neumann condition for virtual nodes: 2×2 σ_v
                self.vertex_scattering_matrix[v] = np.array([[0, 1],
                                                             [1, 0]])
            else:
                if isinstance(vertex_matrices, (np.ndarray, list)):
                    # single σ used for all vertices OR per-vertex list
                    if isinstance(vertex_matrices, np.ndarray):
                        self.vertex_scattering_matrix[v] = vertex_matrices
                    else:
                        self.vertex_scattering_matrix[v] = vertex_matrices[v]
                elif callable(vertex_matrices):
                    # callable → same function for all vertices
                    self.vertex_scattering_matrix[v] = vertex_matrices
                elif isinstance(vertex_matrices, dict):
                    self.vertex_scattering_matrix[v] = vertex_matrices[v]
                else:
                    raise ValueError("vertex_matrices must be a dict, list, array, or callable.")

        # --- STEP 6: Metadata ---
        self.n_vertices_total = len(self.vertex_positions)
        self.n_virtual_vertices = len(self.virtual_vertices)
        self.n_internal_bonds = len(self.updated_bonds)
        self.n_directed_bonds = 2 * self.n_internal_bonds

        # --- STEP 7: Build directed bonds and VIRTUAL lengths ---
        self.directed_bonds = []
        self.directed_bond_index = {}
        self.directed_lengths = {}

        for idx, (a, b) in enumerate(self.updated_bonds):
            # forward direction
            self.directed_bonds.append((a, b))
            self.directed_bond_index[(a, b)] = idx
            # reverse direction
            self.directed_bonds.append((b, a))
            self.directed_bond_index[(b, a)] = idx + self.n_internal_bonds

            # assign length from VIRTUAL lengths
            key = tuple(sorted([a, b]))
            L = self.virtual_lengths[key]
            self.directed_lengths[(a, b)] = L
            self.directed_lengths[(b, a)] = L

        # --- STEP 8: Name & save directory (new logic) ---
        self.name = name
    
        if save_path is None:
            # Default: save under CWD / {name} / _build_save_dir_name()
            base_dir = Path.cwd()
            graph_dir = base_dir / self.name
            coupling_info = getattr(self, "coupling_info", None)  # safe hook for subclasses
            leaf = self._build_save_dir_name(coupling_info)
            self.save_path = graph_dir / leaf
        else:
            # User override: use the provided path directly
            self.save_path = Path(save_path)
    
        # Creation of the saving directory
        self.save_path.mkdir(parents=True, exist_ok=True)
    
        # --- STEP 9: Save parameters ---
        self._save_parameters()

    
    # =========================
    # Spacing helpers (new)
    # =========================
    def _init_spacing_sequence(self, spacing):
        """
        Normalize `virtual_vertex_spacing`.
    
        - float/int -> use as a single spacing value for ALL multi-bond groups
        - list/tuple/ndarray -> cycle values per multi-bond group
        - None/empty -> fall back to geometry-based default in _break_complex_bond
        """
        self._spacing_sequence = None
        self._spacing_ptr = 0
    
        if spacing is None:
            return
    
        if isinstance(spacing, (int, float)):
            # keep self.virtual_vertex_spacing as provided (float)
            return
    
        # list/tuple/array: store a float-cast sequence for cycling
        seq = list(spacing)
        if len(seq) == 0:
            return
        self._spacing_sequence = [float(x) for x in seq]
    
    
    def _next_group_spacing(self):
        """
        Return the spacing for the next *multi-bond group*.
    
        - If a sequence was provided, cycle through it.
        - Else, return the single float in `virtual_vertex_spacing` (may be None).
        """
        if self._spacing_sequence:
            val = self._spacing_sequence[self._spacing_ptr % len(self._spacing_sequence)]
            self._spacing_ptr += 1
            return val
        return self.virtual_vertex_spacing  # float or None
    
    
    def _break_complex_bond(self, bond, loop=False, index=0, n_total=1, group_spacing=None):
        """
        Replace a multiple bond or self-loop with virtual vertices and update virtual_port_map.
    
        Notes
        -----
        - ORIGINAL lengths are preserved in self.lengths.
        - This method ONLY mutates self.virtual_lengths, self.virtual_port_map,
          self.vertex_positions, self.virtual_vertices, and self.updated_bonds.
        """
        (i, pi), (j, pj) = bond
        xi, yi = self.vertex_positions[i]
        xj, yj = self.vertex_positions[j]
    
        dx, dy = xj - xi, yj - yi
        distance = np.hypot(dx, dy)
        norm_dir = np.array([dx, dy]) / distance if distance > 0 else np.array([1.0, 0.0])
        perp = np.array([-norm_dir[1], norm_dir[0]])
    
        # --- unified spacing choice ---
        if group_spacing is not None:
            spacing = float(group_spacing)
        elif isinstance(self.virtual_vertex_spacing, (int, float)):
            spacing = float(self.virtual_vertex_spacing)
        else:
            spacing = 0.3 * distance
    
        # ================================================================
        # CASE 1: LOOP (i == j)
        # ================================================================
        if loop:
            # Retrieve and remove the old loop length from VIRTUAL lengths
            L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)
    
            # Remove the original self-connection from virtual_port_map
            self.virtual_port_map[i][pi] = None
            self.virtual_port_map[i][pj] = None
    
            # Place 3 virtual vertices to form a ring (square/diamond-like)
            center = np.array([xi, yi])
            offset = L_total / 4.0 / np.sqrt(2)
    
            v_positions = [
                (-offset,  offset),   # v1
                ( 0.0,     offset*2), # v2
                ( offset,  offset)    # v3
            ]
    
            v_idx_list = []
            for (dx_v, dy_v) in v_positions:
                v_idx = len(self.vertex_positions)
                self.vertex_positions.append((center[0] + dx_v, center[1] + dy_v))
                self.valencies.append(2)
                self.virtual_vertices.append(v_idx)
                self.virtual_port_map.append([None, None])
                v_idx_list.append(v_idx)
    
            v1, v2, v3 = v_idx_list
    
            # Correct mapping: start at (i,pi) and return to (i,pj)
            self.virtual_port_map[i][pi] = (v1, 0)
            self.virtual_port_map[i][pj] = (v3, 1)
    
            self.virtual_port_map[v1][0] = (i, pi)
            self.virtual_port_map[v1][1] = (v2, 0)
            self.virtual_port_map[v2][0] = (v1, 1)
            self.virtual_port_map[v2][1] = (v3, 0)
            self.virtual_port_map[v3][0] = (v2, 1)
            self.virtual_port_map[v3][1] = (i, pj)
    
            # Add new bonds with equal lengths (L_total/4)
            segment_length = L_total / 4.0
            new_bonds = [
                ((i,  pi), (v1, 0)),
                ((v1, 1), (v2, 0)),
                ((v2, 1), (v3, 0)),
                ((v3, 1), (i,  pj))
            ]
    
            for (a, b) in new_bonds:
                self.updated_bonds.append((a, b))
                self.virtual_lengths[tuple(sorted([a, b]))] = segment_length
    
            return
    
        # ================================================================
        # CASE 2: MULTIPLE BONDS (i ≠ j)
        # ================================================================
        # Retrieve and remove original bond length from VIRTUAL lengths
        L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)
    
        # Midpoint + perpendicular offset (fan-out)
        midpoint = (np.array([xi, yi]) + np.array([xj, yj])) / 2
        offset = (index - (n_total - 1) / 2) * spacing
        pt = midpoint + offset * perp
    
        # Create one virtual vertex
        v_idx = len(self.vertex_positions)
        self.vertex_positions.append(tuple(pt))
        self.valencies.append(2)
        self.virtual_vertices.append(v_idx)
    
        # Add empty ports for virtual vertex
        self.virtual_port_map.append([None, None])
    
        # Remove direct connection (i, pi) ↔ (j, pj)
        self.virtual_port_map[i][pi] = None
        self.virtual_port_map[j][pj] = None
    
        # Connect (i, pi) → (v_idx, 0) and (j, pj) → (v_idx, 1)
        self.virtual_port_map[i][pi] = (v_idx, 0)
        self.virtual_port_map[v_idx][0] = (i,  pi)
        self.virtual_port_map[j][pj] = (v_idx, 1)
        self.virtual_port_map[v_idx][1] = (j,  pj)
    
        # Split the original bond into two equal segments
        segment_length = L_total / 2.0
        self.updated_bonds.append(((i, pi), (v_idx, 0)))
        self.updated_bonds.append(((v_idx, 1), (j, pj)))
    
        self.virtual_lengths[tuple(sorted([(i, pi), (v_idx, 0)]))] = segment_length
        self.virtual_lengths[tuple(sorted([(v_idx, 1), (j, pj)]))] = segment_length

            
    def _order_bonds_and_lengths(self):
        """
        Canonicalize and globally order self.updated_bonds and self.virtual_lengths.
    
        - Each bond is stored as ((i,pi),(j,pj)) with (i,pi) <= (j,pj).
        - self.updated_bonds is sorted by (i,pi) then (j,pj).
        - self.virtual_lengths is reinserted to follow the same order.
    
        NOTE: self.lengths (original lengths) remains untouched.
        """
        from collections import OrderedDict
    
        def canon_edge(a, b):
            # Put the smaller endpoint first, lexicographically on (i, pi)
            return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)
    
        # 1) Canonicalize every bond ((i,pi),(j,pj))
        canon_bonds = [canon_edge(a, b) for (a, b) in self.updated_bonds]
    
        # 2) Global sort by first endpoint, then second
        canon_bonds.sort(key=lambda ab: (ab[0][0], ab[0][1], ab[1][0], ab[1][1]))
    
        # 3) Rebuild VIRTUAL lengths dict to match this order (dict preserves insertion order)
        ordered_lengths = OrderedDict()
        for a, b in canon_bonds:
            key = tuple(sorted([a, b]))  # undirected key convention
            if key not in self.virtual_lengths:
                raise KeyError(f"Missing virtual length for bond {key}.")
            if key not in ordered_lengths:    # keep first occurrence if duplicates appear
                ordered_lengths[key] = self.virtual_lengths[key]
    
        # (Optional) Append any leftover keys defensively (shouldn’t happen)
        for k, v in self.virtual_lengths.items():
            if k not in ordered_lengths:
                ordered_lengths[k] = v
    
        # Commit
        self.updated_bonds = canon_bonds
        self.virtual_lengths = dict(ordered_lengths)

                    
    def _compute_group_index(self):
        """
        Compute the group index ng(k) from neff(k) if ng is not provided.
    
        Returns
        -------
        float or callable
            - If neff is constant → returns the constant value.
            - If neff is callable → returns a function ng(k).
        """

        # Case 1: neff is a constant
        if not callable(self.neff):
            return float(self.neff)
    
        # Case 2: neff is a function → derive ng(k)
        def ng_func(k):
            # numdifftools derivative of neff(k) wrt k
            d_neff = nd.Derivative(self.neff)(k)
            return self.neff(k) - k * d_neff
    
        return ng_func
    
    # =========================
    # Saving helpers
    # =========================

    def _save_parameters(self):
        """
        Save a complete description of the closed quantum graph to a .txt file.
    
        Contents:
          - General metadata (name, date, wavelength range, total/virtual vertices, bonds)
          - Effective index (constant value or callable name + source if available)
          - Group index (if callable, note it; if constant, print value)
          - Explicit port_map (as given at init, before virtualization)
          - Explicit lengths dict (as given or computed, before virtualization)
          - Vertex scattering matrices (numeric if array, otherwise note callable)
          - Virtualization info (number of virtual vertices, etc.)
        """
        filepath = os.path.join(self.save_path, "graph_parameters.txt")
        with open(filepath, "w") as f:
            # ---- Header and metadata ----
            f.write("### Closed Quantum Graph Parameters ###\n")
            f.write(f"Graph Name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n")
            f.write(f"Total vertices: {self.n_vertices_total} "
                    f"(virtual: {self.n_virtual_vertices})\n")
            f.write(f"Internal bonds: {self.n_internal_bonds}\n")
            f.write(f"Wavelength range (vacuum): {self.wavelength_range[0]:.3f} – "
                    f"{self.wavelength_range[1]:.3f} µm\n\n")
    
            # ---- Effective index info ----
            if callable(self.neff):
                f.write("Effective index: function of k\n")
                try:
                    f.write(inspect.getsource(self.neff) + "\n")
                except Exception:
                    f.write("(source code unavailable)\n")
            else:
                f.write(f"Effective index: {float(self.neff):.4f}\n")
    
            # ---- Group index info ----
            if callable(self.ng):
                f.write("\nGroup index: function of k (derived from neff)\n")
            else:
                f.write(f"\nGroup index: {float(self.ng):.4f}\n")
    
            # ---- Vertex scattering matrices ----
            f.write("\n### Vertex Scattering Matrices ###\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v}: function of k\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source code unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency {self.valencies[v]}):\n")
                    np.savetxt(f, sigma, fmt="%-8.4f", delimiter=" ", newline="\n")
                    f.write("\n")
    
            # ---- Explicit Port Map ----
            f.write("\n### Port Map (raw, before virtualization) ###\n")
            for i, ports in enumerate(self.port_map):
                f.write(f"Vertex {i} ports: {ports}\n")
    
            # ---- Explicit Lengths ----
            f.write("\n### Bond Lengths (raw, before virtualization) ###\n")
            for (a, b), L in self.lengths.items():
                f.write(f"{a} — {b} : {L:.6f} µm\n")
    
            # ---- Virtualization summary ----
            f.write("\n### Virtualization Info ###\n")
            f.write(f"Virtual vertices: {self.virtual_vertices}\n")
            f.write(f"Updated bonds (after virtualization): {self.updated_bonds}\n")
    
        print(f"Graph parameters saved to {filepath}")

    
    def _build_save_dir_name(self, coupling_info=None):
        """
        Build the default save directory name based on graph metadata.
    
        The name is deterministic: same inputs always produce the same folder.
        No timestamps or random suffixes are added.
    
        Parameters
        ----------
        coupling_info : str or None
            Additional string describing coupling constants (for subclasses like
            DirectionalGraphs). Ignored if None.
    
        Returns
        -------
        str
            Save directory name.
        """
        # --- Base info string (vertices and bonds) ---
        info_str = f"Nv={self.n_vertices_total}_Nb={self.n_internal_bonds}"
    
        # --- Effective index ---
        if callable(self.neff):
            neff_str = self.neff.__name__ if hasattr(self.neff, "__name__") else "neff_func"
        else:
            neff_str = f"neff={float(self.neff):.3f}"
    
        # --- Wavelength range ---
        lamb_min, lamb_max = self.wavelength_range
        lamb_str = f"lambda={lamb_min:.3f}-{lamb_max:.3f}"
    
        # --- Hash to ensure uniqueness for given connectivity/lengths ---
        hash_input = str(self.port_map) + str(self.lengths)
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
        # --- Assemble final string ---
        if coupling_info:
            return f"{info_str}_{neff_str}_{lamb_str}_{coupling_info}_{hash_digest}"
        else:
            return f"{info_str}_{neff_str}_{lamb_str}_{hash_digest}"


    # ========================================================================
    #         Computation of the propagation and scattering matrices
    # ========================================================================
    def _build_propagation_matrix(self, k):
        """
        Build the diagonal propagation matrix D(k) for all directed bonds.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber (in µm^-1).
    
        Returns
        -------
        D : ndarray (2B × 2B)
            Diagonal matrix with entries exp(i * k * n_eff * L).
        """
        # Resolve n_eff (allow callable neff(k) or constant float)
        n_eff = self.neff(np.real(k)) if callable(self.neff) else self.neff
    
        # Allocate diagonal matrix
        D = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)
    
        # Fill using the canonical directed-bond indexing
        for dbond, idx in self.directed_bond_index.items():
            # dbond is a directed bond key like ((i, p), (j, q))
            L = self.directed_lengths[dbond]  # length associated with that directed bond
            D[idx, idx] = np.exp(1j * k * n_eff * L)
    
        return D

    def _build_sigma_alpha_alpha(self, k):
        """
        Build the internal scattering block Σ_{αα}.

        This submatrix describes how waves scatter between internal directed bonds
        at each vertex (no leads in ClosedGraphs).

        Parameters
        ----------
        k : float
            Vacuum wavenumber (µm^-1).

        Returns
        -------
        Sigma_aa : ndarray (2B × 2B)
        """
        Sigma_aa = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)

        # Loop over all directed bonds
        for dbond_in in self.directed_bonds:
            src, dst = dbond_in        # directed bond: src → dst
            v = dst[0]                 # vertex where the wave arrives
            port_in = dst[1]           # incoming port at vertex v

            # Retrieve σ_v (evaluate if callable)
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(k)

            idx_in = self.directed_bond_index[dbond_in]

            # Scatter from port_in to all outgoing ports of v
            for port_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][port_out]

                # In ClosedGraphs, every port connects to another internal bond
                neighbor, n_port = conn
                dbond_out = ((v, port_out), (neighbor, n_port))

                idx_out = self.directed_bond_index[dbond_out]

                # Fill Σ_{αα}[out, in]
                Sigma_aa[idx_out, idx_in] = sigma_v[port_out, port_in]

        return Sigma_aa
    
    # ========================================================================
    #            Computation of the spectrum and the wavefunctions
    # ========================================================================
    def _nullity_at_k(self, k, tol_abs=1e-4):
        """
        Return the nullity (dimension of ker(I - U(k))) at a given wavenumber.
    
        Parameters
        ----------
        k : float
            Wavenumber (µm⁻¹).
        tol_abs : float
            Absolute threshold: singular values <= tol_abs are treated as zero.
    
        Returns
        -------
        int
            Dimension of the kernel of (I - U(k)).
        """
        D_k = self._build_propagation_matrix(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        U_k = D_k @ Sigma_aa
        A = np.eye(self.n_directed_bonds, dtype=complex) - U_k
        s = np.linalg.svd(A, compute_uv=False)
        return int(np.sum(s <= tol_abs))
    
    def secular_det(self, k):
        """
        Compute the secular determinant of the open quantum graph.
    
        The secular determinant is defined as:
            f(k) = det(I - D(k) Σ_{αα}(k))
    
        where:
        - Σ_{αα}(k) is the internal scattering block
        - D(k) is the diagonal propagation matrix for all directed bonds.
    
        Parameters
        ----------
        k : complex or float
            Complex vacuum wavenumber (µm⁻¹).
    
        Returns
        -------
        det_val : complex
            Value of the secular determinant f(k).
        """
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        D = self._build_propagation_matrix(k)
        U_tilde = D @ Sigma_aa
        return np.linalg.det(np.eye(self.n_directed_bonds) - U_tilde)
    
    def compute_spectrum(self, N_scan=1_000_000, tol=1e-16,
                         show_progress=False, save_spectrum=True, 
                         overwrite=False, maxiter=500, nullity_tol=1e-4):
        """
        Closed-graph spectrum on the real axis via σ_min(I - U(k)), plus degeneracies.
    
        We locate k where the internal map U(k) has an eigenvalue 1 by minimizing
        g(k) = σ_min(I - U(k)) in candidate cells from a coarse scan. For each accepted
        root k*, the degeneracy is computed as the nullity of (I - U(k*)) using SVD
        with absolute threshold `nullity_tol`.
    
        Parameters
        ----------
        N_scan : int
            Number of coarse sampling points between k_min and k_max.
        tol : float
            Target x-accuracy for bounded refinements (options['xatol']).
        show_progress : bool
            Print coarse-scan progress and summary.
        save_spectrum : bool
            Save results to disk.
        overwrite : bool
            Recompute even if a matching file already exists.
        maxiter : int
            Max iterations for each bounded refinement.
        nullity_tol : float
            Absolute SVD threshold for degeneracy detection in _nullity_at_k.
    
        Returns
        -------
        spectrum : np.ndarray
            Sorted array of real eigen-wavenumbers in [k_min, k_max].
        degeneracies : np.ndarray (int)
            Same length as spectrum; degeneracy of each eigenvalue.
        """
    
        # ---- 1) Range from wavelength_range -----------------------------------
        lam_min, lam_max = self.wavelength_range
        k_min = 2*np.pi / lam_max   # longer λ → smaller k
        k_max = 2*np.pi / lam_min   # shorter λ → larger  k
    
        # ---- 2) Early load if present -----------------------------------------
        base = (f"spectrum_sigmaMin_lamMin_{lam_min:.3f}_lamMax_{lam_max:.3f}"
                f"_N{N_scan}_tol{tol:.0e}_maxiter{maxiter:.0f}"
                f"_nulltol{nullity_tol:.0e}")
        filename_npz = f"{base}.npz"
        filepath_npz = os.path.join(self.save_path, filename_npz)
    
        if save_spectrum and (not overwrite) and os.path.exists(filepath_npz):
            if show_progress:
                print(f"✓ Spectrum+degeneracies loaded from file: {filepath_npz}")
            data = np.load(filepath_npz)
            spectrum = np.array(data["spectrum"], dtype=float)
            degeneracies = np.array(data["degeneracies"], dtype=int)
            self.spectrum = spectrum
            self.degeneracies = degeneracies
            self.degenerate_spectrum = np.repeat(spectrum, degeneracies)
    
            # Warn if some saved modes have zero degeneracy (likely too-tight nullity_tol
            # or insufficient refinement tolerance in the run that produced the file).
            zero_idx = np.nonzero(degeneracies == 0)[0]
            if zero_idx.size > 0:
                ks_bad = ", ".join(f"{spectrum[i]:.8g}" for i in zero_idx[:5])
                tail = "" if zero_idx.size <= 5 else f" (and {zero_idx.size-5} more)"
                warnings.warn(
                    "Some loaded modes have nullity 0. Consider loosening nullity_tol "
                    f"(e.g., {nullity_tol:.1e} → {max(nullity_tol*3, 1e-12):.1e}) or refining k further "
                    f"(smaller tol). Examples: k = {ks_bad}{tail}.",
                    RuntimeWarning
                )
            return spectrum, degeneracies
    
        if show_progress:
            print("Starting closed-graph spectrum via σ_min(I-U(k)) ...")
    
        # ---- Helper: U(k) = D(k) Σ(k) -----------------------------------------
        def _U(k):
            Sigma = self._build_sigma_alpha_alpha(k)   # unitary vertex coupling block
            D = self._build_propagation_matrix(k)      # diagonal phase matrix
            return D @ Sigma
    
        # ---- Helper: g(k) = σ_min(I - U(k)) -----------------------------------
        def _sigma_min(k):
            try:
                U = _U(k)
                svals = np.linalg.svd(np.eye(U.shape[0]) - U, compute_uv=False)
                smin = float(svals[-1])
                if not np.isfinite(smin):
                    return np.inf
                return smin
            except Exception:
                return np.inf
    
        # ---- 3) Coarse scan ----------------------------------------------------
        k_vals = np.linspace(k_min, k_max, N_scan)
        g_vals = np.empty(N_scan, dtype=float)
    
        last_pct = -1
        for i, k in enumerate(k_vals):
            g_vals[i] = _sigma_min(k)
            if show_progress:
                pct = int(i * 100 / max(1, N_scan-1))
                if pct % 5 == 0 and pct != last_pct:
                    print(f"Scanning σ_min: {pct:3d}% done", end="\r")
                    last_pct = pct
        if show_progress:
            print("Scanning σ_min: 100% ✓\nFinding candidate wells...")
    
        # ---- 4) Local minima as candidates ------------------------------------
        finite_mask = np.isfinite(g_vals)
        finite_vals = g_vals[finite_mask]
        if finite_vals.size == 0:
            if show_progress:
                print("No finite values encountered; returning empty spectrum.")
            self.spectrum = np.array([])
            self.degeneracies = np.array([], dtype=int)
            self.degenerate_spectrum = np.array([])
            return self.spectrum, self.degeneracies
    
        g_floor = np.min(finite_vals)
        eps_accept = 1e-4  # accept if refined σ_min <= this
    
        if show_progress:
            print(f"   min σ_min on grid: {g_floor:.3e}")
            print(f"   accept ≤ {eps_accept:.3e}")
    
        cand_idx = []
        for i in range(1, N_scan-1):
            if (g_vals[i] < g_vals[i-1]) and (g_vals[i] < g_vals[i+1]):
                cand_idx.append(i)
    
        # ---- 5) Refine each candidate in [k_{i-1}, k_{i+1}] -------------------
        def _refine_in_cell(a, b):
            res = minimize_scalar(_sigma_min, bounds=(a, b), method='bounded',
                                  options={'xatol': tol, 'maxiter': maxiter})
            return res.x, res.fun, res.success
    
        roots = []
        for i in cand_idx:
            a, b = k_vals[i-1], k_vals[i+1]
            k_star, g_star, ok = _refine_in_cell(a, b)
            if ok and (k_min <= k_star <= k_max) and (g_star <= eps_accept):
                roots.append(k_star)
    
        if show_progress:
            print(f"Refined {len(cand_idx)} candidates → kept {len(roots)} roots.")
    
        # ---- 6) Sort roots -----------------------------------------------------
        spectrum = np.array(sorted(roots), dtype=float)
    
        # ---- 7) Compute degeneracies via nullity; warn on zeros ----------------
        if spectrum.size == 0:
            degeneracies = np.array([], dtype=int)
            deg_spectrum = np.array([], dtype=float)
        else:
            degeneracies = np.empty(spectrum.size, dtype=int)
            zero_list = []
            for idx, k in enumerate(spectrum):
                d = int(self._nullity_at_k(float(k), tol_abs=nullity_tol))
                degeneracies[idx] = d
                if d == 0:
                    zero_list.append((idx, k))
    
            if zero_list:
                # Consolidated warning with a few examples, advising how to proceed
                examples = ", ".join(f"{k:.8g}" for _, k in zero_list[:5])
                tail = "" if len(zero_list) <= 5 else f" (and {len(zero_list)-5} more)"
                warnings.warn(
                    "Some refined roots have nullity 0. This often indicates the SVD "
                    "cutoff is too strict or k needs tighter refinement. Consider "
                    f"loosening nullity_tol (e.g., {nullity_tol:.1e} → {max(nullity_tol*3, 1e-12):.1e}) "
                    f"or reducing tol (e.g., {tol:.1e} → {max(tol/10, 1e-20):.1e}). "
                    f"Examples: k = {examples}{tail}.",
                    RuntimeWarning
                )
    
            deg_spectrum = np.repeat(spectrum, degeneracies)
    
        # ---- 8) Save & store ---------------------------------------------------
        if save_spectrum:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez(filepath_npz, spectrum=spectrum, degeneracies=degeneracies)
            if show_progress:
                print(f"✓ Spectrum+degeneracies saved to {filepath_npz}")
    
        self.spectrum = spectrum
        self.degeneracies = degeneracies
        self.degenerate_spectrum = deg_spectrum
    
        return spectrum, degeneracies

    def compute_wavefunction(self, k, tol=1e-4):
        """
        Compute the basis of the kernel (null space) of A = I - U(k),
        where U(k) is the quantum evolution map for the closed graph.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber at which to compute the wavefunction.
        tol : float, optional
            Tolerance for identifying near-zero singular values (default: 1e-6).
    
        Returns
        -------
        null_space : ndarray
            Basis vectors of the kernel as rows (each row is an eigenvector).
        """
        # ---- 1) Compute the quantum evolution map U(k) -----------------------
        # U(k) = D(k) Σ_{αα}(k)
        D_k = self._build_propagation_matrix(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        U_k = D_k @ Sigma_aa
    
        # ---- 2) Build A = I - U(k) ------------------------------------------
        I = np.eye(self.n_directed_bonds, dtype=complex)
        A = I - U_k
    
        # ---- 3) Compute SVD of A --------------------------------------------
        # A = U * S * Vh; singular values in S
        U_svd, S, Vh = svd(A)
    
        # ---- 4) Identify singular values close to zero ----------------------
        null_mask = (S < tol)
    
        # Columns of Vh corresponding to singular values ~ 0 give null space
        null_space = Vh[null_mask].T
    
        # ---- 5) Logging & return --------------------------------------------
        if null_space.size == 0:
            print(f"⚠ Warning: No null space found at k = {k:.6f} within tolerance {tol}.")
        else:
            print(f"✓ Found {null_space.shape[1]} wavefunction(s) in the kernel at k = {k:.6f}.")
    
        # Each row is a wavefunction (transpose for easier iteration)
        return null_space.T

    
    # ========================================================================
    #                          Spectral statistics
    # ========================================================================
    # -------------------------------------------------------------------------
    # Spectral statistics core utilities
    # -------------------------------------------------------------------------
    
    def _is_pure_directional(self):
        """
        Return True iff degeneracy was computed and every level has deg == 2.
        Used to auto-detect fully directional closed graphs for spectral statistics.
        """
        return hasattr(self, "degeneracies") and (self.degeneracies.size > 0) and np.all(self.degeneracies == 2)

    def _stats_selection(self, mode="auto"):
        """
        Decide which spectrum and Weyl scaling to use for spectral statistics.
    
        Parameters
        ----------
        mode : {"auto", "raw", "expanded", "directional"}
            - "auto": if all deg==2, use RAW spectrum and scale Weyl by 1/2,
                      else use EXPANDED spectrum and scale Weyl by 1.
            - "raw": always use self.spectrum (no duplication); Weyl scale = 1.
            - "expanded": always use self.degenerate_spectrum; Weyl scale = 1.
            - "directional": force directional rule (raw + Weyl scale = 1/2).
    
        Returns
        -------
        spectrum_used : np.ndarray
        weyl_scale : float
            Multiply self.total_length by this factor inside Weyl.
        note : str
            A short human-readable note for logs/plots.
        method : {"raw", "expanded", "directional"}
            The actual method chosen (never "auto").
        """
        if mode == "raw":
            return (
                np.asarray(self.spectrum, float),
                1.0,
                "Using raw spectrum (no duplication).",
                "raw",
            )
    
        if mode == "expanded":
            return (
                np.asarray(self.degenerate_spectrum, float),
                1.0,
                "Using expanded (degenerate) spectrum.",
                "expanded",
            )
    
        if mode == "directional":
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run compute_spectrum() first.")
            return (
                np.asarray(self.spectrum, float),
                0.5,
                "Directional assumption: Weyl scaled by 1/2.",
                "directional",
            )
    
        # auto mode
        if self._is_pure_directional():
            return (
                np.asarray(self.spectrum, float),
                0.5,
                "Detected pure directional graph: using raw spectrum; Weyl × 1/2.",
                "directional",
            )
        else:
            if hasattr(self, "degenerate_spectrum"):
                return (
                    np.asarray(self.degenerate_spectrum, float),
                    1.0,
                    "Non-directional or mixed: using expanded spectrum.",
                    "expanded",
                )
            return (
                np.asarray(self.spectrum, float),
                1.0,
                "No degeneracy info: using raw spectrum.",
                "raw",
            )


    def N_weyl_scaled(self, k, length_scale=1.0):
        """
        Weyl staircase with an optional multiplicative scale on total length.
        Equivalent to replacing L -> length_scale * L.
    
        Parameters
        ----------
        k : float or array-like
            Real wavenumber(s) [µm⁻¹].
        length_scale : float
            Factor applied to self.total_length (default 1.0).
    
        Returns
        -------
        float or ndarray
            Predicted cumulative mode count N(k) with scaling applied.
        """
        if callable(self.neff):
            n_eff = self.neff(k)
        else:
            n_eff = self.neff
        return n_eff * (self.total_length * length_scale) / np.pi * k

    
    def N_weyl(self, k):
        """
        Compute the Weyl prediction for the staircase function N(k).
    
        Uses the known optical length of the graph (self.total_length) and the 
        group index self.ng (constant or callable).
    
        Parameters
        ----------
        k : float or array-like
            Real wavenumber(s) [µm⁻¹].
    
        Returns
        -------
        float or array-like
            Predicted cumulative mode count N(k).
        """
        # 1) ---- Evaluate neff(k) --------------------------------------------
        if callable(self.neff):
            n_eff = self.neff(k)   # vectorized if k is array
        else:
            n_eff = self.neff      # constant scalar value
    
        # 2) ---- Compute Weyl staircase --------------------------------------
        return n_eff * self.total_length / np.pi * k
    
    def _compute_unfolded_spectrum(self, spectrum=None, mode=None):
        """
        Unfold a spectrum using the Weyl law with optional directional scaling.
        If spectrum is None, select it automatically (or by 'mode').
    
        Parameters
        ----------
        spectrum : ndarray or None
            k values. If None, chooses according to _stats_selection(mode).
        mode : {"auto", "raw", "expanded", "directional", None}
            Selection policy; None -> use self.stats_mode if present, else "auto".
    
        Returns
        -------
        ndarray
            Unfolded spectrum values N_weyl_scaled(k).
        """
        if spectrum is None:
            chosen_mode = mode or getattr(self, "stats_mode", "auto")
            spectrum_used, weyl_scale, _, _ = self._stats_selection(mode=chosen_mode)
            spectrum = np.asarray(spectrum_used, dtype=float)
        else:
            # Spectrum provided explicitly: do not auto-switch which set to use,
            # but still honor 'mode' for the Weyl scaling (directional vs not).
            spectrum = np.asarray(spectrum, dtype=float)
            chosen_mode = mode or getattr(self, "stats_mode", "auto")
            _, weyl_scale, _, _ = self._stats_selection(mode=chosen_mode)
    
        spectrum = np.sort(np.real(spectrum))
        return self.N_weyl_scaled(spectrum, length_scale=weyl_scale)

    
    # -------------------------------------------------------------------------
    # Staircase function plotting
    # -------------------------------------------------------------------------
    def plot_staircase_function(self, plot_weyl=True, weyl_diff=False,
                                mode=None, wavelength_ranges=None, show_legend=False,
                                save_fig=False, file_format='pdf'):
        """
        Plot the staircase N(k) for spectral statistics, with automatic handling
        of directional graphs (deg=2 everywhere): in that case, use the raw spectrum
        (no duplication) and scale the Weyl law by 1/2.
    
        NEW:
        ----
        wavelength_ranges : None | (lam_min, lam_max) | list of (lam_min, lam_max)
            Wavelength ranges in microns (vacuum) used to select subspectra.
            - None (default): use the full selected spectrum (after _stats_selection).
            - Tuple (min, max): use a single subset with lam_min ≤ λ ≤ lam_max.
            - List of tuples: build one subspectrum per tuple and overlay them.
    
        show_legend : bool, default False
            When True and wavelength_ranges specifies multiple ranges (i.e., we plot
            multiple subsets), display legends on the figures. Otherwise, no legends.
    
        Title and filenames include the ACTUAL method used among:
          - "raw"
          - "expanded"
          - "directional"
        even if mode="auto".
    
        Parameters
        ----------
        plot_weyl : bool
            Overlay the (scaled) Weyl prediction curve.
        weyl_diff : bool
            If True, also plot N_Weyl(k) - N(k) vs index and vs wavelength (nm).
        mode : {"auto", "raw", "expanded", "directional", None}
            Selection policy; None -> use self.stats_mode if present, else "auto".
            - "auto": if all deg==2, use RAW spectrum and scale Weyl by 1/2,
                      else use EXPANDED spectrum and scale Weyl by 1.
            - "raw": always use self.spectrum (no duplication).
            - "expanded": always use self.degenerate_spectrum.
            - "directional": force directional rule (raw + Weyl × 1/2).
        save_fig : bool
            Save figure(s) to self.save_path.
        file_format : str
            Format for saving ('pdf', 'svg', etc.). Default: 'pdf'.
        """
    
        # --- 1) Choose spectrum + Weyl scaling ---------------------------------
        if not hasattr(self, 'spectrum'):
            raise AttributeError('No spectrum found. Run compute_spectrum() first.')
    
        chosen_mode = mode or getattr(self, 'stats_mode', 'auto')
    
        # EXPECTED: _stats_selection returns (spectrum_used, weyl_scale, note, method)
        spectrum_used, weyl_scale, note, method = self._stats_selection(mode=chosen_mode)
    
        spectrum_full = np.sort(np.real(np.asarray(spectrum_used, dtype=float)))
        if spectrum_full.size == 0:
            raise ValueError('Empty spectrum after selection.')
    
        print(f'[Staircase:{method}] {note}')
    
        # Helper: convert wavelength ranges specification to a normalized list
        def _normalize_ranges(rspec):
            if rspec is None:
                return None
            if isinstance(rspec, tuple) and len(rspec) == 2:
                return [tuple(rspec)]
            if isinstance(rspec, list):
                out = []
                for item in rspec:
                    if not (isinstance(item, tuple) and len(item) == 2):
                        raise ValueError('Each wavelength range must be a (min, max) tuple in µm.')
                    out.append(tuple(item))
                return out
            raise ValueError('wavelength_ranges must be None, a (min, max) tuple, or a list of such tuples (in µm).')
    
        ranges_list = _normalize_ranges(wavelength_ranges)
    
        # Helper: format ranges for legend and filename tags
        def _fmt_range_for_legend(r):
            a, b = r
            return fr'$\lambda \in [{a:.3f},\,{b:.3f}]\,\mu m$'
    
        def _fmt_ranges_for_filename(rlist):
            if rlist is None:
                return ''
            tags = []
            for a, b in rlist:
                tags.append(f'[{a:.3f}-{b:.3f}]')
            return '__ranges_' + '_'.join(tags)
    
        # Build the list of subsets to plot
        subsets = []
    
        if ranges_list is None:
            # Single full-spectrum subset
            subsets.append({
                'k': spectrum_full,
                'label': r'$\lambda$: full',
            })
        else:
            # Filter by λ = 2π / k (µm), inclusive bounds
            lam_full = 2 * np.pi / spectrum_full  # µm
            for (lam_min, lam_max) in ranges_list:
                if not (np.isfinite(lam_min) and np.isfinite(lam_max) and lam_min < lam_max):
                    print(f'⚠️ Skipping invalid range (lam_min={lam_min}, lam_max={lam_max}).')
                    continue
                mask = (lam_full >= lam_min) & (lam_full <= lam_max)
                k_subset = spectrum_full[mask]
                if k_subset.size == 0:
                    print(f'⚠️ Range [{lam_min}, {lam_max}] µm yields no modes; skipping.')
                    continue
                subsets.append({
                    'k': np.sort(k_subset),
                    'label': _fmt_range_for_legend((lam_min, lam_max)),
                })
    
            if len(subsets) == 0:
                raise ValueError('All provided wavelength ranges are empty after filtering; nothing to plot.')
    
        # --- 2) Main staircase plot --------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Color handling: black for single subset, distinct colors for multiple subsets
        multiple = (len(subsets) > 1)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    
        global_k_min = np.inf
        global_k_max = -np.inf
        global_N_max = 0
    
        for i, ss in enumerate(subsets):
            k = ss['k']
            N = k.size
            global_N_max = max(global_N_max, N)
            if N == 0:
                continue
    
            # Step data
            N_k = np.arange(N + 1)
            plot_k = np.insert(k, 0, k[0])  # step starts at first k
    
            # Assign color
            if multiple:
                color = color_cycle[i % len(color_cycle)] if color_cycle else None
            else:
                color = 'k'
            ss['color'] = color
    
            # Staircase
            ax.step(plot_k, N_k, where='post', label=ss['label'] if multiple else r'$N(k)$',
                    color=color, lw=2)
    
            # Optional Weyl overlay for this subset with proper anchoring
            if plot_weyl:
                k_vals = np.linspace(k[0], k[-1], 1000) if N > 1 else np.array([k[0], k[0] + 1e-9])
                Nw = self.N_weyl_scaled(k_vals, length_scale=weyl_scale)
                # Anchor: shift so that N_weyl aligns with first step (i.e., 0.5 at k[0])
                Nw -= self.N_weyl_scaled(k[0], length_scale=weyl_scale)
                Nw += 0.5
                # Always red dashed for Weyl, even in multi-range mode
                ax.plot(k_vals, Nw, 'r--', lw=1.5,
                        label=None if (multiple or not show_legend) else r'$N_{\mathrm{Weyl}}(k)$')
    
            global_k_min = min(global_k_min, k[0])
            global_k_max = max(global_k_max, k[-1])
    
        # Titles & labels
        L_eff = self.total_length * weyl_scale
        ax.set_title(fr'$L_{{\rm eff}} = {L_eff:.2f}\ \mu m$   ({method})', fontsize=20)
        ax.set_xlabel(r'$k$ (µm$^{-1}$)', fontsize=22)
        ax.set_ylabel(r'$N(k)$', fontsize=22)
        ax.set_xlim(global_k_min, global_k_max)
        ax.set_ylim(0, global_N_max)
    
        # Legend only if multiple subsets AND user asked for it
        if multiple and show_legend:
            ax.legend(fontsize=14)
    
        # Style
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=18, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
        fig.tight_layout()
    
        # Save figure (filenames embed ACTUAL method, and ranges if any)
        ranges_tag = _fmt_ranges_for_filename(ranges_list)
        if save_fig:
            base = f'staircase_function__{method}{ranges_tag}.{file_format}'
            filepath = os.path.join(self.save_path, base)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f'✓ Staircase plot saved to: {filepath}')
    
        plt.show()
    
        # --- 3) Optional Weyl deviation plots -----------------------------------
        if weyl_diff:
            # (a) Δ vs mode index
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            have_any = False
            for ss in subsets:
                k = ss['k']
                N = k.size
                if N == 0:
                    continue
                have_any = True
    
                # Δ = N_Weyl(k) - N(k) with subset-specific anchoring
                N_weyl_k = self.N_weyl_scaled(k, length_scale=weyl_scale)
                N_weyl_k -= self.N_weyl_scaled(k[0], length_scale=weyl_scale)
                N_weyl_k += 0.5
                N_k_mid = np.arange(1, N + 1)
                delta = N_weyl_k - N_k_mid
    
                ax2.plot(np.arange(N), delta, '-', lw=2, color=ss['color'], label=ss['label'])
    
            if have_any:
                ax2.axhline(0, color='gray', linestyle='--')
                ax2.set_xlabel(r'Mode index', fontsize=22)
                ax2.set_ylabel(r'$N_{\mathrm{Weyl}}(k) - N(k)$', fontsize=22)
                ax2.set_xlim(0, max((ss['k'].size for ss in subsets), default=1) - 1)
    
                if multiple and show_legend:
                    ax2.legend(fontsize=14)
    
                ax2.minorticks_on()
                ax2.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                                labelsize=18, top=True, right=True)
                ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                                top=True, right=True)
                fig2.tight_layout()
    
                if save_fig:
                    base2 = f'staircase_diff_index__{method}{ranges_tag}.{file_format}'
                    filepath2 = os.path.join(self.save_path, base2)
                    fig2.savefig(filepath2, format=file_format, dpi=300, bbox_inches='tight')
                    print(f'✓ Staircase diff (index) saved to: {filepath2}')
                plt.show()
    
            # (b) Δ vs wavelength (nm)
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            have_any = False
            for ss in subsets:
                k = ss['k']
                N = k.size
                if N == 0:
                    continue
                have_any = True
    
                N_weyl_k = self.N_weyl_scaled(k, length_scale=weyl_scale)
                N_weyl_k -= self.N_weyl_scaled(k[0], length_scale=weyl_scale)
                N_weyl_k += 0.5
                N_k_mid = np.arange(1, N + 1)
                delta = N_weyl_k - N_k_mid
    
                lambda_nm = 2 * np.pi / k * 1e3  # nm
                ax3.plot(lambda_nm, delta, '-', lw=2, color=ss['color'], label=ss['label'])
    
            if have_any:
                ax3.axhline(0, color='gray', linestyle='--')
                ax3.set_xlabel(r'Wavelength $\lambda$ (nm)', fontsize=22)
                ax3.set_ylabel(r'$N_{\mathrm{Weyl}}(k) - N(k)$', fontsize=22)
                # x-limits: union across subsets
                all_lams = np.concatenate([2 * np.pi / ss['k'] * 1e3 for ss in subsets if ss['k'].size > 0])
                ax3.set_xlim(np.min(all_lams), np.max(all_lams))
    
                if multiple and show_legend:
                    ax3.legend(fontsize=14)
    
                ax3.minorticks_on()
                ax3.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                                labelsize=18, top=True, right=True)
                ax3.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                                top=True, right=True)
                fig3.tight_layout()
    
                if save_fig:
                    base3 = f'staircase_diff_lambda__{method}{ranges_tag}.{file_format}'
                    filepath3 = os.path.join(self.save_path, base3)
                    fig3.savefig(filepath3, format=file_format, dpi=300, bbox_inches='tight')
                    print(f'✓ Staircase diff (λ) saved to: {filepath3}')
                plt.show()






    # -------------------------------------------------------------------------
    # Short-range correlations: Nearest Neighbor Spacing Distribution (NNSD)
    # -------------------------------------------------------------------------
    def short_range_correlations(self, width=0.1, xlim=None,
                                 plot_wigner=['poissonian', 'goe'],
                                 cumulative_spacing=False,
                                 color_list=None,
                                 mode=None, wavelength_ranges=None, show_legend=False,
                                 save_fig=False, file_format='pdf'):
        """
        Plot nearest-neighbor spacing distribution (NNSD) from the unfolded spectrum,
        respecting the directional-graph logic via _stats_selection():
          - pure directional (all deg==2): use raw spectrum and Weyl × 1/2
          - otherwise: use expanded (degenerate) spectrum and Weyl × 1
    
        NEW:
        ----
        wavelength_ranges : None | (lam_min, lam_max) | list of (lam_min, lam_max)
            Wavelength ranges in microns (vacuum) used to select subspectra.
            - None (default): use the full selected spectrum (after _stats_selection).
            - Tuple (min, max): use a single subset with lam_min ≤ λ ≤ lam_max.
            - List of tuples: build one subspectrum per tuple and overlay them.
    
        show_legend : bool, default False
            When True and wavelength_ranges specifies multiple ranges (i.e., we plot
            multiple subsets), display legends on the figures. Otherwise, no legends.
    
        Parameters
        ----------
        width : float
            Histogram bin width.
        xlim : tuple or None
            X-axis limits for spacing histogram. If None, computed automatically.
        plot_wigner : list of str
            Distributions to overlay: 'poissonian', 'goe', 'gue', 'gse'.
        cumulative_spacing : bool
            If True, adds a subplot for the cumulative distribution function (CDF).
        color_list : list of str or None
            Custom style list for Wigner curves (e.g., ['r-', 'b--']).
        mode : {"auto", "raw", "expanded", "directional", None}
            Selection policy; None -> use self.stats_mode if present, else "auto".
        save_fig : bool
            If True, saves the figure in self.save_path.
        file_format : str
            Output format for the saved figure. Default: 'pdf'.
        """
    
        # 1) ---- Select spectrum + scaling --------------------------------------
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum found. Run compute_spectrum() first.")
    
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
    
        # _stats_selection returns (spectrum_used, weyl_scale, note, method)
        spectrum_used, weyl_scale, note, method = self._stats_selection(mode=chosen_mode)
    
        spectrum_used = np.sort(np.real(np.asarray(spectrum_used, dtype=float)))
        if spectrum_used.size == 0:
            raise ValueError("Empty spectrum after selection.")
        print(f"[NNSD:{method}] {note}")
    
        # ---- Normalize wavelength range spec -----------------------------------
        def _normalize_ranges(rspec):
            if rspec is None:
                return None
            if isinstance(rspec, tuple) and len(rspec) == 2:
                return [tuple(rspec)]
            if isinstance(rspec, list):
                out = []
                for item in rspec:
                    if not (isinstance(item, tuple) and len(item) == 2):
                        raise ValueError("Each wavelength range must be a (min, max) tuple in µm.")
                    out.append(tuple(item))
                return out
            raise ValueError("wavelength_ranges must be None, a (min, max) tuple, or a list of such tuples (in µm).")
    
        ranges_list = _normalize_ranges(wavelength_ranges)
    
        def _fmt_range_for_legend(r):
            a, b = r
            return fr'$\lambda \in [{a:.3f},\,{b:.3f}]\,\mu m$'
    
        def _fmt_ranges_for_filename(rlist):
            if rlist is None:
                return ""
            tags = [f"[{a:.3f}-{b:.3f}]" for (a, b) in rlist]
            return "__ranges_" + "_".join(tags)
    
        # 2) ---- Build subsets (per wavelength range) ---------------------------
        # Each subset will hold: {'k': array, 'label': str, 'color': str, 'spacings': array}
        subsets = []
        if ranges_list is None:
            subsets.append({'k': spectrum_used, 'label': r'$\lambda$: full'})
        else:
            lam_full = 2 * np.pi / spectrum_used  # µm
            for (lam_min, lam_max) in ranges_list:
                if not (np.isfinite(lam_min) and np.isfinite(lam_max) and lam_min < lam_max):
                    print(f"⚠️ Skipping invalid range (lam_min={lam_min}, lam_max={lam_max}).")
                    continue
                mask = (lam_full >= lam_min) & (lam_full <= lam_max)
                k_subset = spectrum_used[mask]
                if k_subset.size < 2:
                    print(f"⚠️ Range [{lam_min}, {lam_max}] µm yields < 2 modes; skipping.")
                    continue
                subsets.append({'k': np.sort(k_subset),
                                'label': _fmt_range_for_legend((lam_min, lam_max))})
            if len(subsets) == 0:
                raise ValueError("All provided wavelength ranges are empty (or too short) after filtering; nothing to plot.")
    
        # 3) ---- Unfold and compute spacings per subset -------------------------
        # Unfolding is performed subset-wise (consistent anchoring with chosen 'method')
        for ss in subsets:
            unfolded = self._compute_unfolded_spectrum(ss['k'], mode=method)
            ss['spacings'] = np.diff(unfolded)
    
        # Determine global x-limits if not provided
        if xlim is None:
            all_spacings = np.concatenate([ss['spacings'] for ss in subsets if ss['spacings'].size > 0]) \
                           if len(subsets) else np.array([])
            xmax = float(np.max(all_spacings)) if all_spacings.size else 1.0
            xlim = (0.0, xmax)
    
        # Common bins for all subsets (for fair overlay)
        bins = np.arange(xlim[0], xlim[1] + width, width)
    
        # 4) ---- Figure setup ----------------------------------------------------
        if cumulative_spacing:
            fig, (ax, ax_cdf) = plt.subplots(2, 1, figsize=(10, 10),
                                             sharex=True,
                                             gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    
        # Color handling for subsets
        multiple = (len(subsets) > 1)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    
        # 5) ---- Plot histograms for each subset --------------------------------
        global_hist_max = 0.0
        for i, ss in enumerate(subsets):
            sp = ss['spacings']
            if sp.size == 0:
                continue
    
            # Assign color
            color = (color_cycle[i % len(color_cycle)] if (multiple and color_cycle) else 'black')
            ss['color'] = color
    
            hist_values, bin_edges = np.histogram(sp, bins=bins, density=True)
            global_hist_max = max(global_hist_max, float(np.max(hist_values)) if hist_values.size else 0.0)
    
            # Step plot
            ax.step(bin_edges[1:], hist_values, where='pre',
                    color=color, linewidth=2,
                    label=ss['label'] if (multiple and show_legend) else None)
            if hist_values.size > 0:
                ax.plot([bins[0], bins[0] + width], [hist_values[0], hist_values[0]],
                        '-', lw=2, color=color)
    
        # 6) ---- Overlay Wigner surmise distributions (unchanged behavior) ------
        s_vals = np.linspace(xlim[0], xlim[1], 1000)
        if color_list is None:
            color_list = ['r-', 'b-.', 'g--', 'orange-', 'purple--']
        wigner_labels = {'poissonian': 'Poisson', 'goe': 'GOE', 'gue': 'GUE', 'gse': 'GSE'}
    
        for i, name_func in enumerate(plot_wigner or []):
            fmt = color_list[i % len(color_list)]
            ax.plot(s_vals, NNSD_func(s_vals, wigner=name_func), fmt, lw=2,
                    label=(wigner_labels.get(name_func, name_func.upper())
                           if (multiple and show_legend) else None))
    
        # 7) ---- Style main plot -------------------------------------------------
        ax.set_ylabel(r'$P(s)$', fontsize=22)
        ax.set_xlabel('Spacing $s$', fontsize=22)
        ax.set_xlim(xlim)
        ax.set_ylim(0.0, max(global_hist_max * 1.05, 1e-9))
        if multiple and show_legend:
            ax.legend(fontsize=16, loc='upper right')
    
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
    
        # 8) ---- Cumulative subplot (optional) ----------------------------------
        if cumulative_spacing:
            # Scatter empirical CDF and overlay Wigner CDFs
            for ss in subsets:
                sp = np.sort(ss['spacings'])
                if sp.size > 0:
                    cdf_vals = np.arange(1, sp.size + 1) / sp.size
                    ax_cdf.scatter(sp, cdf_vals, marker='s', edgecolor=ss['color'],
                                   facecolor='none', s=30, linewidths=2, zorder=2,
                                   label=ss['label'] if (multiple and show_legend) else None)
    
            s_full = np.linspace(xlim[0], xlim[1], 1000)
            ds = s_full[1] - s_full[0] if s_full.size > 1 else 1.0
            for i, name_func in enumerate(plot_wigner or []):
                fmt = color_list[i % len(color_list)]
                pdf = NNSD_func(s_full, wigner=name_func)
                cdf_wigner = np.cumsum(pdf) * ds
                ax_cdf.plot(s_full, cdf_wigner, fmt, lw=2,
                            label=(wigner_labels.get(name_func, name_func.upper())
                                   if (multiple and show_legend) else None))
    
            ax_cdf.set_ylabel(r'Cumulative $P(s)$', fontsize=22)
            ax_cdf.set_xlabel('Spacing $s$', fontsize=22)
            ax_cdf.set_xlim(xlim)
            ax_cdf.set_ylim(0, 1.1)
            if multiple and show_legend:
                ax_cdf.legend(fontsize=14, loc='lower right')
    
            ax_cdf.minorticks_on()
            ax_cdf.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                               labelsize=22, top=True, right=True)
            ax_cdf.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                               top=True, right=True)
    
        fig.tight_layout()
    
        # 9) ---- Save figure(s) with ACTUAL method in filename -------------------
        ranges_tag = _fmt_ranges_for_filename(ranges_list)
        if save_fig:
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            cumulative_part = "_cumulative" if cumulative_spacing else ""
            base = f"NNSD_{wigner_part}_bin{width:.2f}{cumulative_part}__{method}{ranges_tag}.{file_format}"
            filepath = os.path.join(self.save_path, base)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ NNSD plot saved to: {filepath}")
    
        plt.show()




    # -------------------------------------------------------------------------
    # Long-range correlations: Σ²(L) and Δ₃(L)
    # -------------------------------------------------------------------------
    def long_range_correlations(self, L_max=10, L_step=0.2,
                                plot_wigner=['poissonian', 'goe'],
                                correlation_type="both",
                                color_list=None,
                                mode=None, wavelength_ranges=None, show_legend=False,
                                save_fig=False, file_format='pdf'):
        """
        Plot long-range spectral correlations: Σ²(L) and Δ₃(L), respecting the
        directional-graph logic via _stats_selection():
          - pure directional (all deg==2): use raw spectrum and Weyl × 1/2
          - otherwise: use expanded (degenerate) spectrum and Weyl × 1
    
        NEW:
        ----
        wavelength_ranges : None | (lam_min, lam_max) | list of (lam_min, lam_max)
            Wavelength ranges in microns (vacuum) to select subspectra.
            - None (default): use full selected spectrum (after _stats_selection).
            - Tuple (min, max): one subset with lam_min ≤ λ ≤ lam_max.
            - List of tuples: one subspectrum per tuple, all overlaid.
        show_legend : bool, default False
            If True and multiple wavelength ranges are provided, show legends.
            Placement rule (as requested): bottom-right; if plotting both
            Σ² and Δ₃, show legend ONLY on the lower axis (Δ₃).
    
        Parameters
        ----------
        L_max : float
            Maximum L to analyze.
        L_step : float
            Step size for computing L values.
        plot_wigner : list of str
            Wigner ensembles to overlay ('poissonian', 'goe', 'gue', 'gse').
        correlation_type : str
            One of: 'number_variance', 'spectral_rigidity', or 'both'.
        color_list : list of str or None
            Optional matplotlib styles for each Wigner curve.
        mode : {"auto", "raw", "expanded", "directional", None}
            Selection policy; None -> use self.stats_mode if present, else "auto".
        save_fig : bool
            If True, save the figure(s) to disk.
        file_format : str
            Output format for saved figure. Default: 'pdf'.
        """

        # 1) ---- Select spectrum + scaling --------------------------------------
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum found. Run compute_spectrum() first.")
    
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
    
        # _stats_selection returns (spectrum_used, weyl_scale, note, method)
        spectrum_used, weyl_scale, note, method = self._stats_selection(mode=chosen_mode)
        spectrum_used = np.sort(np.real(np.asarray(spectrum_used, dtype=float)))
        if spectrum_used.size == 0:
            raise ValueError("Empty spectrum after selection.")
        print(f"[LongRange:{method}] {note}")
    
        # ---- Normalize wavelength range spec -----------------------------------
        def _normalize_ranges(rspec):
            if rspec is None:
                return None
            if isinstance(rspec, tuple) and len(rspec) == 2:
                return [tuple(rspec)]
            if isinstance(rspec, list):
                out = []
                for item in rspec:
                    if not (isinstance(item, tuple) and len(item) == 2):
                        raise ValueError("Each wavelength range must be a (min, max) tuple in µm.")
                    out.append(tuple(item))
                return out
            raise ValueError("wavelength_ranges must be None, a (min, max) tuple, or a list of such tuples (in µm).")
    
        ranges_list = _normalize_ranges(wavelength_ranges)
    
        def _fmt_range_for_legend(r):
            a, b = r
            return fr'$\lambda \in [{a:.3f},\,{b:.3f}]\,\mu m$'
    
        def _fmt_ranges_for_filename(rlist):
            if rlist is None:
                return ""
            tags = [f"[{a:.3f}-{b:.3f}]" for (a, b) in rlist]
            return "__ranges_" + "_".join(tags)
    
        # 2) ---- Build subsets (per wavelength range) ---------------------------
        # Each subset: {'k': array, 'label': str, 'color': str, 'L_sigma':..., 'sigma2':..., 'L_delta':..., 'delta3':...}
        subsets = []
        if ranges_list is None:
            subsets.append({'k': spectrum_used, 'label': r'$\lambda$: full'})
        else:
            lam_full = 2 * np.pi / spectrum_used  # µm
            for (lam_min, lam_max) in ranges_list:
                if not (np.isfinite(lam_min) and np.isfinite(lam_max) and lam_min < lam_max):
                    print(f"⚠️ Skipping invalid range (lam_min={lam_min}, lam_max={lam_max}).")
                    continue
                mask = (lam_full >= lam_min) & (lam_full <= lam_max)
                k_subset = spectrum_used[mask]
                if k_subset.size < 2:
                    print(f"⚠️ Range [{lam_min}, {lam_max}] µm yields < 2 modes; skipping.")
                    continue
                subsets.append({'k': np.sort(k_subset),
                                'label': _fmt_range_for_legend((lam_min, lam_max))})
            if len(subsets) == 0:
                raise ValueError("All provided wavelength ranges are empty (or too short) after filtering; nothing to plot.")
    
        # 3) ---- Unfold and compute Σ²/Δ₃ per subset ----------------------------
        has_sigma = correlation_type in ["number_variance", "both"]
        has_delta = correlation_type in ["spectral_rigidity", "both"]
    
        for ss in subsets:
            unfolded = self._compute_unfolded_spectrum(ss['k'], mode=method)
            if has_sigma:
                L_sigma, sigma2_vals = self._compute_sigma2(unfolded, L_step, L_max)
                ss['L_sigma'], ss['sigma2'] = L_sigma, sigma2_vals
            if has_delta:
                L_delta, delta3_vals = self._compute_delta3(unfolded, L_step, L_max)
                ss['L_delta'], ss['delta3'] = L_delta, delta3_vals
    
        # 4) ---- Figure setup ----------------------------------------------------
        if color_list is None:
            color_list = ['r-', 'b-.', 'g--', 'orange-', 'purple--']
        wigner_labels = {'poissonian': 'Poisson', 'goe': 'GOE', 'gue': 'GUE', 'gse': 'GSE'}
    
        multiple = (len(subsets) > 1)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    
        if correlation_type == "both":
            fig, (ax_sigma, ax_delta) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={'height_ratios': [1, 1], 'hspace': 0}
            )
        else:
            fig, ax_single = plt.subplots(figsize=(10, 8))
    
        # 5) ---- Σ²(L) plot ------------------------------------------------------
        if has_sigma:
            ax_s = ax_sigma if correlation_type == "both" else ax_single
    
            # Plot all subsets (color-coded) and collect ymax like in old code
            ymax_sigma = 1.0
            all_sigma_vals = []
            for i, ss in enumerate(subsets):
                Ls = ss.get('L_sigma', np.array([]))
                vals = ss.get('sigma2', np.array([]))
                if Ls.size == 0:
                    continue
                color = (color_cycle[i % len(color_cycle)] if (multiple and color_cycle) else 'black')
                ss['color'] = color
                ax_s.scatter(Ls, vals, edgecolor=color, facecolor='white',
                             s=60, linewidth=1.5, zorder=1,
                             label=ss['label'] if (multiple and show_legend and correlation_type != "both") else None)
                if vals.size:
                    all_sigma_vals.append(vals)
    
            # Old-code y-max logic: max of data and RMT curves, then ×1.05
            L_plot = np.linspace(0.0, L_max, 1000)
            if len(all_sigma_vals):
                ymax_sigma = float(np.max(np.concatenate(all_sigma_vals)))
            else:
                ymax_sigma = 1.0
            for name in (plot_wigner or []):
                if name.lower() == "poissonian":
                    continue
                vals = sigma2_RMT(L_plot, wigner=name)
                ymax_sigma = max(ymax_sigma, float(np.max(vals)))
    
            # Draw Wigner curves
            for i, name in enumerate(plot_wigner or []):
                ax_s.plot(L_plot, sigma2_RMT(L_plot, wigner=name),
                          color_list[i % len(color_list)], lw=2,
                          label=(wigner_labels.get(name, name.upper())
                                 if (not multiple or (multiple and show_legend and correlation_type != "both"))
                                 else None))
    
            ax_s.set_ylabel(r'$\Sigma^2(L)$', fontsize=22)
            ax_s.set_xlim(0.0, L_max)
            ax_s.set_ylim(0.0, ymax_sigma * 1.05)
    
            ax_s.minorticks_on()
            ax_s.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                             labelsize=22, top=True, right=True)
            ax_s.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                             top=True, right=True)
    
            # Legend placement rule: if ONLY sigma is plotted (not "both"), bottom-right
            if correlation_type != "both" and ((multiple and show_legend) or (not multiple and show_legend)):
                ax_s.legend(fontsize=18, loc='lower right')
    
            if correlation_type == "number_variance":
                ax_s.set_xlabel("Interval length $L$", fontsize=22)
    
        # 6) ---- Δ₃(L) plot ------------------------------------------------------
        if has_delta:
            ax_d = ax_delta if correlation_type == "both" else ax_single
    
            # Ensure colors assigned if sigma not computed
            if not has_sigma:
                for i, ss in enumerate(subsets):
                    color = (color_cycle[i % len(color_cycle)] if (multiple and color_cycle) else 'black')
                    ss['color'] = color
    
            ymax_delta = 1.0
            all_delta_vals = []
            for ss in subsets:
                Ld = ss.get('L_delta', np.array([]))
                vals = ss.get('delta3', np.array([]))
                if Ld.size == 0:
                    continue
                ax_d.scatter(Ld, vals, edgecolor=ss['color'], facecolor='white',
                             s=60, linewidth=1.5, zorder=1,
                             label=ss['label'] if (multiple and show_legend) else None)
                if vals.size:
                    all_delta_vals.append(vals)
    
            # Old-code y-max logic for Δ3: include RMT curves
            L_plot = np.linspace(0.0, L_max, 1000)
            if len(all_delta_vals):
                ymax_delta = float(np.max(np.concatenate(all_delta_vals)))
            else:
                ymax_delta = 1.0
            for name in (plot_wigner or []):
                if name.lower() == "poissonian":
                    continue
                vals = delta3_RMT(L_plot, wigner=name)
                ymax_delta = max(ymax_delta, float(np.max(vals)))
    
            # Draw Wigner curves
            for i, name in enumerate(plot_wigner or []):
                ax_d.plot(L_plot, delta3_RMT(L_plot, wigner=name),
                          color_list[i % len(color_list)], lw=2,
                          label=(wigner_labels.get(name, name.upper())
                                 if (not multiple or (multiple and show_legend))
                                 else None))
    
            ax_d.set_ylabel(r'$\Delta_3(L)$', fontsize=22)
            ax_d.set_xlabel("Interval length $L$", fontsize=22)
            ax_d.set_xlim(0.0, L_max)
            ax_d.set_ylim(0.0, ymax_delta * 1.05)
    
            ax_d.minorticks_on()
            ax_d.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                             labelsize=22, top=True, right=True)
            ax_d.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                             top=True, right=True)
    
            # Legend placement rule: bottom-right; if "both", show ONLY on lower axis
            if (multiple and show_legend) or (not multiple and show_legend):
                ax_d.legend(fontsize=18, loc='lower right')
    
        fig.tight_layout()
    
        # 7) ---- Save if requested (use ACTUAL method in filename) ---------------
        ranges_tag = _fmt_ranges_for_filename(ranges_list)
        if save_fig:
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            base = (f"LongRange_{correlation_type}_{wigner_part}_"
                    f"Lmax{L_max}_Lstep{L_step}__{method}{ranges_tag}.{file_format}")
            filepath = os.path.join(self.save_path, base)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Long-range correlation plot saved to: {filepath}")
    
        plt.show()



    # -------------------------------------------------------------------------
    # Helper functions for long-range statistics
    # -------------------------------------------------------------------------
    def _compute_sigma2(self, eigvals, dE, L_max):
        """
        Compute number variance Σ²(L) over intervals of length L.

        Parameters
        ----------
        eigvals : ndarray
            Unfolded spectrum (sorted).
        dE : float
            Grid spacing for the analysis.
        L_max : float
            Maximum interval length to consider.

        Returns
        -------
        L_vals : ndarray
            L values.
        sigma2_vals : ndarray
            Corresponding Σ²(L) values.
        """
        E0 = eigvals[0]
        Emax = eigvals[-1]
        NR = int(np.floor((Emax - E0) / dE))
        grid = E0 + np.arange(1, NR + 1) * dE
        Ngrid = np.searchsorted(eigvals, grid, side='left').astype(float)

        IR_max = min(NR - 1, int(np.floor(L_max / dE)))
        L_vals = np.arange(0, IR_max + 1) * dE
        sigma2_vals = np.empty_like(L_vals)

        for IR in range(1, IR_max + 1):
            counts = Ngrid[IR:] - Ngrid[:-IR]
            L = IR * dE
            sigma2_vals[IR] = np.mean((counts - L) ** 2)

        sigma2_vals[0] = 0.0  # by definition
        return L_vals, sigma2_vals


    def _compute_delta3(self, eigvals, dE, L_max):
        """
        Compute spectral rigidity Δ₃(L) using windowed polynomial fitting.

        Parameters
        ----------
        eigvals : ndarray
            Unfolded spectrum (sorted).
        dE : float
            Grid spacing.
        L_max : float
            Maximum window size.

        Returns
        -------
        L_vals : ndarray
            L values.
        delta3_vals : ndarray
            Δ₃(L) values.
        """
        NR = int(np.floor((eigvals[-1] - eigvals[0]) / dE))
        IR_max = min(NR - 1, int(np.floor(L_max / dE)))
        L_vals = np.arange(1, IR_max + 1) * dE
        delta3_vals = np.zeros(len(L_vals))

        for i, L in enumerate(L_vals):
            delta3 = []
            for j in range(len(eigvals)):
                if eigvals[j] + L > eigvals[-1]:
                    continue
                # Center interval around 0
                sub = eigvals[(eigvals > eigvals[j]) & (eigvals <= eigvals[j] + L)]
                sub -= eigvals[j] + L / 2
                count = len(sub)

                # Polynomial deviation calculation
                sum1 = -1 / L**2 * np.sum(sub)**2
                sum2 = 3 * count / (2 * L**2) * np.sum(sub**2)
                sum3 = -3 / L**4 * np.sum(sub**2)**2
                int_array = np.flip(np.arange(-count + 1, count + 1, step=2))
                sum4 = 1 / L * np.sum(int_array * sub)
                total = count**2 / 16 + sum1 + sum2 + sum3 + sum4
                delta3.append(total)

            delta3_vals[i] = np.mean(delta3)

        # Include L=0 for plotting
        L_vals = np.insert(L_vals, 0, 0.0)
        delta3_vals = np.insert(delta3_vals, 0, 0.0)
        return L_vals, delta3_vals


    # ========================================================================
    #                          Plotting methods
    # ========================================================================

    def plot_graph(self, show_lengths=False, save_fig=False, file_format="pdf"):
        """
        Plot the open quantum graph with labeled real nodes and optional edge length annotations.
    
        Parameters
        ----------
        show_lengths : bool
            If True, display bond lengths on the edges. 
        save_fig : bool
            If True, saves the figure in self.save_path.
        file_format : str
            Format for saving the figure ('pdf', 'svg', 'png', etc.).
        """
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Extract node positions (assumed to be stored as node attributes in self.G)
        pos = nx.get_node_attributes(self.G, 'pos')
    
        # Separate real and virtual nodes
        real_nodes = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = self.virtual_vertices
    
        # Draw real vertices: large black nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=900, node_color='black', ax=ax)
    
        # Draw virtual vertices: smaller gray nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color='black', ax=ax)
    
        # Draw labels for real nodes (indexed from 1)
        node_labels = {i: f"{i + 1}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color='white', font_weight='bold', font_size=20, ax=ax)
    
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color='black')
    
        # Show bond lengths if requested
        if show_lengths:
            edge_labels = {k: f"{v:.2f}" for k, v in nx.get_edge_attributes(self.G, 'length').items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=22, ax=ax)
    
        # Final plot adjustments
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=1.0)
    
        # Save the figure
        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
    
            # Indicate whether lengths are shown in the filename
            suffix = "with_lengths" if show_lengths else "no_lengths"
            filename = f"{self.name}_graph_{suffix}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
    
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"Graph plot saved to {filepath}")
    
        plt.show()
        
    def plot_wavefunction(self, a_in, k, cmap=None, save_fig=False, file_format='pdf'):
        """
        Plot the wavefunction energy density on the closed graph, and display the
        entropy of the mode in the title.
    
        Parameters
        ----------
        a_in : ndarray
            Wavefunction coefficients from compute_wavefunction(k),
            shape (N_modes, n_directed_bonds) or (n_directed_bonds,).
            The first self.n_internal_bonds entries correspond to + direction,
            the next self.n_internal_bonds to - direction (updated_bonds order).
        k : float
            Vacuum wavenumber used for the wavefunction (also used for λ in title).
        cmap : str, matplotlib Colormap, or None, optional
            Colormap for edge coloring.
            - If None, use a modified 'Reds' where value=1 corresponds to Reds(0.75).
            - If str or Colormap, use it directly.
        save_fig : bool, optional
            If True, saves the figure(s) to self.save_path.
        file_format : str, optional
            File format for saving (default: 'pdf').
        """
        from matplotlib.colors import LinearSegmentedColormap
    
        # ---- 1) Prepare input coefficients -----------------------------------
        a_in = np.asarray(a_in)
        if a_in.ndim == 1:
            a_in = a_in[np.newaxis, :]
        N_modes = a_in.shape[0]
    
        # ---- 2) Compute wavelength -------------------------------------------
        wavelength = 2 * np.pi / k  # vacuum wavelength (µm)
    
        # ---- 3) Precompute the edgelist IN THE SAME ORDER as updated_bonds ---
        edgelist_ordered = [(a[0], b[0]) for (a, b) in self.updated_bonds]
    
        # ---- 4) Resolve colormap ---------------------------------------------
        if cmap is None:
            base = plt.cm.get_cmap("Reds")
            cmap_obj = LinearSegmentedColormap.from_list(
                "Reds_truncated",
                [base(x) for x in np.linspace(0.0, 0.75, 256)]
            )
        elif isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = cmap
    
        # ---- 5) Loop over modes ----------------------------------------------
        for mode in range(N_modes):
            fig, ax = plt.subplots(figsize=(10, 8))
    
            # --- Energy densities (per undirected internal bond)
            coeffs = a_in[mode]
            rho_pos = np.abs(coeffs[:self.n_internal_bonds])**2
            rho_neg = np.abs(coeffs[self.n_internal_bonds:])**2
            energy_raw = rho_pos + rho_neg
    
            # --- Entropy computation (Shannon) ---
            total_energy = np.sum(energy_raw)
            if total_energy > 0:
                p = energy_raw / total_energy
                mask = p > 0
                S = -np.sum(p[mask] * np.log(p[mask]))
                N_b = energy_raw.size
                S_norm = S / np.log(N_b) if N_b > 1 else 0.0
            else:
                S = 0.0
                S_norm = 0.0
    
            # --- Normalize energies for drawing (max -> 1) ---
            max_energy = np.max(energy_raw) if energy_raw.size else 0.0
            energy_densities = energy_raw / max_energy if max_energy > 0 else energy_raw
    
            edge_widths = 6 + 8 * energy_densities
            border_widths = edge_widths + 2
    
            # ---- 6) Separate real and virtual nodes ---------------------------
            pos = nx.get_node_attributes(self.G, 'pos')
            real_nodes = [i for i in self.G.nodes if i not in self.virtual_vertices]
            virtual_nodes = self.virtual_vertices
    
            # ---- 7) Draw nodes -------------------------------------------------
            nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                                   node_size=900, node_color='black', ax=ax)
            nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                                   node_size=400, node_color='black', ax=ax)
            node_labels = {i: f"{i + 1}" for i in real_nodes}
            nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                    font_color='white', font_weight='bold', font_size=20, ax=ax)
    
            # ---- 8) Draw edges (same edgelist twice) ---------------------------
            nx.draw_networkx_edges(self.G, pos, ax=ax,
                                   edgelist=edgelist_ordered,
                                   width=border_widths, edge_color='black')
    
            edges = nx.draw_networkx_edges(self.G, pos, ax=ax,
                                           edgelist=edgelist_ordered,
                                           width=edge_widths,
                                           edge_color=energy_densities,
                                           edge_cmap=cmap_obj,
                                           edge_vmin=0.0, edge_vmax=1.0)
    
            # ---- 9) Colorbar ---------------------------------------------------
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(edges, cax=cax)
            cbar.set_label('Energy Density', fontsize=22)
            cbar.ax.tick_params(labelsize=22)
    
            # ---- 10) Title & styling ------------------------------------------
            mode_label = f"Mode {mode + 1}, " if N_modes > 1 else ""
            title = (f"{mode_label}"
                     f"$k = {k:.4f}\\,\\mu m^{{-1}}$, "
                     f"$\\lambda = {wavelength:.3f}\\,\\mu m$, "
                     f"$S = {S:.3f}$, $S_\\mathrm{{norm}} = {S_norm:.3f}$")
            ax.set_title(title, fontsize=20)
            ax.set_aspect('equal')
            ax.axis('off')
            plt.tight_layout(pad=3.0)
    
            # ---- 11) Save figure ----------------------------------------------
            if save_fig:
                base = f"{'wavefunction_mode'+str(mode+1) if N_modes>1 else 'wavefunction'}"
                filename = (f"{base}_k_{k:.4f}_lambda_{wavelength:.3f}"
                            f"_S_{S:.3f}_Snorm_{S_norm:.3f}.{file_format}")
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
                print(f"✓ Wavefunction plot saved to {filepath}")
    
            plt.show()


    def plot_zeta_B(self, wavelength_range=None, N_plot=100_000, spectrum=None,
                    save_fig=False, file_format='pdf', xlim=None,
                    ylim_real=None, ylim_imag=None, ylim_mod=None):
        """
        Plot the real, imaginary, and modulus of the secular determinant f(k) as a function of wavelength.
    
        Parameters
        ----------
        wavelength_range : tuple or None
            (λ_min, λ_max) in microns. If None, uses self.wavelength_range.
        N_plot : int
            Number of wavelength points for plotting.
        spectrum : array-like or None
            Optional: eigenvalues in k (µm⁻¹). If provided, marks eigenvalues on the plots.
        save_fig : bool
            If True, saves the figure to self.save_path.
        file_format : str
            File format for saving (default: 'pdf').
        xlim : tuple or None
            (xmin, xmax) for λ-axis zoom.
        ylim_real, ylim_imag, ylim_mod : tuple or None
            Optional y-axis limits for each subplot.
        """
    
        # ---- 1) Determine wavelength range ----------------------------------
        if wavelength_range is None:
            wavelength_range = self.wavelength_range
        lambda_min, lambda_max = wavelength_range
    
        # ---- 2) Build λ array and convert to k -------------------------------
        lambda_vals = np.linspace(lambda_min, lambda_max, N_plot)
        k_vals = 2 * np.pi / lambda_vals  # vacuum wavenumber
    
        # ---- 3) Compute secular determinant ----------------------------------
        det_vals = np.array([self.secular_det(k) for k in k_vals])
        Re_det = np.real(det_vals)
        Im_det = np.imag(det_vals)
        Mod_det = np.abs(det_vals)
    
        # ---- 4) Setup subplots -----------------------------------------------
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
        subplot_titles = [r'Re[$\zeta_B(k)$]', 
                          r'Im[$\zeta_B(k)$]', 
                          r'$|\zeta_B(k)|$']
        data_series = [Re_det, Im_det, Mod_det]
        ylims = [ylim_real, ylim_imag, ylim_mod]
    
        # ---- 5) Plot each subplot --------------------------------------------
        for ax, data, title, ylim in zip(axs, data_series, subplot_titles, ylims):
            ax.plot(lambda_vals, data, 'k-', lw=1.0)
            ax.set_ylabel(title, fontsize=22)
            ax.tick_params(axis='both', labelsize=22)
            ax.grid(True)
    
            # Apply specific y-axis limits if provided
            if ylim:
                ax.set_ylim(ylim)
    
            # Apply λ zoom if specified
            if xlim:
                ax.set_xlim(xlim)
    
        axs[2].set_xlabel(r'Wavelength $\lambda$ (μm)', fontsize=22)
    
        # ---- 6) Spectrum overlay (marks) -------------------------------------
        if spectrum is not None:
            spectrum_lambda = 2 * np.pi / np.array(spectrum)
            for ax in axs:
                ax.plot(spectrum_lambda,
                        np.zeros_like(spectrum_lambda),
                        'r*', markersize=10)
    
        plt.tight_layout()
    
        # ---- 7) Save figure ---------------------------------------------------
        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            lambda0_min = lambda_min
            lambda0_max = lambda_max
            filename = (f"zeta_B_plot_lambdamin_{lambda0_min:.2f}_"
                        f"lambdamax_{lambda0_max:.2f}_Nplot_{N_plot}"
                        f"{'_with_spectrum' if spectrum is not None else ''}.{file_format}")
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300)
            print(f"✓ Zeta_B plot saved to: {filepath}")
    
        plt.show()

        
    def plot_length_spectrum(self, L_max=4999, n_points=100000,
                             save=True, file_format="pdf",
                             ignore_zero_peak=True,
                             zero_window_um=None):
        """
        Compute and plot the length spectrum from the graph's eigenvalue spectrum
        (closed graph). The ℓ≈0/DC spike is optionally ignored when setting ylim.
    
        Parameters
        ----------
        L_max : float, optional
            Maximum length to plot (µm). Default is 4999.
        n_points : int, optional
            Number of points used for the ℓ grid. Default is 100000.
        save : bool, optional
            If True, saves the figure in self.save_path.
        file_format : str, optional
            File format for saving the figure ('pdf', 'png', 'svg', etc.).
        ignore_zero_peak : bool, optional
            If True (default), exclude a small window near ℓ=0 when computing
            the y-limit so the huge DC peak does not set the ylim.
            This does NOT modify the plotted data, only how ylim is chosen.
        zero_window_um : float or None, optional
            Half-open window width (in µm) to exclude at the origin for ylim
            selection only. If None, it defaults to max(100 µm, 50 * Δℓ).
        """
    
        # 1) ---- Get spectrum ----------------------------------------------------
        # Prefer the degenerate spectrum when available; otherwise fall back to spectrum.
        if hasattr(self, "degenerate_spectrum") and self.degenerate_spectrum is not None:
            k_vals = np.asarray(self.degenerate_spectrum, dtype=np.complex128)
            source_tag = "__deg"
        elif hasattr(self, "spectrum") and self.spectrum is not None:
            k_vals = np.asarray(self.spectrum, dtype=np.complex128)
            source_tag = ""
        else:
            raise AttributeError("No spectrum found. Run compute_spectrum() first.")
        if k_vals.size == 0:
            raise ValueError("Spectrum is empty.")
    
        # 2) ---- Build ℓ grid ----------------------------------------------------
        ell = np.linspace(0.0, float(L_max), int(n_points))  # µm
        d_ell = (ell[1] - ell[0]) if ell.size >= 2 else float(L_max)
    
        # 3) ---- Prepare n_eff(k) ------------------------------------------------
        # Accept either a scalar self.neff or a callable self.neff(k).
        if callable(self.neff):
            # Try vectorized call; if not vectorized, fall back to per-element
            try:
                n_eff_vec = self.neff(np.real(k_vals))
                if np.isscalar(n_eff_vec):
                    n_eff_vec = np.full(k_vals.shape, float(n_eff_vec), dtype=float)
                else:
                    n_eff_vec = np.asarray(n_eff_vec, dtype=float)
                    if n_eff_vec.shape != k_vals.shape:
                        n_eff_vec = np.array([float(self.neff(float(np.real(k))))
                                              for k in k_vals], dtype=float)
            except Exception:
                n_eff_vec = np.array([float(self.neff(float(np.real(k))))
                                      for k in k_vals], dtype=float)
        else:
            n_eff_vec = np.full(k_vals.shape, float(self.neff), dtype=float)
    
        # 4) ---- Compute length spectrum ρ̂(ℓ) ----------------------------------
        # ρ̂(ℓ) = | Σ_j exp[-i * k_j * n_eff(Re(k_j)) * ℓ] |
        rho_hat = np.zeros_like(ell, dtype=np.complex128)
        for kj, neff_j in zip(k_vals, n_eff_vec):
            rho_hat += np.exp(-1j * kj * neff_j * ell)
        rho_hat = np.abs(rho_hat)
    
        # 5) ---- Choose ylim without the ℓ≈0 spike ------------------------------
        if ignore_zero_peak:
            if zero_window_um is None:
                zero_window_um = max(100.0, 50 * d_ell)  # default exclusion window near 0
            mask_ylim = ell > float(zero_window_um)
            if np.any(mask_ylim):
                ymax = float(np.max(rho_hat[mask_ylim]))
            else:
                # Fallback: ignore the first point if the grid is too coarse
                ymax = float(np.max(rho_hat[1:])) if rho_hat.size > 1 else float(rho_hat.max())
        else:
            ymax = float(rho_hat.max())
    
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
    
        # 6) ---- Plot ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(ell, rho_hat, color="black", lw=1.5)
    
        ax.set_xlim(0.0, float(L_max))
        ax.set_ylim(0.0, 1.05 * ymax)
        ax.set_xlabel(r"Length $\ell$ ($\mu$m)", fontsize=22)
        ax.set_ylabel(r"$|\hat{\rho}(\ell)|$", fontsize=22)
    
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
    
        plt.tight_layout()
    
        # 7) ---- Save ------------------------------------------------------------
        if save:
            filename = f"LengthSpectrum_Lmax{L_max}_N{n_points}{source_tag}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Length spectrum figure saved to {filepath}")
    
        plt.show()

        
    def plot_entropy(self, wavelength_range=None, show_smooth=False, smooth_window=11,
                     save_fig=False, file_format='pdf'):
        """
        Plot the normalized entropy S_norm of each mode in TWO figures:
          (1) versus k * L_tot
          (2) versus the mode index (after sorting by k)
    
        For each k in self.spectrum (optionally filtered by wavelength_range),
        the wavefunction coefficients are computed with compute_wavefunction(k, tol=1e-4).
        If multiple vectors are returned (degeneracy), the first one is used. If none,
        that k is skipped.
    
        Entropy is computed on undirected internal bonds by pairing directed bonds:
            rho_b = |a_b^(+)|^2 + |a_b^(-)|^2
            p_b = rho_b / sum_b rho_b
            S = -sum_b p_b log(p_b)
            S_norm = S / log(N_b)
    
        Parameters
        ----------
        wavelength_range : None or (lam_min, lam_max)
            Filter on vacuum wavelength λ (in µm), inclusive. If None, use full spectrum.
        show_smooth : bool
            If True, overlay a moving-average of S_norm as a red line on both figures.
        smooth_window : int
            Window size (number of points) for the moving average (must be >= 3).
            If even, it will be incremented to the next odd integer. If larger than
            the number of data points, it will be reduced accordingly.
        save_fig : bool
            If True, saves the figures in self.save_path.
        file_format : str
            Output file format (e.g., 'pdf', 'png', 'svg').
        """

        # ---- 1) Prepare spectrum and optional wavelength filter ----------------
        if not hasattr(self, "spectrum"):
            raise AttributeError("No spectrum found. Run compute_spectrum() first.")
    
        k_all = np.sort(np.real(np.asarray(self.spectrum, dtype=float)))
        k_all = k_all[np.isfinite(k_all)]
        if k_all.size == 0:
            raise ValueError("Empty spectrum.")
    
        if wavelength_range is not None:
            lam_min, lam_max = wavelength_range
            if not (np.isfinite(lam_min) and np.isfinite(lam_max) and lam_min < lam_max):
                raise ValueError("wavelength_range must be a tuple (lam_min, lam_max) with lam_min < lam_max (µm).")
            lam_all = 2 * np.pi / k_all  # µm
            mask = (lam_all >= lam_min) & (lam_all <= lam_max)
            k_sel = k_all[mask]
        else:
            k_sel = k_all
    
        if k_sel.size == 0:
            raise ValueError("No modes in the requested wavelength range.")
    
        # ---- 2) Iterate over k, compute wavefunction(s) and entropy ------------
        S_norm_vals = []
        x_kL_vals = []  # k * L_tot
        kept_k = []     # keep sorted k for reference / index-plot
        L_tot = float(self.total_length)
    
        for k in k_sel:
            coeffs_set = self.compute_wavefunction(k, tol=1e-4)  # rows = eigenvectors
            if coeffs_set is None or coeffs_set.size == 0:
                continue
    
            # If degeneracy > 1, take the first mode
            a = coeffs_set if coeffs_set.ndim == 1 else coeffs_set[0]
            a = np.asarray(a).ravel()
    
            if a.size != 2 * self.n_internal_bonds:
                print(f"⚠ Skipping k={k:.6f}: unexpected coeff length {a.size} "
                      f"(expected {2*self.n_internal_bonds}).")
                continue
    
            rho_pos = np.abs(a[:self.n_internal_bonds])**2
            rho_neg = np.abs(a[self.n_internal_bonds:])**2
            rho = rho_pos + rho_neg  # per undirected internal bond
    
            total = float(np.sum(rho))
            if total <= 0:
                continue
    
            p = rho / total
            mask = p > 0
            S = -np.sum(p[mask] * np.log(p[mask]))
            N_b = rho.size
            S_norm = S / np.log(N_b) if N_b > 1 else 0.0
    
            S_norm_vals.append(S_norm)
            x_kL_vals.append(k * L_tot)
            kept_k.append(k)
    
        S_norm_vals = np.asarray(S_norm_vals, dtype=float)
        x_kL_vals = np.asarray(x_kL_vals, dtype=float)
        kept_k = np.asarray(kept_k, dtype=float)
    
        if x_kL_vals.size == 0:
            raise ValueError("No valid modes after computing wavefunctions and entropies.")
    
        # ---- 3) Sort by k (equivalently by kL) for clean plots -----------------
        order = np.argsort(kept_k)
        kept_k = kept_k[order]
        x_kL_vals = x_kL_vals[order]
        S_norm_vals = S_norm_vals[order]
    
        # ---- 4) Optional moving average (simple centered window) ----------------
        def _moving_average(y, window):
            if y.size < 3:
                return None
            w = int(window)
            if w < 3:
                w = 3
            if w % 2 == 0:
                w += 1
            if w > y.size:
                w = y.size if y.size % 2 == 1 else y.size - 1
                if w < 3:
                    return None
            kernel = np.ones(w, dtype=float) / w
            pad = w // 2
            ypad = np.pad(y, (pad, pad), mode='edge')
            return np.convolve(ypad, kernel, mode='valid')
    
        smooth_y = _moving_average(S_norm_vals, smooth_window) if show_smooth else None
    
        # Prepare shared labels/tags
        if wavelength_range is None:
            title_range = "full spectrum"
            range_tag = "__full"
        else:
            lam_min, lam_max = wavelength_range
            title_range = fr"$\lambda \in [{lam_min:.3f}, {lam_max:.3f}]\,\mu m$"
            range_tag = f"__range_[{lam_min:.3f}-{lam_max:.3f}]"
        smooth_tag = f"__smoothW{smooth_window}" if show_smooth and smooth_y is not None else ""
    
        # =========================
        # FIGURE 1: S_norm vs k L
        # =========================
        fig1, ax1 = plt.subplots(figsize=(10, 8))
    
        # Per-mode entropy curve
        ax1.plot(x_kL_vals, S_norm_vals, 'ko-', markersize=5, linewidth=1)
    
        # Bounds (0 and 1)
        ax1.axhline(0.0, color='black', linestyle='-.', linewidth=1.5)
        ax1.axhline(1.0, color='black', linestyle='-.', linewidth=1.5)
    
        # Optional smooth
        if smooth_y is not None:
            ax1.plot(x_kL_vals, smooth_y, 'r-', linewidth=2)
    
        ax1.set_title(fr"Normalized entropy vs $k L_{{\rm tot}}$   ({title_range})", fontsize=20)
        ax1.set_xlabel(r"$k\,L_{\rm tot}$", fontsize=22)
        ax1.set_ylabel(r"$S_{\rm norm}$", fontsize=22)
        ax1.set_xlim(float(np.min(x_kL_vals)), float(np.max(x_kL_vals)))
        ax1.set_ylim(-0.03, 1.03)
        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                        labelsize=18, top=True, right=True)
        ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                        top=True, right=True)
        fig1.tight_layout()
    
        if save_fig:
            base1 = f"entropy_vs_kL{range_tag}{smooth_tag}.{file_format}"
            filepath1 = os.path.join(self.save_path, base1)
            fig1.savefig(filepath1, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Entropy (kL) plot saved to: {filepath1}")
    
        plt.show()
    
        # ================================
        # FIGURE 2: S_norm vs mode index
        # ================================
        fig2, ax2 = plt.subplots(figsize=(10, 8))
    
        idx = np.arange(S_norm_vals.size)  # 0..N-1 (consistent with other index plots you use)
    
        ax2.plot(idx, S_norm_vals, 'ko-', markersize=5, linewidth=1)
    
        # Bounds (0 and 1)
        ax2.axhline(0.0, color='black', linestyle='-.', linewidth=1.5)
        ax2.axhline(1.0, color='black', linestyle='-.', linewidth=1.5)
    
        if smooth_y is not None:
            # Moving average computed in k-sorted order already matches idx order
            ax2.plot(idx, smooth_y, 'r-', linewidth=2)
    
        ax2.set_title(fr"Normalized entropy vs mode index   ({title_range})", fontsize=20)
        ax2.set_xlabel(r"Mode index", fontsize=22)
        ax2.set_ylabel(r"$S_{\rm norm}$", fontsize=22)
        ax2.set_xlim(0, max(1, int(idx[-1])) if idx.size else 1)
        ax2.set_ylim(-0.03, 1.03)
        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                        labelsize=18, top=True, right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                        top=True, right=True)
        fig2.tight_layout()
    
        if save_fig:
            base2 = f"entropy_vs_index{range_tag}{smooth_tag}.{file_format}"
            filepath2 = os.path.join(self.save_path, base2)
            fig2.savefig(filepath2, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Entropy (index) plot saved to: {filepath2}")
    
        plt.show()


        
     
# ========================================================================
#                 Subclass of OpenGraphs: DirectionalGraphs
# ========================================================================
        
class DirectionalGraphs(ClosedGraphs):
    """
    Construct a closed quantum graph made of 4-port directional couplers.

    This subclass of `ClosedGraphs` automatically generates the vertex
    scattering matrices from user-specified coupling constants.  
    Each vertex is assumed to be a lossless 4-port directional coupler.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of the vertices in µm.

    port_map : list of lists
        Connectivity specification. For vertex i and port p:
            port_map[i][p] = (j, q)
        meaning port p of vertex i connects to port q of vertex j.
        No `None` entries are allowed in `DirectionalGraphs`.

    lengths : dict or None, optional
        Explicit dictionary mapping bonds to physical lengths:
            {((i, p), (j, q)) : length_in_um}
        If None, lengths are inferred from Euclidean distances between
        vertex positions.

    vertex_couplings : dict, list, float, or callable
        Coupling constants for each vertex.
        - dict/list: one entry per vertex.
        - float: same constant for all vertices.
        - callable: returns C(k) as a function of wavenumber.

    neff : float or callable, optional
        Effective refractive index. May be constant or k-dependent.

    ng : float or callable, optional
        Group index. If None, it is derived numerically from neff(k).

    wavelength_range : tuple of (float, float), optional
        Vacuum wavelength range (µm) defining the working spectral window.

    virtual_vertex_spacing : float, sequence, or None, optional
        Spacing used when splitting multiple bonds. If a sequence is given,
        values are cycled across multi-bond groups.

    name : str, optional
        Label for the graph (used in plots and default save paths).

    save_path : str, pathlib.Path, or None, optional
        Directory where results will be stored.
        - If None (default), results are saved under:
              {Path.cwd()}/{name}/{_build_save_dir_name(coupling_info)}
        - If provided, this path is used directly (overrides `name`).
    """

    def __init__(self,
                 vertex_positions,
                 port_map,
                 lengths=None,
                 vertex_couplings=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 name="MyClosedDirectionalGraph",
                 save_path=None):
        """
        Initialize a closed directional graph (all vertices = 4-port couplers).  
        Builds the vertex scattering matrices automatically from couplings.  
        """
        # --- Store couplings ---
        self.vertex_couplings = vertex_couplings

        # --- Build vertex scattering matrix dictionary ---
        vertex_matrices = self._build_vertex_matrices(vertex_positions, port_map)

        # --- Build coupling info string (used in _build_save_dir_name) ---
        self.coupling_info = self._get_coupling_description()

        # --- Call parent constructor ---
        super().__init__(
            vertex_positions=vertex_positions,
            port_map=port_map,
            lengths=lengths,
            vertex_matrices=vertex_matrices,
            neff=neff,
            ng=ng,
            wavelength_range=wavelength_range,
            virtual_vertex_spacing=virtual_vertex_spacing,
            name=name,
            save_path=save_path
        )

    def _build_vertex_matrices(self, vertex_positions, port_map):
        """
        Build the vertex_scattering_matrix dictionary from coupling constants.

        Each vertex must be a 4-port directional coupler.
        """
        n = len(vertex_positions)
        valencies = [len(ports) for ports in port_map]
        matrices = {}

        for v in range(n):
            val = valencies[v]
            if val != 4:
                raise ValueError(
                    f"Only 4-port directional couplers are supported, "
                    f"but vertex {v} has valency {val}."
                )

            # Determine coupling constant for this vertex
            if isinstance(self.vertex_couplings, (dict, list)):
                if v not in self.vertex_couplings:
                    raise ValueError(f"Missing coupling constant for vertex {v}.")
                C = self.vertex_couplings[v]
            else:
                # Single coupling used for all vertices
                C = self.vertex_couplings

            # Assign scattering matrix (callable if C is callable)
            matrices[v] = self._directional_coupler_matrix(C)

        return matrices

    def _directional_coupler_matrix(self, C):
        """
        Return a 4x4 directional coupler scattering matrix from coupling constant C.

        Parameters
        ----------
        C : float or callable
            Coupling constant (power fraction). Callable if dependent on k.

        Returns
        -------
        np.ndarray or function
            - Constant matrix if C is a float.
            - Callable matrix function(k) if C is a function of k.
        """

        def matrix_fn(k=None):
            c_val = C(k) if callable(C) else C
            sC = np.sqrt(c_val)
            s1C = np.sqrt(1 - c_val)
            return np.array([
                [0,     0,     s1C, 1j * sC],
                [0,     0,     1j * sC, s1C],
                [s1C, 1j * sC, 0,     0],
                [1j * sC, s1C, 0,     0]
            ], dtype=complex)

        return matrix_fn if callable(C) else matrix_fn()

    def _get_coupling_description(self):
        """
        Build a compact string description for the coupling constants.
        If a single global coupling is used, describe it.
        If per-vertex couplings are given, mark as 'custom_vertex_couplings'.
        """
        # Case 1: dictionary or list (per-vertex)
        if isinstance(self.vertex_couplings, (dict, list)):
            return "custom_vertex_couplings"

        # Case 2: single coupling (float or callable)
        C = self.vertex_couplings
        if callable(C):
            return getattr(C, '__name__', 'Cfunc')
        return f"C={C:.2f}"
    
    def plot_graph(self,
                   show_lengths=False,
                   show_port_arrows=False,
                   head_length_frac=0.12,   # head length as fraction of Lmin
                   head_width_frac=0.08,    # head width  as fraction of Lmin
                   head_offset_frac=0.24,   # CENTER distance from node = head_offset_frac * Lmin
                   save_fig=False,
                   file_format="pdf"):
        """
        Plot the closed quantum graph with labeled real nodes, optional edge lengths,
        and head-only arrows à la Dietz on 4-port directional vertices:
          - GREEN for incoming ports (0,1)
          - RED   for outgoing ports (2,3)
    
        Arrowheads are custom triangular patches with:
          - SAME absolute center distance from the node for all ports (equidistant),
          - Size (length, width) controlled by fractions of Lmin.
        """
    
        # ------------------------------ Colors ---------------------------------
        GREEN = "#2ca02c"  # incoming ports (0,1)
        RED   = "#d62728"  # outgoing ports (2,3)
        EDGE_COLOR = "black"
    
        # -------------------- Compute Lmin for geometric scaling ----------------
        if not hasattr(self, "virtual_lengths") or len(self.virtual_lengths) == 0:
            raise AttributeError("self.virtual_lengths must be a non-empty dict of bond lengths.")
        Lmin = float(min(self.virtual_lengths.values()))
        head_length = head_length_frac * Lmin
        head_width  = head_width_frac  * Lmin
        offset_abs  = head_offset_frac * Lmin   # <-- single absolute offset (equidistant for all)
    
        # ------------------------ Figure and layout -----------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.get_node_attributes(self.G, 'pos')
        real_nodes = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = list(self.virtual_vertices)
    
        # Draw nodes/edges FIRST (lower zorder)
        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=900, node_color='black', ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color='black', ax=ax)
    
        # 1-indexed labels for closed graph
        node_labels = {i: f"{i + 1}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color='white', font_weight='bold',
                                font_size=18, ax=ax)
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color=EDGE_COLOR)
    
        if show_lengths:
            edge_labels = {k: f"{v:.2f}"
                           for k, v in nx.get_edge_attributes(self.G, 'length').items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=20, ax=ax)
    
        # ---------------------- Low-level arrowhead helper ----------------------
        def draw_arrowhead(ax, center, direction, length, width, color, z=100):
            """
            Draw a filled isosceles triangular arrowhead, always on top (zorder=z).
            The 'direction' points from base toward tip.
            """
            center = np.asarray(center, dtype=float)
            d = np.asarray(direction, dtype=float)
            nrm = np.hypot(d[0], d[1])
            if nrm == 0:
                return
            u = d / nrm
            n = np.array([-u[1], u[0]])  # +90° rotation
    
            tip = center + (length * 0.5) * u
            base_mid = center - (length * 0.5) * u
            half_w = 0.5 * width
            bl = base_mid - half_w * n
            br = base_mid + half_w * n
    
            poly = plt.Polygon([tip, br, bl], closed=True,
                               facecolor=color, edgecolor=color,
                               linewidth=0, zorder=z, clip_on=False)
            ax.add_patch(poly)
    
        # --------------------- Helper to place centers safely -------------------
        def center_at_offset_from_vertex(vertex_xy, neighbor_xy, desired_offset, head_len):
            """
            Place the arrowhead center along the segment starting at 'vertex_xy' toward 'neighbor_xy'
            at a distance 'desired_offset' from the vertex, but clamp so the head stays on-segment.
            """
            vx, vy = vertex_xy
            nx_, ny_ = neighbor_xy
            dx, dy = nx_ - vx, ny_ - vy
            Lseg = float(np.hypot(dx, dy))
            if Lseg == 0.0:
                return None, None, 0.0
    
            u = np.array([dx, dy]) / Lseg
    
            # Keep head fully on the segment: center must be at least half head_len from both ends
            margin = 0.55
            max_center = max(0.0, (Lseg * 0.5) - margin * (head_len * 0.5))
            eff_offset = min(desired_offset, max_center)
    
            center = np.array([vx, vy]) + eff_offset * u
            return center, u, Lseg
    
        # ------------------------ Port heads (internal) -------------------------
        if show_port_arrows:
            # If you keep a valency mask, you can use it; otherwise we rely solely on virtual_port_map.
            for v in self.G.nodes:
                if v in self.virtual_vertices:
                    continue
                if not hasattr(self, "virtual_port_map"):
                    raise AttributeError("Closed graph needs self.virtual_port_map to place arrows.")
                # Optional: if valency info exists and you want exactly 4-port nodes:
                if hasattr(self, "valencies"):
                    try:
                        if self.valencies[v] != 4:
                            continue
                    except Exception:
                        pass  # fall back to drawing if mapping exists
    
                xv, yv = pos[v]
                vxy = (xv, yv)
    
                for p in range(4):
                    conn = self.virtual_port_map[v][p]
                    if conn is None:
                        continue  # closed graphs shouldn't have leads; skip just in case
    
                    n, q = conn
                    nxy = pos[n]
    
                    center, u, Lseg = center_at_offset_from_vertex(vxy, nxy, offset_abs, head_length)
                    if center is None:
                        continue
    
                    if p in (0, 1):
                        # Incoming: arrow points toward node => -u
                        draw_arrowhead(ax, center, -u, head_length, head_width, GREEN)
                    else:
                        draw_arrowhead(ax, center, +u, head_length, head_width, RED)
    
        # --------------------------- Final styling ------------------------------
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=1.0)
    
        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            tags = ["with_lengths" if show_lengths else "no_lengths"]
            if show_port_arrows:
                tags.append("port_heads")
            filename = f"{self.name}_graph_{'_'.join(tags)}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"Graph plot saved to {filepath}")
    
        plt.show()

        
    


    
    








