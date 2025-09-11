#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open Quantum Graphs — Numerical modeling of wave propagation on open quantum graphs.

This module defines the `OpenGraphs` class to construct and analyze open quantum graphs
as described in the vertex scattering matrix formalism [Smilansky, Gnutzmann].

Each graph is initialized with:
- vertex positions (embedded geometry),
- a port_map to define connectivity (including open leads as `None`),
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
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
from scipy.fft import fft, fftfreq, next_fast_len
from scipy.signal import windows, find_peaks
from scipy.optimize import minimize, Bounds
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import numdifftools as nd
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

class OpenGraphs:
    """
    Construct an open quantum graph in the vertex–scattering formalism.

    An open graph allows external leads: ports may connect either internally
    to another vertex or externally to infinity. During initialization,
    geometry, bond lengths, and scattering matrices are set up, while
    self-loops and multiple bonds are automatically resolved using virtual
    vertices with Neumann boundary conditions.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of the vertices in µm.

    port_map : list of lists
        Connectivity specification. For vertex i and port p:
            port_map[i][p] = (j, q)  if connected internally, or
            port_map[i][p] = None    if port is an external lead.

    lengths : dict or None, optional
        Explicit dictionary mapping bonds to physical lengths:
            {((i, p), (j, q)) : length_in_um}
        If None, inferred from Euclidean distances.

    vertex_matrices : dict or None, optional
        Dictionary {v: σ_v(k)} of vertex scattering matrices (possibly k-dependent).

    internal_matrix : ndarray, callable, or None, optional
        Scattering matrix or function used for internal (non-lead) vertices.

    external_matrix : ndarray, callable, or None, optional
        Scattering matrix or function used for external (lead-connected) vertices.

    neff : float or callable, optional
        Effective refractive index (may depend on vacuum wavenumber).

    ng : float or callable, optional
        Group index. If None, derived numerically from neff(k).

    wavelength_range : tuple of (float, float), optional
        Vacuum wavelength range (µm) defining the working spectral window.

    virtual_vertex_spacing : float, sequence, or None, optional
        Spacing used when splitting multiple bonds or loops.
        If None, defaults to 10% of the bond length.

    lead_orientations : dict or None, optional
        Optional mapping that assigns geometric directions to external leads
        for plotting purposes.

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
                 internal_matrix=None, external_matrix=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 lead_orientations=None,
                 name="MyOpenGraph",
                 save_path=None):
        
        """
        Initialize an open quantum graph (with possible external leads).  
        Sets geometry, bond lengths, and vertex scattering matrices,  
        while resolving loops and multiple bonds with virtual vertices.  
        """
        
        # --- Basic attributes ---
        self.vertex_positions = list(vertex_positions)
        self.port_map = port_map
        self.virtual_port_map = copy.deepcopy(self.port_map)  # mutated by virtual splitting
        self.lengths = lengths                                # ORIGINAL (immutable after init)
        self.n_vertices = len(vertex_positions)
        self.wavelength_range = wavelength_range
        self.neff = neff
        self.ng = ng if ng is not None else self._compute_group_index()
        self.lead_orientations = lead_orientations  # can be None or a dict {(vertex, port): angle_deg}
    
        # Normalize & prepare spacing input (supports float OR cycling list)
        self._init_spacing_sequence(virtual_vertex_spacing)
        self.virtual_vertex_spacing = virtual_vertex_spacing  # keep original for metadata
    
        # --- Compute ORIGINAL lengths if not provided ---
        if self.lengths is None:
            self.lengths = {}
            for v_idx, ports in enumerate(self.port_map):
                for p_idx, bond in enumerate(ports):
                    if bond is not None:
                        v2, p2 = bond
                        # Avoid double-counting (only compute if v_idx < v2)
                        if v_idx < v2:
                            x1, y1 = self.vertex_positions[v_idx]
                            x2, y2 = self.vertex_positions[v2]
                            dist = np.hypot(x2 - x1, y2 - y1)
                            self.lengths[((v_idx, p_idx), (v2, p2))] = dist
    
        # Preserve pristine total length (for reference)
        self.total_length = sum(self.lengths.values())
    
        # Create the VIRTUAL lengths dict that WILL be modified by splitting
        self.virtual_lengths = copy.deepcopy(self.lengths)
    
        # STEP 1: Build bond list and lead list
        self.bonds = []       # List of tuples (i, pi, j, pj)
        self.leads = []       # List of (i, pi)
        self.valencies = []   # Number of ports at each vertex
    
        for i, ports in enumerate(self.port_map):
            self.valencies.append(len(ports))
            for pi, conn in enumerate(ports):
                if conn is None:
                    self.leads.append((i, pi))
                else:
                    j, pj = conn
                    # --- FILTER: skip degenerate self-loops like (0,2,0,2) ---
                    if i == j and pi == pj:
                        continue
                    # Avoid double-counting
                    if (j, pj, i, pi) not in self.bonds:
                        self.bonds.append((i, pi, j, pj))
    
        self.n_bonds = len(self.bonds)
        self.n_leads = len(self.leads)
    
        # STEP 2: Group by vertex pair (i, j), ignoring ports
        vertex_pair_bonds = defaultdict(list)
        for (i, pi, j, pj) in self.bonds:
            vpair = tuple(sorted([i, j]))
            vertex_pair_bonds[vpair].append(((i, pi), (j, pj)))
    
        # --- STEP 3: Break multiple bonds / loops (choose one spacing per multi-bond group) ---
        self.virtual_vertices = []
        self.updated_bonds = []
    
        for (i, j), bond_list in vertex_pair_bonds.items():
            if i == j:
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
    
        # >>> Canonicalize and order everything consistently (uses virtual_lengths) <<<
        self._order_bonds_and_lengths()
    
        # STEP 4: Build networkx graph (use VIRTUAL geometry)
        self.G = nx.Graph()
        self.vertex_scattering_matrix = {}
    
        for i, pos in enumerate(self.vertex_positions):
            self.G.add_node(i, pos=pos)
    
        for (a, pa), (b, pb) in self.updated_bonds:
            key = tuple(sorted([(a, pa), (b, pb)]))
            self.G.add_edge(a, b, length=self.virtual_lengths[key])
    
        # STEP 5: Assign vertex scattering matrices
        for v in range(len(self.vertex_positions)):
            if v in self.virtual_vertices:
                # Neumann 2×2 matrix for virtual nodes
                self.vertex_scattering_matrix[v] = np.array([[0, 1], [1, 0]])
                continue
    
            if vertex_matrices is not None and v in vertex_matrices:
                self.vertex_scattering_matrix[v] = vertex_matrices[v]
            else:
                # Decide if vertex is connected to a lead
                is_external = any((v, p) in self.leads for p in range(self.valencies[v]))
    
                if is_external:
                    if external_matrix is None:
                        raise ValueError(f"No external_matrix provided for external vertex {v}")
                    self.vertex_scattering_matrix[v] = external_matrix
                else:
                    if internal_matrix is None:
                        raise ValueError(f"No internal_matrix provided for internal vertex {v}")
                    self.vertex_scattering_matrix[v] = internal_matrix
    
        # STEP 6: Metadata
        self.n_vertices_total = len(self.vertex_positions)
        self.n_virtual_vertices = len(self.virtual_vertices)
        self.n_internal_bonds = len(self.updated_bonds)
        self.n_directed_bonds = 2 * self.n_internal_bonds
    
        # STEP 7: Build directed bonds and VIRTUAL lengths (for propagation & scattering)
        self.directed_bonds = []          # list of ((i, pi), (j, pj)) for each directed bond
        self.directed_bond_index = {}     # lookup: directed bond -> index
        self.directed_lengths = {}        # lookup: directed bond -> length
    
        for idx, (a, b) in enumerate(self.updated_bonds):
            # Forward direction
            self.directed_bonds.append((a, b))
            self.directed_bond_index[(a, b)] = idx
    
            # Reverse direction
            self.directed_bonds.append((b, a))
            self.directed_bond_index[(b, a)] = idx + self.n_internal_bonds
    
            # Length for both directions (same value) from VIRTUAL lengths
            key = tuple(sorted([a, b]))
            if key not in self.virtual_lengths:
                raise ValueError(f"No virtual length provided for bond {key}")
            L = self.virtual_lengths[key]
    
            self.directed_lengths[(a, b)] = L
            self.directed_lengths[(b, a)] = L
    
        # STEP 8: Generate the lead_orientations dictionary (if the helper exists)
        if hasattr(self, "_generate_lead_orientations"):
            self._generate_lead_orientations()
    
        ## STEP 9: Name & save directory (new logic)
        self.name = name
    
        if save_path is None:
            # Default: save under CWD / {name} / _build_save_dir_name()
            base_dir = Path.cwd()
            graph_dir = base_dir / self.name
            coupling_info = getattr(self, "coupling_info", None)  # safe hook for subclasses
            leaf = self._build_save_dir_name(coupling_info)
            self.save_path = graph_dir / leaf
        else:
            # User override: use the provided path directly (name is ignored)
            self.save_path = Path(save_path)
    
        # Reuse folder silently; never add PID suffixes
        self.save_path.mkdir(parents=True, exist_ok=True)
    
        # STEP 10: Save key parameters to text
        self._save_parameters()
        

    def _break_complex_bond(self, bond, loop=False, index=0, n_total=1, group_spacing=None):
        """
        Replace a multiple bond or self-loop with virtual vertices and update virtual_port_map.
    
        Parameters
        ----------
        bond : tuple
            ((i, pi), (j, pj)) port-level bond.
        loop : bool
            True if bond is a loop (i == j).
        index : int
            Index of this bond among duplicates.
        n_total : int
            Total number of redundant bonds for this pair.
        group_spacing : float or None
            Spacing chosen for the entire multi-bond group (if provided).
        """
    
        (i, pi), (j, pj) = bond
        xi, yi = self.vertex_positions[i]
        xj, yj = self.vertex_positions[j]
    
        dx, dy = xj - xi, yj - yi
        distance = np.hypot(dx, dy)
        norm_dir = np.array([dx, dy]) / distance if distance > 0 else np.array([1.0, 0.0])
        perp = np.array([-norm_dir[1], norm_dir[0]])
    
        # --- unified spacing choice ---
        # Priority:
        # 1) group_spacing from caller (one per multi-bond group)
        # 2) self.virtual_vertex_spacing when it's a single float
        # 3) default to 0.3 * distance
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
            # Optional fixed fallback for loops when no spacing provided
            if self.virtual_vertex_spacing is None and group_spacing is None:
                spacing = 10.0  # stable size for the ring geometry
    
            # Remove the original self-connection from virtual_port_map
            self.virtual_port_map[i][pi] = None
            self.virtual_port_map[i][pj] = None
    
            # Pop the original loop length from VIRTUAL lengths (keep originals intact)
            L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)
    
            center = np.array([xi, yi])
            angles = [0, 2*np.pi/3, 4*np.pi/3]
            idxs = []
    
            # Create 3 virtual vertices around the center
            for t in angles:
                pt = center + spacing * np.cos(t) * perp + spacing * np.sin(t) * norm_dir
                v_idx = len(self.vertex_positions)
                self.vertex_positions.append(tuple(pt))
                self.valencies.append(2)
                self.virtual_vertices.append(v_idx)
                self.virtual_port_map.append([None, None])  # two ports
                idxs.append(v_idx)
    
            # Build ring: i — v1 — v2 — v3 — i
            all_nodes = [i] + idxs + [i]
            segment_length = L_total / 4.0
    
            for k in range(4):
                a, b = all_nodes[k], all_nodes[k+1]
    
                # Update topology
                self.updated_bonds.append(((a, 0), (b, 0)))
    
                # Set VIRTUAL lengths
                self.virtual_lengths[tuple(sorted([(a, 0), (b, 0)]))] = segment_length
    
                # Connect ports in virtual_port_map (fill port 0, then 1)
                if self.virtual_port_map[a][0] is None:
                    self.virtual_port_map[a][0] = (b, 0)
                elif self.virtual_port_map[a][1] is None:
                    self.virtual_port_map[a][1] = (b, 0)
    
                if self.virtual_port_map[b][0] is None:
                    self.virtual_port_map[b][0] = (a, 0)
                elif self.virtual_port_map[b][1] is None:
                    self.virtual_port_map[b][1] = (a, 0)
    
            return
    
        # ================================================================
        # CASE 2: MULTIPLE BONDS (i ≠ j)
        # ================================================================
        # Fan-out point for distinct parallels
        midpoint = (np.array([xi, yi]) + np.array([xj, yj])) / 2
        offset = (index - (n_total - 1) / 2) * spacing
        pt = midpoint + offset * perp
    
        # Create one virtual vertex
        v_idx = len(self.vertex_positions)
        self.vertex_positions.append(tuple(pt))
        self.valencies.append(2)
        self.virtual_vertices.append(v_idx)
        self.virtual_port_map.append([None, None])
    
        # Remove direct connection (i, pi) ↔ (j, pj)
        self.virtual_port_map[i][pi] = None
        self.virtual_port_map[j][pj] = None
    
        # Connect (i, pi) → (v_idx, 0) and (j, pj) → (v_idx, 1)
        self.virtual_port_map[i][pi] = (v_idx, 0)
        self.virtual_port_map[v_idx][0] = (i, pi)
        self.virtual_port_map[j][pj] = (v_idx, 1)
        self.virtual_port_map[v_idx][1] = (j, pj)
    
        # Pop the original bond length from VIRTUAL lengths and split
        L_total = self.virtual_lengths.pop(((i, pi), (j, pj)), 40.0)
        segment_length = L_total / 2.0
    
        self.updated_bonds.append(((i, pi), (v_idx, 0)))
        self.updated_bonds.append(((v_idx, 1), (j, pj)))
    
        self.virtual_lengths[tuple(sorted([(i, pi), (v_idx, 0)]))] = segment_length
        self.virtual_lengths[tuple(sorted([(v_idx, 1), (j, pj)]))] = segment_length
       
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
            # leave self.virtual_vertex_spacing as the given float
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
      
    def _generate_lead_orientations(self):
        """
        Generate or validate lead orientations.
    
        - If self.lead_orientations is provided, validate and convert to radians.
        - If None, auto-generate based on the graph geometry after virtual vertices have been added.
        """
        self.lead_orientations_rad = {}
    
        if self.lead_orientations is not None:
            # ✅ User provided lead orientations (in degrees)
            for (v, p) in self.leads:
                if (v, p) not in self.lead_orientations:
                    raise ValueError(f"Missing orientation for lead (vertex {v}, port {p})")
                angle_deg = self.lead_orientations[(v, p)]
                self.lead_orientations_rad[(v, p)] = np.deg2rad(angle_deg)
            return  # nothing else to do
    
        # ✅ Auto-generate lead orientations
        self.lead_orientations = {}
    
        # Group leads by vertex
        from collections import defaultdict
        leads_by_vertex = defaultdict(list)
        for (v, p) in self.leads:
            leads_by_vertex[v].append(p)
    
        for v, ports in leads_by_vertex.items():
            # Find internal bonds connected to this vertex
            connected_vecs = []
            for port_idx, conn in enumerate(self.virtual_port_map[v]):
                if conn is not None:
                    neighbor, n_port = conn
                    vec = np.array(self.vertex_positions[neighbor]) - np.array(self.vertex_positions[v])
                    connected_vecs.append(vec)
    
            if len(connected_vecs) == 0:
                # No internal bonds → default outward direction along +x
                base_angle = 0.0
            else:
                avg_vec = np.mean(connected_vecs, axis=0)
                base_vec = -avg_vec  # opposite direction
                base_angle = np.degrees(np.arctan2(base_vec[1], base_vec[0]))
    
            n_leads = len(ports)
    
            if n_leads == 1:
                # Only one lead → points in base direction
                self.lead_orientations[(v, ports[0])] = base_angle
                self.lead_orientations_rad[(v, ports[0])] = np.deg2rad(base_angle)
            else:
                # Multiple leads → spread them symmetrically around base direction
                spread = 120 / (n_leads - 1)  # narrower spacing if more leads (default is ±60° for two leads)
    
                start_angle = base_angle - spread * (n_leads - 1) / 2
                for i, port in enumerate(ports):
                    angle = start_angle + i * spread
                    self.lead_orientations[(v, port)] = angle
                    self.lead_orientations_rad[(v, port)] = np.deg2rad(angle)
                    
    def _order_bonds_and_lengths(self):
        """
        Canonicalize and globally order self.updated_bonds and self.lengths.
    
        - Each bond is stored as ((i,pi),(j,pj)) with (i,pi) <= (j,pj).
        - self.updated_bonds is sorted by (i,pi) then (j,pj).
        - self.lengths is reinserted to follow the same order.
        """

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
            if key not in ordered_lengths:  # keep first occurrence if duplicates appear
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

    def _save_parameters(self):
        """
        Save a complete, human-readable description of the open quantum graph.
    
        Contents:
          - Metadata (name, path, date, counts)
          - Optical parameters (neff/ng; constants or callables with source if available)
          - Vertex scattering (per-vertex; internal/external matrices if used)
          - Raw port_map (as provided)
          - Raw bond lengths (as provided or geometry-derived)
          - Leads and (optional) lead_orientations
          - Virtualization summary (virtual vertices, updated bonds, virtual lengths)
          - Directed-bond bookkeeping
          - Deterministic hash (port_map + lengths)
        """

        filepath = os.path.join(self.save_path, "graph_parameters.txt")
        with open(filepath, "w") as f:
    
            # ===== Header & metadata =====
            f.write("### Open Quantum Graph Parameters ###\n")
            f.write(f"Graph Name: {self.name}\n")
            f.write(f"Saved on: {datetime.datetime.now()}\n")
            f.write(f"Save path: {self.save_path}\n")
            f.write(f"Wavelength range (vacuum): {self.wavelength_range[0]:.3f} – "
                    f"{self.wavelength_range[1]:.3f} µm\n")
            f.write("\n")
    
            # Counts
            f.write(f"Total vertices: {self.n_vertices_total} "
                    f"(virtual: {self.n_virtual_vertices})\n")
            f.write(f"Internal bonds: {self.n_internal_bonds}\n")
            f.write(f"Directed bonds: {self.n_directed_bonds}\n")
            f.write(f"Leads: {getattr(self, 'n_leads', len(getattr(self, 'leads', [])))}\n")
            f.write("\n")
    
            # Deterministic hash (based on connectivity & raw lengths)
            hash_input = str(self.port_map) + str(self.lengths)
            hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            f.write(f"Config hash (port_map + lengths): {hash_digest}\n\n")
    
            # ===== Optical parameters =====
            f.write("### Optical Parameters ###\n")
            # neff
            if callable(self.neff):
                f.write("Effective index neff(k): function of k\n")
                try:
                    f.write(inspect.getsource(self.neff) + "\n")
                except Exception:
                    f.write("(source code unavailable)\n")
            else:
                f.write(f"Effective index neff: {float(self.neff):.6f}\n")
    
            # ng
            if callable(self.ng):
                f.write("Group index ng(k): function of k\n")
                try:
                    f.write(inspect.getsource(self.ng) + "\n")
                except Exception:
                    f.write("(source code unavailable)\n")
            else:
                f.write(f"Group index ng: {float(self.ng):.6f}\n")
            f.write("\n")
    
            # ===== Vertex scattering matrices =====
            f.write("### Vertex Scattering Matrices ###\n")
            # If the graph was configured with global internal/external matrices, log them:
            internal_matrix = locals().get("internal_matrix", None)  # not visible here
            external_matrix = locals().get("external_matrix", None)  # but we can try attributes
            # Prefer attributes set earlier (if user stored them on self)
            if hasattr(self, "internal_matrix_default"):
                internal_matrix = getattr(self, "internal_matrix_default")
            if hasattr(self, "external_matrix_default"):
                external_matrix = getattr(self, "external_matrix_default")
    
            if internal_matrix is not None:
                f.write("Internal (bulk) vertex matrix (used for vertices without leads):\n")
                try:
                    if callable(internal_matrix):
                        f.write("function of k\n")
                        f.write(inspect.getsource(internal_matrix) + "\n")
                    else:
                        np.savetxt(f, np.asarray(internal_matrix), fmt="%-10.6f")
                        f.write("\n")
                except Exception:
                    f.write("(unavailable)\n")
    
            if external_matrix is not None:
                f.write("External (lead-connected) vertex matrix:\n")
                try:
                    if callable(external_matrix):
                        f.write("function of k\n")
                        f.write(inspect.getsource(external_matrix) + "\n")
                    else:
                        np.savetxt(f, np.asarray(external_matrix), fmt="%-10.6f")
                        f.write("\n")
                except Exception:
                    f.write("(unavailable)\n")
    
            f.write("\nPer-vertex matrices:\n")
            for v, sigma in self.vertex_scattering_matrix.items():
                if callable(sigma):
                    f.write(f"Vertex {v} (valency {self.valencies[v]}): function of k\n")
                    try:
                        f.write(inspect.getsource(sigma) + "\n")
                    except Exception:
                        f.write("(source code unavailable)\n")
                else:
                    f.write(f"Vertex {v} (valency {self.valencies[v]}):\n")
                    try:
                        np.savetxt(f, np.asarray(sigma), fmt="%-10.6f")
                    except Exception:
                        f.write(str(sigma))
                    f.write("\n")
            f.write("\n")
    
            # ===== Raw connectivity =====
            f.write("### Port Map (raw, as provided) ###\n")
            for i, ports in enumerate(self.port_map):
                f.write(f"Vertex {i} ports: {ports}\n")
            f.write("\n")
    
            # Leads
            f.write("### Leads ###\n")
            if hasattr(self, "leads") and self.leads:
                for (i, p) in self.leads:
                    f.write(f"Lead at (vertex {i}, port {p})\n")
            else:
                f.write("(none)\n")
            f.write("\n")
    
            # Optional lead orientations
            f.write("### Lead Orientations (deg) ###\n")
            if isinstance(getattr(self, "lead_orientations", None), dict) and self.lead_orientations:
                for key, ang in self.lead_orientations.items():
                    f.write(f"{key}: {ang}\n")
            else:
                f.write("(none)\n")
            f.write("\n")
    
            # ===== Raw bond lengths =====
            f.write("### Bond Lengths (raw, before virtualization) ###\n")
            if self.lengths:
                for (a, b), L in self.lengths.items():
                    f.write(f"{a} — {b} : {L:.6f} µm\n")
            else:
                f.write("(none)\n")
            f.write("\n")
    
            # ===== Virtualization summary =====
            f.write("### Virtualization Info ###\n")
            f.write(f"Virtual vertices: {getattr(self, 'virtual_vertices', [])}\n")
            f.write("Updated bonds (after virtualization):\n")
            for bond in getattr(self, "updated_bonds", []):
                f.write(f"  {bond}\n")
    
            f.write("\nVirtual lengths (after virtualization):\n")
            if getattr(self, "virtual_lengths", None):
                for key, L in self.virtual_lengths.items():
                    f.write(f"{key} : {L:.6f} µm\n")
            else:
                f.write("(none)\n")
            f.write("\n")
    
            # ===== Directed-bond bookkeeping =====
            f.write("### Directed Bonds (indexing) ###\n")
            try:
                for db in self.directed_bonds:
                    idx = self.directed_bond_index.get(db, None)
                    L = self.directed_lengths.get(db, None)
                    if L is not None:
                        f.write(f"{db} -> idx {idx}, L={L:.6f} µm\n")
                    else:
                        f.write(f"{db} -> idx {idx}, L=(missing)\n")
            except Exception:
                f.write("(unavailable)\n")
    
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
        info_str = f"Nv={self.n_vertices_total}_Nb={self.n_internal_bonds}_Nl={self.n_leads}"
    
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
    #                     Computation of the S-matrix
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
    
        This submatrix describes how waves scatter between internal directed bonds.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber (µm^-1).
    
        Returns
        -------
        Sigma_aa : ndarray (2B × 2B)
        """
        Sigma_aa = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)
    
        # Loop over all directed bonds explicitly (no reliance on enumerate order)
        for dbond_in in self.directed_bonds:
            src, dst = dbond_in   # directed bond: src → dst
            v = dst[0]            # vertex where the wave arrives
            port_in = dst[1]      # incoming port at vertex v
    
            # Get matrix for this vertex
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
    
            # Index for this input directed bond (robust lookup)
            idx_in = self.directed_bond_index[dbond_in]
    
            # For each outgoing port at vertex v
            for port_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][port_out]
    
                # Only handle internal bonds here
                if conn is not None:
                    neighbor, n_port = conn
                    dbond_out = ((v, port_out), (neighbor, n_port))
    
                    # Get correct outgoing index from mapping
                    idx_out = self.directed_bond_index[dbond_out]
    
                    # Fill Σ_{αα}[out, in]
                    Sigma_aa[idx_out, idx_in] = sigma_v[port_out, port_in]
    
                # If conn is None → this is a lead (Σ_{βα} handles that)
        return Sigma_aa
  
    def _build_sigma_alpha_betahat(self, k):
        """
        Build the scattering block Σ_{α β̂}.
    
        This matrix describes how an incoming lead injects into internal directed bonds.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber (µm^-1).
    
        Returns
        -------
        Sigma_ab : ndarray (2B × L)
            Rows: internal directed bonds, Columns: incoming leads.
        """
        Sigma_ab = np.zeros((self.n_directed_bonds, self.n_leads), dtype=complex)
    
        for lead_idx, (v, p_in) in enumerate(self.leads):
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
    
            # Scatter from this lead port to ALL outgoing ports of v
            for p_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p_out]
    
                if conn is not None:
                    # Outgoing port connects to an internal directed bond
                    neighbor, n_port = conn
                    dbond_out = ((v, p_out), (neighbor, n_port))
                    idx_out = self.directed_bond_index[dbond_out]
    
                    Sigma_ab[idx_out, lead_idx] = sigma_v[p_out, p_in]
    
                # If conn is None → outgoing to another lead, handled in Σ_{ββ̂}
        return Sigma_ab
    
    def _build_sigma_beta_alpha(self, k):
        """
        Build the scattering block Σ_{β α}.
    
        This matrix describes how internal directed bonds send energy into outgoing leads.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber (µm^-1).
    
        Returns
        -------
        Sigma_ba : ndarray (L × 2B)
            Rows: outgoing leads, Columns: internal directed bonds.
        """
        Sigma_ba = np.zeros((self.n_leads, self.n_directed_bonds), dtype=complex)
    
        for dbond_in in self.directed_bonds:
            src, dst = dbond_in
            v = dst[0]        # vertex where the bond ends
            port_in = dst[1]
    
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
    
            # ✅ Use correct index for the input bond
            idx_in = self.directed_bond_index[dbond_in]
    
            # Look at all outgoing ports from this vertex
            for p_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p_out]
    
                if conn is None:  # outgoing to a lead
                    try:
                        lead_idx = self.leads.index((v, p_out))
                    except ValueError:
                        raise RuntimeError(f"Lead (v={v}, port={p_out}) not found in self.leads")
    
                    # ✅ Fill matrix with the correct column index
                    Sigma_ba[lead_idx, idx_in] = sigma_v[p_out, port_in]
    
        return Sigma_ba

    def _build_sigma_beta_betahat(self, k):
        """
        Build the direct lead-to-lead scattering block Σ_{β β̂}.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber (µm^-1).
    
        Returns
        -------
        Sigma_bb : ndarray (L × L)
            Rows: outgoing leads, Columns: incoming leads.
        """
        Sigma_bb = np.zeros((self.n_leads, self.n_leads), dtype=complex)
    
        for lead_in_idx, (v, p_in) in enumerate(self.leads):
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
    
            for p_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p_out]
    
                if conn is None:
                    # This outgoing port is ALSO a lead
                    lead_out_idx = self.leads.index((v, p_out))
    
                    Sigma_bb[lead_out_idx, lead_in_idx] = sigma_v[p_out, p_in]
        return Sigma_bb
    
    def compute_S_matrix(self, k):
        """
        Compute the scattering matrix S(k) for the open quantum graph.
    
        Parameters
        ----------
        k : float
            Vacuum wavenumber (µm^-1).
    
        Returns
        -------
        S : ndarray (L × L)
            Scattering matrix at wavenumber k.
        """
        # --- Step 1: Build propagation matrix ---
        D = self._build_propagation_matrix(k)
    
        # --- Step 2: Build Σ-blocks ---
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        Sigma_ab = self._build_sigma_alpha_betahat(k)
        Sigma_ba = self._build_sigma_beta_alpha(k)
        Sigma_bb = self._build_sigma_beta_betahat(k)
    
        # --- Step 3: Internal map (subunitary) ---
        U_tilde = D @ Sigma_aa
    
        # --- Step 4: Multiple scattering series (resolvent) ---
        resolvent = np.linalg.inv(np.eye(self.n_directed_bonds) - U_tilde)
    
        # --- Step 5: Assemble full S-matrix ---
        S = Sigma_bb + Sigma_ba @ resolvent @ D @ Sigma_ab
        return S
    
    # ========================================================================
    #                     Computation of the resonances
    # ========================================================================
    def _results_npz_path(self,
                          re_vals: np.ndarray,
                          im_vals: np.ndarray,
                          eps_accept: float,
                          nullity_tol: float,
                          tol_step: float,
                          maxiter: int) -> str:
        """
        Build the canonical output path for the resonances results (.npz),
        using the same naming scheme as in _refine_candidates_and_finalize.
    
        Parameters
        ----------
        re_vals : np.ndarray
            Real-axis samples (defines N_re and the Re(k) range).
        im_vals : np.ndarray
            Imag-axis samples (defines N_im and the Im(k) range).
        eps_accept : float
            Acceptance threshold used for σ_min.
        nullity_tol : float
            SVD cutoff for degeneracy evaluation.
        tol_step : float
            Step tolerance used during 2D refinement.
        maxiter : int
            Max iterations used during 2D refinement.
    
        Returns
        -------
        out_path : str
            Absolute path to the .npz results file.
        """
        lam_min, lam_max = self.wavelength_range
        fname = (f"resonances_sigma2d_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_Nre_{re_vals.size}_Nim_{im_vals.size}"
                 f"_imMin_{im_vals.min():.6e}_imMax_{im_vals.max():.6e}"
                 f"_eps_{eps_accept:.1e}_nulltol_{nullity_tol:.1e}"
                 f"_tolstep_{tol_step:.1e}_maxiter_{int(maxiter)}.npz")
        return os.path.join(self.save_path, fname)


    def _U(self, k):
        """
        Build the internal quantum map U(k) = D(k) @ Σ_{αα}(k).
    
        Parameters
        ----------
        k : complex or float
            Vacuum wavenumber (in µm^-1). May be complex (for resonances).
    
        Returns
        -------
        U_k : ndarray (2B × 2B) complex128
            Internal map on directed bonds.
        """
        # Ensure complex64/128 arithmetic downstream is stable for SVD etc.
        k = complex(k)
    
        # Internal scattering block and propagation matrix
        Sigma_aa = self._build_sigma_alpha_alpha(k)   # shape (2B, 2B), complex
        D_k = self._build_propagation_matrix(k)       # shape (2B, 2B), complex diagonal
    
        # Explicit cast to complex128 to avoid mixed dtypes in downstream linear algebra
        U_k = (D_k.astype(np.complex128, copy=False) @
               Sigma_aa.astype(np.complex128, copy=False))
    
        return U_k
    
    def _sigma_min(self, k):
        """
        Return the smallest singular value of (I - U(k)) at the (possibly complex) wavenumber k.
    
        This is the objective g(k) used to locate resonances:
            g(k) = σ_min( I - U(k) ),  with  U(k) = D(k) @ Σ_{αα}(k).
    
        Parameters
        ----------
        k : complex or float
            Vacuum wavenumber (µm^-1). May be complex.
    
        Returns
        -------
        smin : float
            The smallest singular value (non-negative real). Returns np.inf on failure.
        """
        try:
            U = self._U(k)
            A = np.eye(self.n_directed_bonds, dtype=np.complex128) - U
            svals = np.linalg.svd(A, compute_uv=False)
            smin = float(svals[-1])  # σ_min is the last value (ascending order guaranteed)
            if not np.isfinite(smin):
                return np.inf
            # Numerical guard (should be ≥ 0 already, but just in case)
            if smin < 0.0:
                smin = abs(smin)
            return smin
        except Exception:
            # Any failure (nan in U, SVD issues, etc.) → treat as non-candidate
            return np.inf

    def _open_or_create_sigma_grid(self,
                                   N_re: int,
                                   N_im: int,
                                   im_min: float,
                                   im_max: float = 0.0,
                                   grid_dtype: str = "float32",
                                   overwrite: bool = False,
                                   verbose: bool = True):
        """
        Create or open a memmapped .npy grid to store g(k) = σ_min(I - U(k)) on a
        (N_im × N_re) grid over k = Re + i Im.
    
        The filename encodes ONLY:
            lam_min, lam_max, N_re, N_im, im_min, im_max
    
        The array is initialized with NaNs when created (to support resumability).
        Existing files are reused unless 'overwrite=True'.
    
        Parameters
        ----------
        N_re : int
            Number of samples along Re(k) (columns).
        N_im : int
            Number of samples along Im(k) (rows).
        im_min : float
            Lower bound for Im(k) (negative).
        im_max : float, default 0.0
            Upper bound for Im(k), usually 0.
        grid_dtype : {"float32", "float64"}
            Storage dtype for the grid file. Computations still use float64.
        overwrite : bool
            If True, (re-)create the grid file even if it exists.
        verbose : bool
            If True, print informative messages.
    
        Returns
        -------
        mem : numpy.memmap
            Memmapped array of shape (N_im, N_re), dtype=grid_dtype, filled with NaNs if new.
        re_vals : np.ndarray (float64)
            Re(k) axis of length N_re (ascending).
        im_vals : np.ndarray (float64)
            Im(k) axis of length N_im (descending from im_max to im_min).
        grid_path : str
            Absolute path to the .npy grid file.
        """
        # --- 1) Axes from wavelength_range & imag bounds --------------------------
        lam_min, lam_max = self.wavelength_range
        k_min = 2 * np.pi / lam_max  # longer λ → smaller k
        k_max = 2 * np.pi / lam_min  # shorter λ → larger  k
    
        re_vals = np.linspace(k_min, k_max, N_re, dtype=np.float64)
        # Descending Im so plots naturally go from 0 downwards
        im_vals = np.linspace(im_max, im_min, N_im, dtype=np.float64)
    
        # --- 2) Filename (ONLY the requested parameters) --------------------------
        fname = (f"sigma_grid_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                 f"_Nre_{N_re}_Nim_{N_im}"
                 f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}.npy")
        grid_path = os.path.join(self.save_path, fname)
    
        # --- 3) Create/open memmap ------------------------------------------------
        os.makedirs(self.save_path, exist_ok=True)
    
        if overwrite and os.path.exists(grid_path):
            if verbose:
                print(f"Overwriting existing σ_min grid: {grid_path}")
            os.remove(grid_path)
    
        if not os.path.exists(grid_path):
            if verbose:
                print(f"Creating σ_min grid: {grid_path}")
                print(f"  shape = ({N_im}, {N_re}), dtype = {grid_dtype}")
            mem = np.lib.format.open_memmap(grid_path, mode="w+",
                                            dtype=grid_dtype, shape=(N_im, N_re))
            # Initialize with NaNs for resumability (float32 NaN is fine if dtype=float32)
            mem[:] = np.nan
            mem.flush()
        else:
            if verbose:
                print(f"Reusing existing σ_min grid: {grid_path}")
            # Open read/write; shape & dtype consistency are ensured by .npy header
            mem = np.lib.format.open_memmap(grid_path, mode="r+")
            # Sanity checks on shape/dtype
            if mem.shape != (N_im, N_re):
                raise ValueError(f"Existing grid shape {mem.shape} does not match "
                                 f"requested {(N_im, N_re)}.")
            if str(mem.dtype) != grid_dtype:
                if verbose:
                    print(f"⚠ dtype mismatch: file has {mem.dtype}, requested {grid_dtype}. "
                          f"Proceeding with on-disk dtype {mem.dtype}.")
                # We keep the on-disk dtype to avoid rewriting; computations upstream stay float64.
    
        return mem, re_vals, im_vals, grid_path

    def _fill_sigma_grid(self,
                         mem,                 # numpy.memmap from _open_or_create_sigma_grid
                         re_vals: np.ndarray, # shape (N_re,)
                         im_vals: np.ndarray, # shape (N_im,)
                         grid_path: str,
                         verbose: bool = True,
                         flush_every_rows: int = 1,
                         report_every_pct: int = 5):
        """
        Evaluate g(k) = σ_min(I - U(k)) across the (N_im × N_re) grid and store
        the results in the provided memmapped array 'mem'. This function supports
        clean resumability: rows that contain no NaNs are skipped.
    
        Parameters
        ----------
        mem : numpy.memmap
            Memmapped array of shape (N_im, N_re), dtype float32/float64.
        re_vals : np.ndarray
            Real-axis samples, ascending, length N_re.
        im_vals : np.ndarray
            Imag-axis samples, typically descending from 0 to im_min, length N_im.
        grid_path : str
            Path to the .npy file (used for logging).
        verbose : bool
            Print progress/status messages.
        flush_every_rows : int
            Flush memmap to disk after this many completed rows (default: every row).
        report_every_pct : int
            Emit a progress line whenever the overall completion percent hits a multiple
            of this value (default: 5%%).
    
        Returns
        -------
        None
        """
        N_im, N_re = mem.shape
    
        if verbose:
            print(f"Evaluating σ_min grid row-by-row → {grid_path}")
            print(f"  Total rows: {N_im}, columns per row: {N_re}")
            # Compute initial completion
            done_rows = int(np.sum(~np.isnan(mem).all(axis=1)))
            pct0 = int(done_rows * 100 / max(1, N_im))
            print(f"  Resuming: {done_rows}/{N_im} rows already completed ({pct0}%).")
    
        rows_completed_since_flush = 0
        last_report_pct = -1
    
        for i in range(N_im):
            row = mem[i]  # view
    
            # Skip row if fully computed (no NaNs)
            if not np.isnan(row).any():
                continue
    
            im = im_vals[i]
            if verbose:
                print(f"  Row {i+1}/{N_im}: Im(k) = {im:.6e} — computing σ_min along Re...")
    
            # Compute only NaN entries (supports partially-computed rows)
            nan_mask = np.isnan(row)
            if nan_mask.all():
                # Fast path: compute entire row
                for j in range(N_re):
                    k = re_vals[j] + 1j * im
                    row[j] = self._sigma_min(k)
            else:
                # Sparse fill for remaining NaNs
                cols_to_do = np.where(nan_mask)[0]
                for j in cols_to_do:
                    k = re_vals[j] + 1j * im
                    row[j] = self._sigma_min(k)
    
            rows_completed_since_flush += 1
    
            # Periodic flush to ensure durability
            if rows_completed_since_flush >= flush_every_rows:
                mem.flush()
                rows_completed_since_flush = 0
    
            # Progress report
            if verbose and report_every_pct > 0:
                done_rows = int(np.sum(~np.isnan(mem).any(axis=1)))  # rows with no NaNs
                pct = int(done_rows * 100 / max(1, N_im))
                if pct % report_every_pct == 0 and pct != last_report_pct:
                    print(f"    Progress: {pct:3d}% (rows {done_rows}/{N_im})", end="\r")
                    last_report_pct = pct
    
        # Final flush
        mem.flush()
        if verbose:
            print("\nσ_min grid evaluation complete and flushed to disk.")
            
    def _find_candidates_from_sigma_grid(self,
                                         mem,                 # numpy.memmap (N_im, N_re) with σ_min values
                                         re_vals: np.ndarray, # shape (N_re,)
                                         im_vals: np.ndarray, # shape (N_im,)
                                         neighbor_cols: int = 1,
                                         neighbor_rows: int = 1,
                                         skip_edges: bool = True,
                                         verbose: bool = True):
        """
        Identify 2D candidate minima from the σ_min grid using a two-stage rule:
          1) Per-row 1D local minima along Re(k)
          2) Promote to 2D candidate if also minimal vs. neighboring rows (±neighbor_rows)
             at the same and ±neighbor_cols columns.
    
        Parameters
        ----------
        mem : numpy.memmap
            σ_min grid, shape (N_im, N_re), fully computed (no NaNs in used rows).
        re_vals : np.ndarray
            Real-axis samples, length N_re.
        im_vals : np.ndarray
            Imag-axis samples, length N_im.
        neighbor_cols : int
            Number of columns left/right to check in neighboring rows (default: 1).
        neighbor_rows : int
            Number of rows up/down to check (default: 1).
        skip_edges : bool
            If True, ignore minima that lie within 'neighbor_cols' of left/right edges
            or within 'neighbor_rows' of top/bottom edges (reduces spurious edge picks).
        verbose : bool
            Print summary counts.
    
        Returns
        -------
        cand_indices : np.ndarray, shape (M, 2), dtype=int
            Array of (i, j) grid indices promoted as 2D candidates.
        k0_list : np.ndarray, shape (M,), dtype=complex128
            Corresponding complex initial guesses k0 = re_vals[j] + 1j * im_vals[i].
        """
        g = mem  # alias
        N_im, N_re = g.shape
    
        if verbose:
            print("Selecting candidates from σ_min grid:")
            print(f"  Grid shape: rows N_im={N_im}, cols N_re={N_re}")
            print(f"  neighbor_rows={neighbor_rows}, neighbor_cols={neighbor_cols}, skip_edges={skip_edges}")
    
        # ------------- 1) Per-row 1D local minima along Re -----------------------
        row_minima = []  # list of (i, j) pairs (pre-promotion)
        j_lo = neighbor_cols if skip_edges else 1
        j_hi = N_re - 1 - (neighbor_cols if skip_edges else 1)
    
        for i in range(N_im):
            row = g[i]
            # If row has NaNs (incomplete), skip (should not happen if grid evaluation finished)
            if np.isnan(row).any():
                continue
    
            # Identify strict local minima: g[i,j] < g[i,j-1] and < g[i,j+1]
            for j in range(j_lo, j_hi):
                gj = row[j]
                if gj < row[j - 1] and gj < row[j + 1]:
                    row_minima.append((i, j))
    
        if verbose:
            print(f"  Found {len(row_minima)} per-row local minima before 2D check.")
    
        # ------------- 2) Promote to 2D candidates via cross-row check -----------
        cand = []
        i_lo = neighbor_rows if skip_edges else 0
        i_hi = N_im - 1 - (neighbor_rows if skip_edges else 0)
    
        for (i, j) in row_minima:
            if i < i_lo or i > i_hi:
                # Skip edges if requested
                continue
    
            val = g[i, j]
            # Gather neighborhood values in neighboring rows at same/±neighbor_cols columns
            is_minimal = True
            for di in range(-neighbor_rows, neighbor_rows + 1):
                if di == 0:
                    continue  # current row already tested in 1D
                ii = i + di
                # column neighborhood in that neighbor row
                j_start = max(j - neighbor_cols, 0)
                j_stop  = min(j + neighbor_cols, N_re - 1)
                neigh_vals = g[ii, j_start:j_stop + 1]
                if np.any(val > neigh_vals):
                    is_minimal = False
                    break
    
            if is_minimal:
                cand.append((i, j))
    
        cand_indices = np.array(cand, dtype=int)
        if cand_indices.size == 0:
            if verbose:
                print("  No 2D candidates promoted.")
            return cand_indices, np.array([], dtype=np.complex128)
    
        # ------------- Build initial guesses k0 for refinement --------------------
        k0_list = np.array([re_vals[j] + 1j * im_vals[i] for (i, j) in cand_indices],
                           dtype=np.complex128)
    
        if verbose:
            print(f"  Promoted {len(cand_indices)} 2D candidates.")
    
        return cand_indices, k0_list

    def _refine_candidate_2d(self,
                             k0: complex,
                             re_vals: np.ndarray,
                             im_vals: np.ndarray,
                             refine_box_mult_re: int = 2,
                             refine_box_mult_im: int = 2,
                             tol_step: float = 1e-12,
                             maxiter: int = 300,
                             verbose: bool = False):
        """
        Refine a 2D candidate starting from k0 within a small bounded box
        using a derivative-free optimizer.
    
        Minimizes g(k) = σ_min(I - U(k)) over k = x + i y with bounds:
          x ∈ [x0 ± refine_box_mult_re * Δx],  y ∈ [y0 ± refine_box_mult_im * Δy],
        where Δx and Δy are the grid steps from (re_vals, im_vals).
    
        Parameters
        ----------
        k0 : complex
            Initial guess (from grid) for the resonance.
        re_vals : np.ndarray
            Real-axis samples (length N_re), used to infer Δx and for clamping bounds.
        im_vals : np.ndarray
            Imag-axis samples (length N_im), used to infer Δy and for clamping bounds.
        refine_box_mult_re : int
            Half-width in units of Δx for the Re bound box (default: 2).
        refine_box_mult_im : int
            Half-width in units of Δy for the Im bound box (default: 2).
        tol_step : float
            Target step tolerance for the optimizer (mapped to optimizer xtol).
        maxiter : int
            Maximum number of iterations for the local optimizer.
        verbose : bool
            If True, prints a short refinement log.
    
        Returns
        -------
        k_star : complex
            Refined complex wavenumber.
        g_star : float
            σ_min at the refined point.
        success : bool
            True if optimizer reported success; False otherwise (warning emitted).
        """
    
        # --- 1) Initial point and grid steps -------------------------------------
        x0 = float(np.real(k0))
        y0 = float(np.imag(k0))
    
        # Grid steps (assume uniform sampling as constructed)
        if re_vals.size >= 2:
            dx = float(re_vals[1] - re_vals[0])
        else:
            dx = abs(x0) * 1e-12 + 1e-12  # fallback tiny step
    
        if im_vals.size >= 2:
            # im_vals is typically descending (0 → negative), but step magnitude is positive
            dy = float(abs(im_vals[1] - im_vals[0]))
        else:
            dy = abs(y0) * 1e-12 + 1e-12  # fallback tiny step
    
        # --- 2) Bounded box around k0 (clamped to overall domain) ----------------
        x_lo = max(x0 - refine_box_mult_re * dx, float(re_vals.min()))
        x_hi = min(x0 + refine_box_mult_re * dx, float(re_vals.max()))
        y_lo = max(y0 - refine_box_mult_im * dy, float(im_vals.min()))
        y_hi = min(y0 + refine_box_mult_im * dy, float(im_vals.max()))
    
        # Ensure valid bounds (avoid zero-width)
        if not (x_hi > x_lo):
            eps = max(1e-15, dx * 1e-6)
            x_lo, x_hi = x0 - eps, x0 + eps
        if not (y_hi > y_lo):
            eps = max(1e-15, dy * 1e-6)
            y_lo, y_hi = y0 - eps, y0 + eps
    
        bounds = Bounds([x_lo, y_lo], [x_hi, y_hi])
    
        # --- 3) Objective on R^2 --------------------------------------------------
        def obj_xy(xy):
            x, y = float(xy[0]), float(xy[1])
            k = x + 1j * y
            return self._sigma_min(k)
    
        x_init = np.array([x0, y0], dtype=np.float64)
    
        # --- 4) Local bounded optimization (Powell, no gradients) ----------------
        # Powell supports bounds in SciPy and is robust without gradients.
        res = minimize(
            obj_xy,
            x_init,
            method="Powell",
            bounds=bounds,
            options={
                "xtol": tol_step,
                "ftol": 1e-12,
                "maxiter": int(maxiter),
                "disp": False,
            },
        )
    
        x_star, y_star = float(res.x[0]), float(res.x[1])
        k_star = complex(x_star, y_star)
        g_star = float(res.fun)
        success = bool(res.success)
    
        # --- 5) Reporting / warnings ---------------------------------------------
        if verbose:
            iters = res.nit if hasattr(res, "nit") else None
            msg = res.message if hasattr(res, "message") else ""
            print(f"    refine @ k0={x0:+.6e}{y0:+.6e}i → "
                  f"k*={x_star:+.6e}{y_star:+.6e}i, g*={g_star:.3e}, "
                  f"success={success}, iters={iters}, msg={msg}")
    
        if not success:
            warnings.warn(
                f"Refinement did not converge (success={success}) at k0="
                f"{x0:+.6e}{y0:+.6e}i; result k*={x_star:+.6e}{y_star:+.6e}i, "
                f"g*={g_star:.3e}. Consider loosening tol_step or increasing maxiter.",
                RuntimeWarning
            )
        else:
            # If optimizer hit iteration cap, SciPy usually sets success=False,
            # but in case it sets success=True with high nit, we can still notify.
            if hasattr(res, "nit") and int(res.nit) >= int(maxiter):
                warnings.warn(
                    f"Refinement reached maxiter={maxiter} at k0="
                    f"{x0:+.6e}{y0:+.6e}i (k*={x_star:+.6e}{y_star:+.6e}i, g*={g_star:.3e}).",
                    RuntimeWarning
                )
    
        return k_star, g_star, success
    
    def _refine_candidates_and_finalize(self,
                                        k0_list: np.ndarray,   # complex initial guesses from _find_candidates_from_sigma_grid
                                        re_vals: np.ndarray,
                                        im_vals: np.ndarray,
                                        eps_accept: float = 1e-4,
                                        tol_step: float = 1e-12,
                                        maxiter: int = 300,
                                        refine_box_mult_re: int = 2,
                                        refine_box_mult_im: int = 2,
                                        nullity_tol: float = 1e-4,
                                        save_results: bool = True,
                                        overwrite_results: bool = False,
                                        verbose: bool = True,
                                        out_path: str | None = None):
        """
        Batch-refine 2D candidates, accept those with σ_min ≤ eps_accept, compute degeneracies,
        sort results, save .npz, and set class attributes.
    
        Parameters
        ----------
        k0_list : np.ndarray (complex)
            Initial guesses for resonances.
        re_vals, im_vals : np.ndarray
            Axes used to define the grid, for computing Δx, Δy and bounds.
        eps_accept : float
            Acceptance threshold on σ_min at the refined point.
        tol_step : float
            Step tolerance for the local 2D refinement.
        maxiter : int
            Max iterations for the local 2D refinement.
        refine_box_mult_re : int
            Half-width of the refinement box in Re bins (± multipliers of grid Δx).
        refine_box_mult_im : int
            Half-width of the refinement box in Im bins (± multipliers of grid Δy).
        nullity_tol : float
            Absolute SVD cutoff for degeneracy (nullity) evaluation.
        save_results : bool
            If True, save spectrum & degeneracies into a .npz.
        overwrite_results : bool
            If True, overwrite existing .npz if present.
        verbose : bool
            Print refinement progress and summaries.
        out_path : str or None
            If provided, save to this exact path; otherwise build one from parameters.
    
        Returns
        -------
        spectrum : np.ndarray (complex)
            Sorted refined resonance positions.
        degeneracies : np.ndarray (int)
            Same length as spectrum.
        """
    
        if k0_list.size == 0:
            if verbose:
                print("No candidates to refine. Returning empty arrays.")
            self.spectrum = np.array([], dtype=np.complex128)
            self.degeneracies = np.array([], dtype=int)
            self.degenerate_spectrum = np.array([], dtype=np.complex128)
            return self.spectrum, self.degeneracies
    
        # Domain bounds for safety (consistent with grid axes)
        k_re_min = float(re_vals.min())
        k_re_max = float(re_vals.max())
        k_im_min = float(im_vals.min())
        k_im_max = float(im_vals.max())
    
        # --- 1) Refine all candidates --------------------------------------------
        if verbose:
            print(f"Refining {k0_list.size} candidates with tol_step={tol_step}, maxiter={maxiter} ...")
    
        refined = []
        accepted = []
    
        for idx, k0 in enumerate(k0_list, 1):
            if verbose and (idx % max(1, k0_list.size // 10) == 0 or idx == 1 or idx == k0_list.size):
                print(f"  Candidate {idx}/{k0_list.size}: starting at k0 = "
                      f"{np.real(k0):+.6e}{np.imag(k0):+.6e}i")
    
            k_star, g_star, ok = self._refine_candidate_2d(
                k0=k0,
                re_vals=re_vals,
                im_vals=im_vals,
                refine_box_mult_re=refine_box_mult_re,
                refine_box_mult_im=refine_box_mult_im,
                tol_step=tol_step,
                maxiter=maxiter,
                verbose=False
            )
            refined.append((k_star, g_star, ok))
    
            # Apply acceptance criteria
            if ok and (g_star <= eps_accept):
                # Ensure within the global domain box (clamped by _refine_candidate_2d, but double-check)
                if (k_re_min <= np.real(k_star) <= k_re_max) and (k_im_min <= np.imag(k_star) <= k_im_max):
                    accepted.append(k_star)
    
        refined = np.array(refined, dtype=object)
        accepted = np.array(accepted, dtype=np.complex128)
    
        if verbose:
            n_ok = int(np.sum([ok for *_, ok in refined]))
            n_acc = accepted.size
            print(f"  Refinement done: {n_ok}/{k0_list.size} converged; {n_acc} accepted (σ_min ≤ {eps_accept:g}).")
    
        if accepted.size == 0:
            if verbose:
                print("No accepted resonances. Returning empty arrays.")
            self.spectrum = np.array([], dtype=np.complex128)
            self.degeneracies = np.array([], dtype=int)
            self.degenerate_spectrum = np.array([], dtype=np.complex128)
            return self.spectrum, self.degeneracies
    
        # --- 2) Sort by Re asc, then Im desc -------------------------------------
        order = np.lexsort((-np.imag(accepted), np.real(accepted)))
        spectrum = accepted[order]
    
        # --- 3) Degeneracy via nullity of (I - U(k*)) -----------------------------
        def _nullity_at_k_complex(kc: complex, atol: float) -> int:
            """Nullity of (I - U(kc)) with absolute SVD threshold 'atol'."""
            U = self._U(kc)
            A = np.eye(U.shape[0], dtype=np.complex128) - U
            s = np.linalg.svd(A, compute_uv=False)
            return int(np.sum(s <= atol))
    
        degeneracies = np.empty(spectrum.size, dtype=int)
        zero_deg_indices = []
    
        if verbose:
            print("Computing degeneracies (nullity of I - U(k*)) ...")
    
        for i, k_star in enumerate(spectrum):
            d = _nullity_at_k_complex(k_star, atol=nullity_tol)
            degeneracies[i] = d
            if d == 0:
                zero_deg_indices.append(i)
    
        if zero_deg_indices:
            # Show up to a few examples for guidance
            examples = ", ".join(f"{spectrum[i].real:.8g}{spectrum[i].imag:+.8g}i" for i in zero_deg_indices[:5])
            tail = "" if len(zero_deg_indices) <= 5 else f" (and {len(zero_deg_indices)-5} more)"
            warnings.warn(
                "Some refined resonances have nullity 0. This often means the refinement step "
                f"tolerance is too large or 'nullity_tol' is too small. Consider decreasing "
                f"tol_step (e.g., {tol_step:.1e} → {max(tol_step/10, 1e-20):.1e}) or increasing "
                f"nullity_tol (e.g., {nullity_tol:.1e} → {max(nullity_tol*3, 1e-12):.1e}). "
                f"Examples: k = {examples}{tail}.",
                RuntimeWarning
            )
    
        # Expanded degenerate spectrum (not saved, but kept for convenience)
        if spectrum.size:
            degenerate_spectrum = np.repeat(spectrum, degeneracies)
        else:
            degenerate_spectrum = np.array([], dtype=np.complex128)
    
        # --- 4) Save results (npz) and set class attributes -----------------------
        self.spectrum = spectrum
        self.degeneracies = degeneracies
        self.degenerate_spectrum = degenerate_spectrum
    
        if save_results:
            # Use provided out_path if any; otherwise build a canonical one
            if out_path is None:
                out_path = self._results_npz_path(
                    re_vals=re_vals, im_vals=im_vals,
                    eps_accept=eps_accept, nullity_tol=nullity_tol,
                    tol_step=tol_step, maxiter=maxiter
                )
    
            if (not overwrite_results) and os.path.exists(out_path):
                if verbose:
                    print(f"Results file exists, not overwriting: {out_path}")
            else:
                os.makedirs(self.save_path, exist_ok=True)
                if verbose:
                    print(f"Saving spectrum & degeneracies to: {out_path}")
                np.savez(
                    out_path,
                    spectrum=spectrum,
                    degeneracies=degeneracies,
                    re_vals=re_vals,
                    im_vals=im_vals,
                    eps_accept=np.array(eps_accept, dtype=np.float64),
                    nullity_tol=np.array(nullity_tol, dtype=np.float64),
                    tol_step=np.array(tol_step, dtype=np.float64),
                    maxiter=np.array(maxiter, dtype=np.int64),
                )
    
        if verbose:
            print(f"Done. Found {spectrum.size} resonances. "
                  f"Total multiplicity (len degenerate_spectrum) = {degenerate_spectrum.size}.")
    
        return spectrum, degeneracies


    def find_resonances_sigma2d(self,
                                N_re: int = 500_000,
                                N_im: int = 100,
                                im_min: float = -1e-4,
                                im_max: float = 0.0,
                                eps_accept: float = 1e-4,
                                tol_step: float = 1e-12,
                                maxiter: int = 300,
                                refine_box_mult_re: int = 2,
                                refine_box_mult_im: int = 2,
                                nullity_tol: float = 1e-4,
                                grid_dtype: str = "float32",
                                save_grid: bool = True,
                                overwrite_grid: bool = False,
                                save_results: bool = True,
                                overwrite_results: bool = False,
                                ensure_grid: bool = False,
                                plot: bool = True,
                                verbose: bool = True):
        """
        Find complex resonances of the open quantum graph using the σ_min method
        on a 2D grid in the complex k-plane, with early-load of results and optional
        forced grid computation for heatmaps.
    
        Behavior:
          • If a matching results .npz exists and overwrite_results=False:
              - Load spectrum/degeneracies and return.
              - If plot=True:
                  - If ensure_grid=True: compute (or resume) σ_min grid, then plot heatmap + crosses.
                  - Else: if a matching grid exists, plot heatmap + crosses; otherwise plot crosses only.
          • Otherwise (no results or overwrite requested):
              - Compute (or resume) the σ_min grid.
              - Extract candidates, refine, accept, compute degeneracies, save .npz.
              - Plot heatmap + crosses if plot=True.
    
        Parameters
        ----------
        N_re, N_im : int
            Grid sizes along Re(k) and Im(k).
        im_min, im_max : float
            Imaginary-part search interval (im_min < 0 ≤ im_max).
        eps_accept : float
            Acceptance threshold on σ_min at the refined point.
        tol_step : float
            Step tolerance for local 2D refinement (Powell).
        maxiter : int
            Max iterations for local 2D refinement.
        refine_box_mult_re, refine_box_mult_im : int
            Refinement box half-widths in units of grid steps (± multipliers).
        nullity_tol : float
            Absolute SVD cutoff for degeneracy (nullity) evaluation.
        grid_dtype : {"float32","float64"}
            Storage dtype for σ_min grid cache (.npy). Computations remain float64.
        save_grid : bool
            Keep the σ_min grid cached as .npy (True recommended for resumability).
        overwrite_grid : bool
            Recompute the σ_min grid even if a matching .npy exists.
        save_results : bool
            Save spectrum & degeneracies to .npz.
        overwrite_results : bool
            Recompute spectrum & degeneracies even if a matching .npz exists.
        ensure_grid : bool
            When results are loaded from .npz, force computing the σ_min grid so that
            a heatmap can be plotted. Ignored if recomputing results anyway.
        plot : bool
            Plot results (heatmap if grid present or ensured; otherwise crosses only).
        verbose : bool
            Print progress and status messages.
    
        Returns
        -------
        spectrum : np.ndarray (complex)
        degeneracies : np.ndarray (int)
        """
        # ---- Build axes (to derive canonical filenames) --------------------------
        lam_min, lam_max = self.wavelength_range
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
        re_vals = np.linspace(k_min, k_max, N_re, dtype=np.float64)
        im_vals = np.linspace(im_max, im_min, N_im, dtype=np.float64)  # descending
    
        # Canonical path for the results file
        out_path = self._results_npz_path(
            re_vals=re_vals, im_vals=im_vals,
            eps_accept=eps_accept, nullity_tol=nullity_tol,
            tol_step=tol_step, maxiter=maxiter
        )
    
        # ---- Early-load of results if available and not overwriting -------------
        if (not overwrite_results) and os.path.exists(out_path):
            if verbose:
                print(f"✓ Loaded resonances from file: {out_path}")
            data = np.load(out_path)
            spectrum = np.array(data["spectrum"], dtype=np.complex128)
            degeneracies = np.array(data["degeneracies"], dtype=int)
            # Update class attributes
            self.spectrum = spectrum
            self.degeneracies = degeneracies
            self.degenerate_spectrum = np.repeat(spectrum, degeneracies) if spectrum.size else np.array([], dtype=np.complex128)
    
            # ---- Plotting path when results are loaded --------------------------
            if plot:
                # Determine matching σ_min grid filename (no need to compute axes again)
                grid_mem = None
                grid_path = os.path.join(
                    self.save_path,
                    (f"sigma_grid_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                     f"_Nre_{N_re}_Nim_{N_im}"
                     f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}.npy")
                )
    
                if ensure_grid:
                    if verbose:
                        print("ensure_grid=True → computing/resuming σ_min grid for heatmap.")
                    mem, re_vals2, im_vals2, grid_path2 = self._open_or_create_sigma_grid(
                        N_re=N_re, N_im=N_im, im_min=im_min, im_max=im_max,
                        grid_dtype=grid_dtype, overwrite=overwrite_grid, verbose=verbose
                    )
                    self._fill_sigma_grid(mem, re_vals2, im_vals2, grid_path2, verbose=verbose)
                    grid_mem = mem  # use for plotting
                    re_vals_plot, im_vals_plot = re_vals2, im_vals2
                else:
                    if os.path.exists(grid_path):
                        if verbose:
                            print(f"σ_min grid found: {grid_path} → plotting heatmap + crosses.")
                        grid_mem = np.lib.format.open_memmap(grid_path, mode="r")
                        re_vals_plot, im_vals_plot = re_vals, im_vals
                    else:
                        if verbose:
                            print("σ_min grid not found and ensure_grid=False → plotting crosses only.")
    
                # --- Plot
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 8))
                if grid_mem is not None:
                    imshow_obj = ax.imshow(grid_mem,
                       extent=[re_vals_plot[0], re_vals_plot[-1], im_vals_plot[-1], im_vals_plot[0]],
                       aspect='auto',
                       cmap="inferno_r",
                       norm=LogNorm(vmin=grid_mem[np.isfinite(grid_mem)].min(), vmax=grid_mem[np.isfinite(grid_mem)].max()))
                    cbar = plt.colorbar(imshow_obj, ax=ax)
                    cbar.set_label(r"$\sigma_{\min}(I-U(k))$")
                cross_color = 'red' if grid_mem is not None else 'black'
                ax.scatter(np.real(spectrum), np.imag(spectrum),
                           marker='x', color=cross_color, label='Resonances')
                ax.set_xlabel("Re(k) [µm$^{-1}$]")
                ax.set_ylabel("Im(k) [µm$^{-1}$]")
                plt.tight_layout()
                plt.show()
    
            return spectrum, degeneracies
    
        # ---- Otherwise: full pipeline (compute grid → candidates → refinement) ---
        if verbose:
            print("No matching results found or overwrite requested → computing σ_min grid and resonances.")
    
        # 1) Create or open σ_min grid cache (always needed for candidate extraction)
        mem, re_vals, im_vals, grid_path = self._open_or_create_sigma_grid(
            N_re=N_re,
            N_im=N_im,
            im_min=im_min,
            im_max=im_max,
            grid_dtype=grid_dtype,
            overwrite=overwrite_grid,
            verbose=verbose
        )
    
        # 2) Fill σ_min grid (resume-friendly)
        self._fill_sigma_grid(mem, re_vals, im_vals, grid_path, verbose=verbose)
    
        # 3) Candidate selection
        cand_indices, k0_list = self._find_candidates_from_sigma_grid(
            mem, re_vals, im_vals, verbose=verbose
        )
    
        # 4) Refinement, degeneracy, save results (uses precomputed out_path)
        spectrum, degeneracies = self._refine_candidates_and_finalize(
            k0_list=k0_list,
            re_vals=re_vals,
            im_vals=im_vals,
            eps_accept=eps_accept,
            tol_step=tol_step,
            maxiter=maxiter,
            refine_box_mult_re=refine_box_mult_re,
            refine_box_mult_im=refine_box_mult_im,
            nullity_tol=nullity_tol,
            save_results=save_results,
            overwrite_results=overwrite_results,
            verbose=verbose,
            out_path=out_path
        )
    
        # 5) Plot heatmap + crosses if requested
        if plot and spectrum.size > 0:
            import matplotlib.pyplot as plt
            
            # Use the freshly computed grid
            grid_mem = mem
            re_vals_plot = re_vals
            im_vals_plot = im_vals
    
            fig, ax = plt.subplots(figsize=(10, 8))
            imshow_obj = ax.imshow(grid_mem,
                       extent=[re_vals_plot[0], re_vals_plot[-1], im_vals_plot[-1], im_vals_plot[0]],
                       aspect='auto',
                       cmap="inferno_r",
                       norm=LogNorm(vmin=grid_mem[np.isfinite(grid_mem)].min(), vmax=grid_mem[np.isfinite(grid_mem)].max()))
            cbar = plt.colorbar(imshow_obj, ax=ax)
            cbar.set_label(r"$\sigma_{\min}(I-U(k))$")
            cross_color = 'red' if mem is not None else 'black'
            ax.scatter(np.real(spectrum), np.imag(spectrum),
                       marker='x', color=cross_color, label='Resonances')
            ax.set_xlabel("Re(k) [µm$^{-1}$]")
            ax.set_ylabel("Im(k) [µm$^{-1}$]")
            plt.tight_layout()
            plt.show()
    
        return spectrum, degeneracies


    # ========================================================================
    #                     Experimental resonance finding and plotting
    # ========================================================================
    # Experimental resonances are the (real) resonances we extract from the 
    # transmission spectra.
    
    def find_experimental_resonances(self,
                                     n_points=5_000_000,
                                     component=(1, 0),
                                     k_imag=1e-7,
                                     prominence=None,
                                     manual_lambda_peaks=None,
                                     save=True,
                                     overwrite=False,
                                     plot_zoom=True,
                                     verbose=True,
                                     file_format="pdf"):
        """
        Compute and store the experimental resonances of the graph by finding dips in |S_ij|²
        for a chosen component. The dips are identified as local minima after adding a small
        imaginary part to k to smooth the spectrum.
    
        Parameters
        ----------
        n_points : int
            Number of sampling points between lambda_min and lambda_max (default: 5,000,000).
        component : tuple
            Transmission channel to analyze (i, j) (default: (1, 0)).
        k_imag : float
            Small imaginary part added to k to smooth the dips (default: 1e-7).
        prominence : float or None
            Required prominence of the dips (passed to scipy.signal.find_peaks). If None, all local minima are considered.
        manual_lambda_peaks : list of floats or None
            Optional manual peaks (in λ, microns) to include in the detected resonances.
        save : bool
            If True, save the detected resonances and the plots (default: True).
        overwrite : bool
            If True, recompute even if a saved resonance file already exists.
        plot_zoom : bool
            If True, generate zoomed plots (default: True).
        verbose : bool
            If True, print progress updates.
        file_format : str
            File format for saving the figures (default: "pdf").
    
        Returns
        -------
        experimental_resonances : np.ndarray
            Array of detected resonances (k values in µm⁻¹).
        """
    
        # 1) ---- Build filename and check if already computed -------------------
        i, j = component
        lam_min, lam_max = self.wavelength_range
        prom_tag = f"prom{prominence:.1e}" if prominence is not None else "promNone"
        manual_count = len(manual_lambda_peaks) if manual_lambda_peaks is not None else 0
        manual_tag = f"manual{manual_count}"
        filename = (f"experimental_resonances_S{i}{j}_"
                    f"lam{lam_min:.3f}-{lam_max:.3f}_"
                    f"N{n_points}_"
                    f"kIm{k_imag:.1e}_{prom_tag}_{manual_tag}.npy")
        filepath = os.path.join(self.save_path, filename)
    
        if save and (not overwrite) and os.path.exists(filepath):
            if verbose:
                print(f"✓ Loaded experimental resonances from file: {filepath}")
            self.exp_spectrum = np.load(filepath)
            return self.exp_spectrum
    
        # 2) ---- Build k-grid and wavelength array ------------------------------
        k_min, k_max = 2 * np.pi / lam_max, 2 * np.pi / lam_min
        k_values = np.linspace(k_min, k_max, n_points) + 1j * k_imag
        lambda_values = 2 * np.pi / np.real(k_values)
    
        # 3) ---- Compute |S_ij|² spectrum ---------------------------------------
        if verbose:
            print(f"Computing |S[{i},{j}]|² on {n_points} points...")
    
        mod_sq = np.zeros(n_points, dtype=float)
        last_report = -1
    
        for idx, k in enumerate(k_values):
            S = self.compute_S_matrix(k)
            mod_sq[idx] = np.abs(S[i, j])**2
    
            if verbose:
                pct = int((idx + 1) / n_points * 100)
                if pct % 5 == 0 and pct != last_report:
                    print(f"   Progress: {pct:3d}%", end="\r")
                    last_report = pct
    
        if verbose:
            print("   Progress: 100% ✓")
            print("Spectrum computed.")
    
        # 4) ---- Find local minima (resonances) ---------------------------------
        peaks, _ = find_peaks(-mod_sq, prominence=prominence)
        k_resonances = np.real(k_values[peaks])  # store only the real part
    
        # 4b) ---- Add manual peaks (convert λ → k) ------------------------------
        if manual_lambda_peaks is not None and len(manual_lambda_peaks) > 0:
            manual_k = [2 * np.pi / lam for lam in manual_lambda_peaks]
            if verbose:
                print(f"✓ Adding {len(manual_k)} manually specified peaks.")
            k_resonances = np.concatenate([k_resonances, manual_k])
    
        if verbose:
            print(f"✓ Total resonances (auto + manual): {len(k_resonances)}")
    
        # Sort resonances by increasing k
        k_resonances = np.sort(k_resonances)
    
        # 5) ---- Save results ----------------------------------------------------
        if save:
            np.save(filepath, k_resonances)
            if verbose:
                print(f"✓ Experimental resonances saved to {filepath}")
    
        self.exp_spectrum = k_resonances
    
        # 6) ---- Plot full spectrum with resonance lines ------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(lambda_values, mod_sq, color="black", linewidth=1.2)
    
        # Draw red dashed lines at resonance wavelengths
        lambda_resonances = 2 * np.pi / k_resonances
        for lam in lambda_resonances:
            ax.axvline(lam, color="red", linestyle="--", linewidth=1)
    
        # Axis labels and style
        ax.set_xlabel("Wavelength (µm)", fontsize=22)
        ax.set_ylabel(rf"$|S_{{{i}{j}}}|^2$", fontsize=22)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
        ax.set_xlim(lam_min, lam_max)
    
        plt.tight_layout()
    
        if save:
            fig_name = (f"Spectrum_with_resonances_S{i}{j}_"
                        f"lam{lam_min:.3f}-{lam_max:.3f}_"
                        f"N{n_points}_kIm{k_imag:.1e}_{prom_tag}_{manual_tag}.{file_format}")
            fig_path = os.path.join(self.save_path, fig_name)
            fig.savefig(fig_path, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Full spectrum figure saved to {fig_path}")
    
        plt.show()
    
        # 7) ---- Zoomed plots (optional) ----------------------------------------
        lambda_range = lam_max - lam_min
        if plot_zoom and lambda_range > 0.02:
            center = 0.5 * (lam_max + lam_min)
            zoom_min, zoom_max = center - 0.01, center + 0.01
    
            # Zoom 1: No resonance lines
            fig_zoom, ax_zoom = plt.subplots(figsize=(10, 8))
            ax_zoom.plot(lambda_values, mod_sq, color="black", linewidth=1.2)
    
            ax_zoom.set_xlim(zoom_min, zoom_max)
            ax_zoom.set_xlabel("Wavelength (µm)", fontsize=22)
            ax_zoom.set_ylabel(rf"$|S_{{{i}{j}}}|^2$", fontsize=22)
            ax_zoom.minorticks_on()
            ax_zoom.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                                labelsize=22, top=True, right=True)
            ax_zoom.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                                top=True, right=True)
    
            plt.tight_layout()
    
            if save:
                zoom_name = (f"Zoom_no_resonances_S{i}{j}_lam{zoom_min:.3f}-{zoom_max:.3f}_"
                             f"N{n_points}_kIm{k_imag:.1e}_{prom_tag}_{manual_tag}.{file_format}")
                zoom_path = os.path.join(self.save_path, zoom_name)
                fig_zoom.savefig(zoom_path, format=file_format, dpi=300, bbox_inches="tight")
                if verbose:
                    print(f"✓ Zoom (no lines) saved to {zoom_path}")
    
            plt.show()
    
            # Zoom 2: With resonance lines
            fig_zoom2, ax_zoom2 = plt.subplots(figsize=(10, 8))
            ax_zoom2.plot(lambda_values, mod_sq, color="black", linewidth=1.2)
    
            for lam in lambda_resonances:
                ax_zoom2.axvline(lam, color="red", linestyle="--", linewidth=1)
    
            ax_zoom2.set_xlim(zoom_min, zoom_max)
            ax_zoom2.set_xlabel("Wavelength (µm)", fontsize=22)
            ax_zoom2.set_ylabel(rf"$|S_{{{i}{j}}}|^2$", fontsize=22)
            ax_zoom2.minorticks_on()
            ax_zoom2.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                                 labelsize=22, top=True, right=True)
            ax_zoom2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                                 top=True, right=True)
    
            plt.tight_layout()
    
            if save:
                zoom_name2 = (f"Zoom_with_resonances_S{i}{j}_lam{zoom_min:.3f}-{zoom_max:.3f}_"
                              f"N{n_points}_kIm{k_imag:.1e}_{prom_tag}_{manual_tag}.{file_format}")
                zoom_path2 = os.path.join(self.save_path, zoom_name2)
                fig_zoom2.savefig(zoom_path2, format=file_format, dpi=300, bbox_inches="tight")
                if verbose:
                    print(f"✓ Zoom (with lines) saved to {zoom_path2}")
    
            plt.show()
    
        elif plot_zoom:
            print("λ range already smaller than 0.2 µm → skipping zoom plots.")
    
        return self.exp_spectrum


    
    # ========================================================================
    #                          Spectral statistics
    # ========================================================================
    # -------------------------------------------------------------------------
    # Spectral statistics core utilities
    # -------------------------------------------------------------------------
    
    def _is_pure_directional(self):
        """
        Return True iff degeneracies are available and every level has deg == 2.
    
        Notes
        -----
        - Numerical resonances (with find_resonances) may populate:
            * self.degeneracies           (array of ints)
            * self.degenerate_spectrum    (duplicated spectrum)
          In this case, a fully directional closed-like behavior corresponds to all deg == 2.
    
        - Experimental peaks (with find_experimental_resonances) typically do NOT
          define degeneracies; we conceptually treat deg == 1 everywhere, hence this
          function will return False in that case.
        """
        return hasattr(self, "degeneracies") and (self.degeneracies.size > 0) and np.all(self.degeneracies == 2)

    def _stats_selection(self, mode="auto", use_experimental=False):
        """
        Decide which spectrum and Weyl scaling to use for spectral statistics.
    
        Parameters
        ----------
        mode : {"auto", "raw", "expanded", "directional"}
            - "auto":
                * if all deg==2 (pure directional), use RAW spectrum + Weyl × 1/2
                * else use EXPANDED spectrum if available
                * else fall back to RAW
            - "raw": always use the raw spectrum (no duplication)
            - "expanded": use the degenerate/expanded spectrum if available,
                          otherwise fall back to raw
            - "directional": force directional rule (raw + Weyl × 1/2)
        use_experimental : bool
            If True, use self.exp_spectrum as the source spectrum. In this case,
            degeneracies are not defined, so expanded == raw by construction.
    
        Returns
        -------
        spectrum_used : np.ndarray
        weyl_scale : float
        note : str
        method : {"raw", "expanded", "directional"}
        """
        # ---------------- EXPERIMENTAL PATH ----------------
        if use_experimental:
            if not hasattr(self, "exp_spectrum"):
                raise AttributeError(
                    "use_experimental=True but exp_spectrum is missing. "
                    "Run find_experimental_resonances() first."
                )
            spectrum_exp = np.asarray(self.exp_spectrum, float)
            if spectrum_exp.size == 0:
                raise ValueError("exp_spectrum is empty.")
    
            if mode == "raw":
                return spectrum_exp, 1.0, "Experimental raw spectrum (no duplication).", "raw"
    
            if mode == "expanded":
                return (
                    spectrum_exp,
                    1.0,
                    "Experimental: expanded = raw (no degeneracy info).",
                    "expanded",
                )
    
            if mode == "directional":
                return (
                    spectrum_exp,
                    0.5,
                    "Experimental directional assumption: Weyl × 1/2.",
                    "directional",
                )
    
            # auto on experimental data → treat as non-directional → expanded==raw
            return (
                spectrum_exp,
                1.0,
                "Experimental (auto): using expanded=raw (no degeneracy info).",
                "expanded",
            )
    
        # ---------------- NUMERICAL/DEFAULT PATH ----------------
        # RAW
        if mode == "raw":
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return (
                np.real(np.asarray(self.spectrum)),
                1.0,
                "Using raw spectrum (no duplication).",
                "raw",
            )
    
        # EXPANDED
        if mode == "expanded":
            if hasattr(self, "degenerate_spectrum"):
                return (
                    np.real(np.asarray(self.degenerate_spectrum)),
                    1.0,
                    "Using expanded (degenerate) spectrum.",
                    "expanded",
                )
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return (
                np.real(np.asarray(self.spectrum)),
                1.0,
                "No degeneracy info: expanded=raw spectrum.",
                "expanded",
            )
    
        # DIRECTIONAL
        if mode == "directional":
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return (
                np.real(np.asarray(self.spectrum)),
                0.5,
                "Directional assumption: raw spectrum; Weyl × 1/2.",
                "directional",
            )
    
        # AUTO
        if self._is_pure_directional():
            return (
                np.real(np.asarray(self.spectrum)),
                0.5,
                "Detected pure directional case: raw spectrum; Weyl × 1/2.",
                "directional",
            )
        else:
            if hasattr(self, "degenerate_spectrum"):
                return (
                    np.real(np.asarray(self.degenerate_spectrum)),
                    1.0,
                    "Non-directional: using expanded spectrum.",
                    "expanded",
                )
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            return (
                np.real(np.asarray(self.spectrum)),
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
        # Effective index n_eff may be a scalar or a callable of k
        if callable(self.neff):
            n_eff = self.neff(np.real(k))
        else:
            n_eff = self.neff
    
        return n_eff * (self.total_length * length_scale) / np.pi * np.real(k)
    
    def _compute_unfolded_spectrum(self, spectrum=None, mode=None, use_experimental=False):
        """
        Unfold a spectrum using the (optionally scaled) Weyl law.
        If spectrum is None, choose according to _stats_selection(mode, use_experimental).
    
        Parameters
        ----------
        spectrum : ndarray or None
            k values. If None, chosen via _stats_selection.
        mode : {"auto", "raw", "expanded", "directional", None}
            Selection policy; None -> use self.stats_mode if present, else "auto".
        use_experimental : bool
            If True, use self.exp_spectrum as the source inside _stats_selection.
    
        Returns
        -------
        ndarray
            Unfolded spectrum values N_weyl_scaled(k).
        """
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
    
        if spectrum is None:
            spectrum_used, weyl_scale, _, _ = self._stats_selection(
                mode=chosen_mode, use_experimental=use_experimental
            )
            spectrum = np.asarray(spectrum_used, dtype=float)
        else:
            spectrum = np.asarray(spectrum, dtype=float)
            # Even if spectrum is passed explicitly, we honor the chosen directional scaling
            _, weyl_scale, _, _ = self._stats_selection(
                mode=chosen_mode, use_experimental=use_experimental
            )
    
        spectrum = np.sort(np.real(spectrum))
        return self.N_weyl_scaled(spectrum, length_scale=weyl_scale)


    
    # -------------------------------------------------------------------------
    # Staircase function plotting
    # -------------------------------------------------------------------------
    def plot_staircase_function(self, plot_weyl=True, weyl_diff=False,
                                mode=None, use_experimental=False,
                                save_fig=False, file_format='pdf'):
        """
        Plot the staircase N(k) for spectral statistics.
    
        Parameters
        ----------
        plot_weyl : bool
            Overlay the (scaled) Weyl prediction curve.
        weyl_diff : bool
            If True, also plot N_Weyl(k) - N(k) vs index and vs wavelength.
        mode : {"auto", "raw", "expanded", "directional", None}
            Spectrum selection policy. None -> use self.stats_mode if present, else "auto".
        use_experimental : bool
            If True, use self.exp_spectrum (no degeneracies; expanded == raw).
        save_fig : bool
            Save figure(s) to self.save_path.
        file_format : str
            Format for saving ('pdf', 'svg', 'png', etc.). Default: 'pdf'.
        """
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
    
        # Choose spectrum + Weyl scaling (handles experimental/numerical)
        spectrum_used, weyl_scale, note, method = self._stats_selection(
            mode=chosen_mode, use_experimental=use_experimental
        )
    
        spectrum = np.sort(np.real(np.asarray(spectrum_used, dtype=float)))
        if spectrum.size == 0:
            raise ValueError("Empty spectrum after selection.")
    
        N_modes = spectrum.size
        N_k = np.arange(N_modes + 1)
        plot_spectrum = np.insert(spectrum, 0, spectrum[0])  # step start
        print(f"[Staircase:{method}] {note}")
    
        # --- Main staircase plot ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.step(plot_spectrum, N_k, where='post', label=r'$N(k)$', color='k', lw=2)
    
        # Optional Weyl overlay (anchored to first step)
        if plot_weyl:
            k_vals = np.linspace(spectrum[0], spectrum[-1], 1000)
            Nw = self.N_weyl_scaled(k_vals, length_scale=weyl_scale)
            Nw -= self.N_weyl_scaled(spectrum[0], length_scale=weyl_scale)
            Nw += 0.5
            ax.plot(k_vals, Nw, 'r--', lw=1.5, label=r'$N_{\mathrm{Weyl}}(k)$')
    
        # Titles & labels
        L_eff = self.total_length * weyl_scale
        tag = "__exp" if use_experimental else ""  # filename tag
        ax.set_title(fr"$L_{{\rm eff}} = {L_eff:.2f}\ \mu m$   ({method}{', exp' if use_experimental else ''})",
                     fontsize=20)
        ax.set_xlabel(r'$k$ (µm$^{-1}$)', fontsize=22)
        ax.set_ylabel(r'$N(k)$', fontsize=22)
        ax.set_xlim(spectrum[0], spectrum[-1])
        ax.set_ylim(0, N_modes)
        ax.legend(fontsize=16)
    
        # Style
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=18, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
        fig.tight_layout()
    
        if save_fig:
            base = f"staircase_function__{method}{tag}.{file_format}"
            filepath = os.path.join(self.save_path, base)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Staircase plot saved to: {filepath}")
    
        plt.show()
    
        # --- Optional Weyl deviation plots ---
        if weyl_diff:
            N_weyl_k = self.N_weyl_scaled(spectrum, length_scale=weyl_scale)
            N_weyl_k -= self.N_weyl_scaled(spectrum[0], length_scale=weyl_scale)
            N_weyl_k += 0.5
            N_k_mid = np.arange(1, N_modes + 1)  # step midpoints
            delta = N_weyl_k - N_k_mid
    
            # (a) Δ vs mode index
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.plot(np.arange(N_modes), delta, 'k-', lw=2)
            ax2.axhline(0, color='gray', linestyle='--')
            ax2.set_xlabel(r'Mode index', fontsize=22)
            ax2.set_ylabel(r'$N_{\mathrm{Weyl}}(k) - N(k)$', fontsize=22)
            ax2.set_xlim(0, N_modes - 1)
            ax2.minorticks_on()
            ax2.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                            labelsize=18, top=True, right=True)
            ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                            top=True, right=True)
            fig2.tight_layout()
    
            if save_fig:
                base2 = f"staircase_diff_index__{method}{tag}.{file_format}"
                filepath2 = os.path.join(self.save_path, base2)
                fig2.savefig(filepath2, format=file_format, dpi=300, bbox_inches='tight')
                print(f"✓ Staircase diff (index) saved to: {filepath2}")
            plt.show()
    
            # (b) Δ vs wavelength (nm)
            lambda_nm = 2 * np.pi / spectrum * 1e3
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ax3.plot(lambda_nm, delta, 'k-', lw=2)
            ax3.axhline(0, color='gray', linestyle='--')
            ax3.set_xlabel(r'Wavelength $\lambda$ (nm)', fontsize=22)
            ax3.set_ylabel(r'$N_{\mathrm{Weyl}}(k) - N(k)$', fontsize=22)
            ax3.set_xlim(np.min(lambda_nm), np.max(lambda_nm))
            ax3.minorticks_on()
            ax3.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                            labelsize=18, top=True, right=True)
            ax3.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                            top=True, right=True)
            fig3.tight_layout()
    
            if save_fig:
                base3 = f"staircase_diff_lambda__{method}{tag}.{file_format}"
                filepath3 = os.path.join(self.save_path, base3)
                fig3.savefig(filepath3, format=file_format, dpi=300, bbox_inches='tight')
                print(f"✓ Staircase diff (λ) saved to: {filepath3}")
            plt.show()


    # -------------------------------------------------------------------------
    # Short-range correlations: Nearest Neighbor Spacing Distribution (NNSD)
    # -------------------------------------------------------------------------
    def short_range_correlations(self, width=0.1, xlim=None,
                                 plot_wigner=['poissonian', 'goe'],
                                 cumulative_spacing=False,
                                 color_list=None,
                                 mode=None, 
                                 use_experimental=False,
                                 save_fig=False, 
                                 file_format='svg'):
        """
        Plot nearest-neighbor spacing distribution (NNSD) from the unfolded spectrum.
    
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
            Custom color/style list for Wigner curves. Each entry can be 'r-', 'b--', etc.
            Defaults to ['r-', 'b-.', 'g--', 'orange-', 'purple--'].
        mode : {"auto", "raw", "expanded", "directional", None}
            Spectrum selection policy. None -> use self.stats_mode if present, else "auto".
        use_experimental : bool
            If True, use self.exp_spectrum (no degeneracies; expanded == raw).
        save_fig : bool
            If True, saves the figure in self.save_path.
        file_format : str
            Output format for the saved figure (e.g., 'svg', 'png').
        """
        # --- 1) Choose spectrum + Weyl scaling (records method for filenames) ---
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(
            mode=chosen_mode, use_experimental=use_experimental
        )
    
        # Unfold using the same selection/scaling
        unfolded = self._compute_unfolded_spectrum(
            spectrum=None, mode=chosen_mode, use_experimental=use_experimental
        )
    
        # --- 2) Nearest-neighbor spacings --------------------------------------
        spacings = np.diff(np.sort(np.real(unfolded)))
        if spacings.size == 0:
            raise ValueError("Not enough levels to compute spacings.")
    
        if xlim is None:
            xlim = (0.0, float(np.max(spacings)))
    
        bins = np.arange(xlim[0], xlim[1] + width, width)
        if bins.size < 2:
            bins = np.array([xlim[0], xlim[1]])
    
        hist_values, bin_edges = np.histogram(spacings, bins=bins, density=True)
    
        # --- 3) Figure setup ----------------------------------------------------
        if cumulative_spacing:
            fig, (ax, ax_cdf) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    
        # Histogram step plot
        ax.step(bin_edges[1:], hist_values, where='pre', color='black', linewidth=2)
        if hist_values.size > 0:
            ax.plot([xlim[0], min(xlim[0] + width, xlim[1])],
                    [hist_values[0], hist_values[0]], c='k', lw=2)
    
        # --- 4) Overlay Wigner-surmise distributions ---------------------------
        s_vals = np.linspace(xlim[0], xlim[1], 1000)
        if color_list is None:
            color_list = ['r-', 'b-.', 'g--', 'orange-', 'purple--']
    
        wigner_labels = {
            'poissonian': 'Poisson',
            'goe': 'GOE',
            'gue': 'GUE',
            'gse': 'GSE'
        }
    
        for i, name_func in enumerate(plot_wigner or []):
            style = color_list[i % len(color_list)]
            ax.plot(s_vals, NNSD_func(s_vals, wigner=name_func), style, lw=2,
                    label=wigner_labels.get(name_func, name_func.upper()))
    
        # --- 5) Style main plot -------------------------------------------------
        tag = "__exp" if use_experimental else ""
        ax.set_ylabel(r'$P(s)$', fontsize=22)
        ax.set_xlabel('Spacing $s$', fontsize=22)
        ax.set_xlim(xlim)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
        ax.legend(fontsize=20, loc='upper right')
    
        # --- 6) Cumulative subplot (optional) ----------------------------------
        if cumulative_spacing:
            sorted_s = np.sort(spacings)
            cdf_vals = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    
            ax_cdf.scatter(sorted_s, cdf_vals, marker='s', edgecolor='black',
                           facecolor='none', s=30, linewidths=2, zorder=2)
    
            s_full = np.linspace(xlim[0], xlim[1], 1000)
            ds = s_full[1] - s_full[0]
            for i, name_func in enumerate(plot_wigner or []):
                style = color_list[i % len(color_list)]
                pdf = NNSD_func(s_full, wigner=name_func)
                cdf_wigner = np.cumsum(pdf) * ds
                ax_cdf.plot(s_full, cdf_wigner, style, lw=2,
                            label=wigner_labels.get(name_func, name_func.upper()))
    
            ax_cdf.set_ylabel(r'Cumulative $P(s)$', fontsize=22)
            ax_cdf.set_xlabel('Spacing $s$', fontsize=22)
            ax_cdf.minorticks_on()
            ax_cdf.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                               labelsize=22, top=True, right=True)
            ax_cdf.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                               top=True, right=True)
    
        fig.tight_layout()
    
        # --- 7) Save figure if requested ---------------------------------------
        if save_fig:
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            cumulative_part = "__cumulative" if cumulative_spacing else ""
            filename = f"nnsd__{method}{tag}__{wigner_part}__bin{width:.2f}{cumulative_part}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
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
                                mode=None, use_experimental=False,
                                save_fig=False, file_format='svg'):
        """
        Plot long-range spectral correlations: Σ²(L) and/or Δ₃(L).
    
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
            Spectrum selection policy. None -> use self.stats_mode if present, else "auto".
        use_experimental : bool
            If True, use self.exp_spectrum (no degeneracies; expanded == raw).
        save_fig : bool
            If True, save the figure to disk.
        file_format : str
            Output format for saved figure (e.g., 'svg', 'png').
        """

        # --- 1) Select spectrum + Weyl scaling ---------------------------------
        chosen_mode = mode or getattr(self, "stats_mode", "auto")
        spectrum_used, weyl_scale, note, method = self._stats_selection(
            mode=chosen_mode, use_experimental=use_experimental
        )
        spectrum_used = np.sort(np.real(np.asarray(spectrum_used, dtype=float)))
        if spectrum_used.size == 0:
            raise ValueError("Empty spectrum after selection.")
        print(f"[LongRange:{method}] {note}")
    
        # --- 2) Unfold spectrum -------------------------------------------------
        unfolded = self._compute_unfolded_spectrum(
            spectrum=spectrum_used, mode=chosen_mode, use_experimental=use_experimental
        )
    
        # --- 3) Compute Σ² and/or Δ₃ --------------------------------------------
        has_sigma = correlation_type in ["number_variance", "both"]
        has_delta = correlation_type in ["spectral_rigidity", "both"]
    
        if has_sigma:
            L_sigma, sigma2_vals = self._compute_sigma2(unfolded, L_step, L_max)
        if has_delta:
            L_delta, delta3_vals = self._compute_delta3(unfolded, L_step, L_max)
    
        # --- 4) Figure setup ----------------------------------------------------
        if color_list is None:
            color_list = ['r-', 'b-.', 'g--', 'orange-', 'purple--']
        wigner_labels = {'poissonian': 'Poisson', 'goe': 'GOE', 'gue': 'GUE', 'gse': 'GSE'}
    
        if correlation_type == "both":
            fig, (ax_sigma, ax_delta) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={'height_ratios': [1, 1], 'hspace': 0}
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    
        # --- 5) Σ²(L) plot ------------------------------------------------------
        if has_sigma:
            ax_s = ax_sigma if correlation_type == "both" else ax
            ax_s.scatter(L_sigma, sigma2_vals, edgecolor='black', facecolor='white',
                         s=60, linewidth=1.5, zorder=1)
            ax_s.set_ylabel(r'$\Sigma^2(L)$', fontsize=22)
            ax_s.set_xlim(0.0, L_max)
    
            # Overlay RMT predictions
            L_plot = np.linspace(0.0, L_max, 1000)
            ymax_sigma = np.max(sigma2_vals) if len(sigma2_vals) else 1.0
            for i, name in enumerate(plot_wigner or []):
                if name.lower() == "poissonian":
                    continue
                vals = sigma2_RMT(L_plot, wigner=name)
                ymax_sigma = max(ymax_sigma, float(np.max(vals)))
            for i, name in enumerate(plot_wigner or []):
                ax_s.plot(L_plot, sigma2_RMT(L_plot, wigner=name),
                          color_list[i % len(color_list)], lw=2,
                          label=wigner_labels.get(name, name.upper()))
            ax_s.set_ylim(0.0, ymax_sigma * 1.05)
    
            if correlation_type == "number_variance":
                ax_s.set_xlabel("Interval length $L$", fontsize=22)
    
            ax_s.minorticks_on()
            ax_s.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                             labelsize=22, top=True, right=True)
            ax_s.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                             top=True, right=True)
            if correlation_type != "both":
                ax_s.legend(fontsize=18, loc='lower right')
    
        # --- 6) Δ₃(L) plot ------------------------------------------------------
        if has_delta:
            ax_d = ax_delta if correlation_type == "both" else ax
            ax_d.scatter(L_delta, delta3_vals, edgecolor='black', facecolor='white',
                         s=60, linewidth=1.5, zorder=1)
            ax_d.set_ylabel(r'$\Delta_3(L)$', fontsize=22)
            ax_d.set_xlabel("Interval length $L$", fontsize=22)
            ax_d.set_xlim(0.0, L_max)
    
            # Overlay RMT predictions
            L_plot = np.linspace(0.0, L_max, 1000)
            ymax_delta = np.max(delta3_vals) if len(delta3_vals) else 1.0
            for i, name in enumerate(plot_wigner or []):
                if name.lower() == "poissonian":
                    continue
                vals = delta3_RMT(L_plot, wigner=name)
                ymax_delta = max(ymax_delta, float(np.max(vals)))
            for i, name in enumerate(plot_wigner or []):
                ax_d.plot(L_plot, delta3_RMT(L_plot, wigner=name),
                          color_list[i % len(color_list)], lw=2,
                          label=wigner_labels.get(name, name.upper()))
            ax_d.set_ylim(0.0, ymax_delta * 1.05)
    
            ax_d.minorticks_on()
            ax_d.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                             labelsize=22, top=True, right=True)
            ax_d.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                             top=True, right=True)
            ax_d.legend(fontsize=18, loc='lower right')
    
        fig.tight_layout()
    
        # --- 7) Save if requested -----------------------------------------------
        if save_fig:
            tag = "__exp" if use_experimental else ""
            wigner_part = "_".join(plot_wigner) if plot_wigner else "no_wigner"
            base = (f"longrange__{correlation_type}__{wigner_part}_"
                    f"Lmax{L_max}_Lstep{L_step}__{method}{tag}.{file_format}")
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
        Plot the open quantum graph with labeled real nodes, optional edge length annotations,
        and dashed black lines for external leads.
    
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
    
        # Extract node positions
        pos = nx.get_node_attributes(self.G, 'pos')
    
        # Separate real and virtual nodes
        real_nodes = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = list(self.virtual_vertices)
    
        # Draw real vertices: large black nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=900, node_color='black', ax=ax)
    
        # Draw virtual vertices: smaller black nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color='black', ax=ax)
    
        # Draw labels for real nodes (OpenGraphs convention: no +1 offset)
        node_labels = {i: f"{i}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color='white', font_weight='bold',
                                font_size=20, ax=ax)
    
        # Draw internal edges
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color='black')
    
        # Show bond lengths if requested
        if show_lengths:
            edge_labels = {k: f"{v:.2f}"
                           for k, v in nx.get_edge_attributes(self.G, 'length').items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=19, ax=ax)
    
        # ---- External leads: dashed black lines --------------------------------
        lead_length = 80.0
        for v in self.G.nodes:
            if v in self.virtual_vertices:
                continue

            xv, yv = pos[v]
            for p in range(4):
                conn = self.virtual_port_map[v][p]
                if conn is not None:
                    continue  # not a lead

                # Orientation in degrees must be provided
                if not hasattr(self, "lead_orientations") or (v, p) not in self.lead_orientations:
                    continue

                angle_deg = self.lead_orientations[(v, p)]
                theta = np.deg2rad(angle_deg)
                direction = np.array([np.cos(theta), np.sin(theta)])

                start = np.array([xv, yv])
                end   = start + lead_length * direction

                ax.plot([start[0], end[0]], [start[1], end[1]],
                        linestyle="--", linewidth=2, color="black")
    
        # Final plot adjustments
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=1.0)
    
        # Save the figure
        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            suffix = "with_lengths" if show_lengths else "no_lengths"
            filename = f"{self.name}_graph_{suffix}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
    
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"Graph plot saved to {filepath}")
    
        plt.show()

        
    def plot_energy_distribution(self, k, injection_port, show_leads=True,
                                 cmap=None, save=False, file_format="pdf"):
        """
        Plot the energy distribution on the graph bonds for a given injection port and wavenumber.
    
        Parameters
        ----------
        k : float or complex
            Wavenumber at which to evaluate the energy distribution.
        injection_port : int or tuple
            Injection lead. Can be an integer (index into self.leads) or a tuple (vertex, port).
        show_leads : bool, optional
            If True, draw injection, reflection, and transmission arrows (default: True).
        cmap : str, matplotlib Colormap, or None, optional
            Colormap for edge coloring.
            - If None, use a modified 'Reds' where value=1 corresponds to Reds(0.75).
            - If str or Colormap, use it directly.
        save : bool, optional
            If True, saves the figure in self.save_path (default: False).
        file_format : str, optional
            Format for saving the figure (e.g. 'pdf', 'png', 'svg'). Default: 'pdf'.
        """
        from matplotlib.colors import LinearSegmentedColormap
    
        # ---- 1) Identify injection port ----------------------------------------
        if isinstance(injection_port, int):
            v_inj, p_inj = self.leads[injection_port]
            injection_index = injection_port
        else:
            v_inj, p_inj = injection_port
            injection_index = self.leads.index((v_inj, p_inj))
    
        # ---- 2) Compute internal amplitudes ------------------------------------
        e_j = np.zeros((len(self.leads),), dtype=complex)
        e_j[injection_index] = 1.0
    
        D = self._build_propagation_matrix(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        Sigma_ab = self._build_sigma_alpha_betahat(k)
    
        resolvent = np.linalg.inv(np.eye(self.n_directed_bonds) - D @ Sigma_aa)
        a_alpha = resolvent @ D @ Sigma_ab @ e_j
    
        # ---- 3) Compute energy density per undirected bond ---------------------
        energy_per_bond = np.zeros(self.n_internal_bonds)
        for idx, (a, b) in enumerate(self.updated_bonds):
            fwd_idx = self.directed_bond_index[(a, b)]
            rev_idx = self.directed_bond_index[(b, a)]
            energy_per_bond[idx] = np.abs(a_alpha[fwd_idx])**2 + np.abs(a_alpha[rev_idx])**2
    
        # Normalize so the maximum energy is 1 (relative energy density)
        max_energy = np.max(energy_per_bond) if np.max(energy_per_bond) > 0 else 1.0
        norm_energies = energy_per_bond / max_energy
    
        # ---- 4) Prepare plot and nodes -----------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
    
        pos = nx.get_node_attributes(self.G, 'pos')
        real_nodes = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = self.virtual_vertices
    
        # Real nodes: larger black nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=900, node_color='black', ax=ax)
        # Virtual nodes: smaller black nodes (no label)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color='black', ax=ax)
    
        # White labels for real vertices (indexed from 1)
        node_labels = {i: str(i) for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos,
                                labels=node_labels,
                                font_color='white',
                                font_weight='bold',
                                font_size=20,
                                ax=ax)
    
        # ---- 5) Draw edges with energy density ---------------------------------
        edge_widths = [6 + 8 * val for val in norm_energies]
        edge_colors = list(norm_energies)
        edgelist = [(a[0], b[0]) for (a, b) in self.updated_bonds]
    
        # Resolve colormap (default = truncated Reds to 0..0.75)
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
    
        # Draw black borders first (slightly wider than colored edges)
        nx.draw_networkx_edges(self.G, pos, ax=ax,
                               edgelist=edgelist,
                               width=[w + 2 for w in edge_widths],
                               edge_color='black')
    
        # Draw energy-colored edges on top using the resolved colormap
        edges = nx.draw_networkx_edges(self.G, pos, ax=ax,
                                       edgelist=edgelist,
                                       width=edge_widths,
                                       edge_color=edge_colors,
                                       edge_cmap=cmap_obj,
                                       edge_vmin=0.0,
                                       edge_vmax=1.0)
    
        # Adding the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(edges, cax=cax)
        cbar.set_label('Energy Density', fontsize=22)
        cbar.ax.tick_params(labelsize=22)
    
        # ---- 6) Compute S-matrix for lead arrows -------------------------------
        if show_leads:
            S = self.compute_S_matrix(k)
            power_matrix = np.abs(S)**2
    
            j_idx = injection_index
            arrow_length = 80
            arrow_offset = 5
            node_margin = 15
    
            for lead_idx, (v, p) in enumerate(self.leads):
                angle = self.lead_orientations_rad[(v, p)]
                direction = np.array([np.cos(angle), np.sin(angle)])
                start_point = np.array(self.vertex_positions[v])
    
                # Arrow endpoints (margin applied)
                outward_start = start_point + node_margin * direction
                outward_end = start_point + (arrow_length + node_margin) * direction
    
                if lead_idx == j_idx:
                    normal = np.array([-direction[1], direction[0]])
    
                    # Incoming green arrow
                    incoming_start = outward_end - arrow_offset * normal
                    incoming_end = outward_start - arrow_offset * normal
                    ax.add_patch(FancyArrowPatch(incoming_start, incoming_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='green',
                                                 fill=True, clip_on=False))
    
                    # Black reflection arrow
                    reflection_start = outward_start + arrow_offset * normal
                    reflection_end = outward_end + arrow_offset * normal
                    ax.add_patch(FancyArrowPatch(reflection_start, reflection_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='black',
                                                 fill=True, clip_on=False))
    
                    # Label reflection
                    R_val = power_matrix[j_idx, j_idx]
                    # Move label further along arrow direction AND slightly sideways
                    label_pos = reflection_end + 10.0 * direction + 3.0 * normal
                    ax.text(label_pos[0], label_pos[1],
                            rf"$R = {R_val:.2f}$",
                            fontsize=22, ha='left', va='center',
                            clip_on=False)
                else:
                    # Transmission arrows
                    ax.add_patch(FancyArrowPatch(outward_start, outward_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='black',
                                                 fill=True, clip_on=False))
    
                    # Label transmission
                    T_val = power_matrix[lead_idx, j_idx]
                    label_pos = outward_end + 0.5 * direction
                    ax.text(label_pos[0], label_pos[1],
                            rf"$T_{{{lead_idx}{j_idx}}} = {T_val:.2f}$",
                            fontsize=22, ha='left', va='center', clip_on=False)
    
        # ---- 7) Title & final styling ------------------------------------------
        lambda0 = 2 * np.pi / np.real(k)
    
        # Compute imaginary contribution if k is complex
        if np.iscomplex(k):
            imag_term = np.imag(k) * self.total_length
            title_str = (rf"$\lambda = {lambda0:.3f} \ \mu m \ \ "
                         rf"(\mathrm{{Im}}[k L_{{tot}}] = {imag_term:.3f})$")
        else:
            imag_term = None
            title_str = rf"$\lambda_0 = {lambda0:.3f} \ \mu m$"
    
        ax.set_title(title_str, fontsize=22)
    
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=3.0)
    
        # ---- 8) Save figure if requested ---------------------------------------
        if save:
            inj_label = f"lead{injection_index}"
            leads_tag = "withLeads" if show_leads else "noLeads"
    
            if imag_term is not None:
                filename = (f"EnergyDistribution_{inj_label}_{leads_tag}_"
                            f"lambda{lambda0:.5f}_ImkLtot{imag_term:.3f}.{file_format}")
            else:
                filename = (f"EnergyDistribution_{inj_label}_{leads_tag}_"
                            f"lambda{lambda0:.5f}.{file_format}")
    
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Energy distribution figure saved to {filepath}")
    
        # ---- 9) Show figure ----------------------------------------------------
        plt.show()

        
    def plot_S_components(self, lambda_min, lambda_max, n_points,
                          components, spectrum=None, k_imag=0.0,
                          plot_fft=False, save=False, file_format="pdf"):
        """
        Plot the square modulus of selected S-matrix components as a function of wavelength,
        and optionally plot their FFT to extract optical lengths.
    
        Parameters
        ----------
        lambda_min : float
            Minimum wavelength for the plot (in µm).
        lambda_max : float
            Maximum wavelength for the plot (in µm).
        n_points : int
            Number of points to evaluate between lambda_min and lambda_max.
        components : tuple or list of tuples
            Components to plot, e.g. (0,1) or [(0,1), (1,0)].
        spectrum : array-like or None, optional
            If given, vertical red dashed lines are plotted at these real k positions.
        k_imag : float, optional
            Fixed imaginary part to add to k for all evaluations (default: 0.0).
        plot_fft : bool, optional
            If True, compute FFT of each |S_ij|² curve and plot it in optical length domain.
        save : bool, optional
            If True, saves the figures to self.save_path with given file_format.
        file_format : str, optional
            File format for saving the figures (default: "pdf").
        """   
        # --- Ensure components is a list ---
        if isinstance(components, tuple):
            components = [components]
    
        # --- Build wavelength array and convert to k (include fixed imaginary part) ---
        k_min, k_max = 2 * np.pi / lambda_max, 2 * np.pi / lambda_min
        real_k_values = np.linspace(k_min, k_max, n_points)
        lambda_values = 2 * np.pi / real_k_values
        k_values = real_k_values + 1j * k_imag
        delta_k = k_values[1].real - k_values[0].real  # should be uniform
    
        # --- Compute |S_ij|² curves for each component ---
        all_mod_sq = {}
        for (i, j) in components:
            mod_sq = []
            for k in k_values:
                S = self.compute_S_matrix(k)
                mod_sq.append(np.abs(S[i, j])**2)
            all_mod_sq[(i, j)] = np.array(mod_sq)
    
        # ======================================================================
        # 1️) Main |S_ij|² plot
        # ======================================================================
        fig, ax = plt.subplots(figsize=(10, 8))
    
        colors = {}  # store colors for later FFT plot consistency
    
        for idx, (i, j) in enumerate(components):
            if len(components) == 1:
                line, = ax.plot(lambda_values, all_mod_sq[(i, j)], color="black", linewidth=1.2)
                colors[(i, j)] = "black"
            else:
                line, = ax.plot(lambda_values, all_mod_sq[(i, j)], linewidth=2, label=f"|S[{i},{j}]|²")
                colors[(i, j)] = line.get_color()  # grab auto-assigned color
    
        # Optional vertical lines if spectrum is provided
        if spectrum is not None:
            lambda_spec = 2 * np.pi / np.real(spectrum)  # convert spectrum k → λ
            for lam in lambda_spec:
                ax.axvline(lam, color="red", linestyle="--", linewidth=1)
    
        # Axis labels
        ax.set_xlabel("Wavelength (µm)", fontsize=24)
        if len(components) == 1:
            i, j = components[0]
            ax.set_ylabel(rf"$|S_{{{i}{j}}}|^2$", fontsize=24)
        else:
            ax.set_ylabel(r"$|S_{ij}|^2$", fontsize=24)
    
        # Style
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
        ax.set_xlim(lambda_min, lambda_max)
        if len(components) > 1:
            ax.legend(fontsize=18, frameon=False)
    
        plt.tight_layout()
    
        # --- Save S plot ---
        comp_tag = "_".join([f"S{i}{j}" for i, j in components])
        spec_tag = "withSpec" if spectrum is not None else "noSpec"
        filename_S = (f"Scomponents_{comp_tag}_{spec_tag}_"
                      f"lam{lambda_min:.3f}-{lambda_max:.3f}_"
                      f"N{n_points}_"
                      f"kIm{k_imag:.2e}.{file_format}")
        if save:
            fig_path = os.path.join(self.save_path, filename_S)
            fig.savefig(fig_path, format=file_format, dpi=300, bbox_inches="tight")
            print(f"✓ |S_ij|² figure saved to {fig_path}")
    
        plt.show()
    
        # ======================================================================
        # 2️) FFT PLOT (optional)
        # ======================================================================
        if plot_fft:
            # Prepare FFT padding
            n_fft = next_fast_len(n_points)
    
            # Prepare subplot structure
            fig_fft, axs = plt.subplots(len(components), 1, figsize=(10, 8),
                                        sharex=True)
            if len(components) == 1:
                axs = [axs]  # ensure iterable
    
            fig_fft.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.12, hspace=0)
    
            max_val = 0
            for idx, (i, j) in enumerate(components):
                signal = all_mod_sq[(i, j)]
    
                # Apply Hann window
                window = windows.hann(len(signal))
                signal_windowed = signal * window
    
                # FFT → optical length (positive side only)
                fft_vals = np.abs(fft(signal_windowed, n=n_fft))[:n_fft // 2]
                optical_length = 2 * np.pi * fftfreq(n_fft, d=delta_k)[:n_fft // 2]
                
                print('optical_length =', optical_length)
                print('fft_vals =', fft_vals)
    
                # Plot FFT using stored color (black for single component, matching colors otherwise)
                axs[idx].plot(optical_length, fft_vals, color=colors[(i, j)], lw=1)
                axs[idx].set_xlim(0, 4999)
                axs[idx].set_yticks([])
    
                # Track max for consistent scaling
                non_dc_peak = np.max(fft_vals[50:])  # ignore DC peak
                max_val = max(max_val, non_dc_peak)
    
                # Label each subplot
                if len(components) > 1:
                    axs[idx].text(0.98, 0.90, rf"$S_{{{i}{j}}}$", ha="right", va="top",
                                  transform=axs[idx].transAxes, fontsize=20)
    
            # Common style
            for ax in axs:
                ax.set_ylim(0, 1.1 * max_val)
                ax.tick_params(axis='x', which='major', direction='in', length=6, width=2,
                               labelsize=20, top=True)
                ax.tick_params(axis='x', which='minor', direction='in', length=3, width=1.5,
                               top=True)
    
            axs[-1].set_xlabel("Optical length (µm)", fontsize=22)
            if len(components) == 1:
                axs[-1].set_ylabel("Fourier Transform", fontsize=22)
            else:
                fig_fft.text(0.08, 0.5, "Fourier Transform", fontsize=22,
                             va='center', rotation='vertical')
        
            # --- Save FFT plot ---
            if save:
                filename_fft = filename_S.replace(f".{file_format}", f"_FFT.{file_format}")
                fig_path_fft = os.path.join(self.save_path, filename_fft)
                fig_fft.savefig(fig_path_fft, format=file_format, dpi=300, bbox_inches='tight')
                print(f"✓ FFT figure saved to {fig_path_fft}")
    
            plt.show()
        
    def plot_resonances_sigma(self,
                              lam_min: float = None,
                              lam_max: float = None,
                              N_re: int = 500_000,
                              N_im: int = 100,
                              im_min: float = -1e-4,
                              im_max: float = 0.0,
                              plot_heatmap: bool = True,
                              overwrite_grid: bool = False,
                              grid_dtype: str = "float32",
                              save: bool = True,
                              file_format: str = "pdf",
                              verbose: bool = True):
        """
        Plot the complex resonances in the k-plane, optionally overlaying the σ_min(I-U(k))
        heatmap loaded from (or computed into) the cached grid.
    
        Requirements
        ------------
        - This function expects 'self.spectrum' to already be available (run
          find_resonances_sigma2d first). If not present, raises a RuntimeError.
    
        Parameters
        ----------
        lam_min, lam_max : float or None
            Wavelength bounds (µm) used to define Re(k) range for plotting and grid lookup.
            If None, the corresponding bound from self.wavelength_range is used.
        N_re : int
            Number of samples along Re(k) for the cached grid (used for filename/lookup).
        N_im : int
            Number of samples along Im(k) for the cached grid (used for filename/lookup).
        im_min : float
            Minimum Im(k) (negative) for plotting and grid lookup.
        im_max : float
            Maximum Im(k) (usually 0) for plotting and grid lookup.
        plot_heatmap : bool
            If True, load (or compute) the σ_min grid and show as a heatmap; crosses in red.
            If False, do NOT load/compute the grid; plot resonances as black crosses.
        overwrite_grid : bool
            If True and plot_heatmap=True, recompute the σ_min grid even if a matching file exists.
        grid_dtype : {"float32","float64"}
            Storage dtype for the σ_min grid (computation was done in float64).
        save : bool
            Save the figure.
        file_format : str
            Figure format, e.g., "pdf" (default).
        verbose : bool
            Print progress/status information.
    
        Returns
        -------
        None
        """ 
    
        # ---- 0) Validate presence of resonances ---------------------------------
        if not hasattr(self, "spectrum"):
            raise RuntimeError(
                "'self.spectrum' is required but missing. "
                "Run 'find_resonances_sigma2d' first."
            )
    
        spectrum = np.asarray(self.spectrum, dtype=np.complex128)
    
        # ---- 1) Handle wavelength range -----------------------------------------
        if lam_min is None or lam_max is None:
            lam_min0, lam_max0 = self.wavelength_range
            lam_min = lam_min if lam_min is not None else lam_min0
            lam_max = lam_max if lam_max is not None else lam_max0
    
        # k-axis bounds from λ-bounds
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
    
        # ---- 2) Axes arrays for extent/filenames --------------------------------
        re_vals_plot = np.linspace(k_min, k_max, N_re, dtype=np.float64)
        im_vals_plot = np.linspace(im_max, im_min, N_im, dtype=np.float64)  # descending
    
        # Grid filename (must match the σ_min caching scheme)
        grid_fname = (f"sigma_grid_lamMin_{lam_min:.6f}_lamMax_{lam_max:.6f}"
                      f"_Nre_{N_re}_Nim_{N_im}"
                      f"_imMin_{im_min:.6e}_imMax_{im_max:.6e}.npy")
        grid_path = os.path.join(self.save_path, grid_fname)
    
        # ---- 3) Prepare figure ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # ---- 4) Heatmap (if requested) ------------------------------------------
        grid_mem = None
        if plot_heatmap:
            if overwrite_grid or (not os.path.exists(grid_path)):
                if verbose:
                    if overwrite_grid:
                        print(f"Overwriting σ_min grid: {grid_path}")
                    else:
                        print(f"σ_min grid not found. Computing: {grid_path}")
                # Create/compute the grid using existing helpers
                mem, re_vals2, im_vals2, grid_path2 = self._open_or_create_sigma_grid(
                    N_re=N_re, N_im=N_im, im_min=im_min, im_max=im_max,
                    grid_dtype=grid_dtype, overwrite=overwrite_grid, verbose=verbose
                )
                self._fill_sigma_grid(mem, re_vals2, im_vals2, grid_path2, verbose=verbose)
                grid_mem = mem
                # Ensure axes for extent match the file we just created
                re_vals_plot, im_vals_plot = re_vals2, im_vals2
            else:
                if verbose:
                    print(f"Loading σ_min grid from: {grid_path}")
                grid_mem = np.lib.format.open_memmap(grid_path, mode="r")
    
            # Plot using requested normalization/color map
            finite = grid_mem[np.isfinite(grid_mem)]
            # Guard against non-positive vmin for LogNorm
            vmin = float(np.nanmin(finite)) if finite.size else 1e-16
            if vmin <= 0.0:
                vmin = 1e-16
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
    
            imshow_obj = ax.imshow(
                grid_mem,
                extent=[re_vals_plot[0], re_vals_plot[-1], im_vals_plot[-1], im_vals_plot[0]],
                aspect='auto',
                cmap="inferno_r",
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            cbar = plt.colorbar(imshow_obj, ax=ax)
            cbar.set_label(r"$\sigma_{\min}(I-U(k))$", fontsize=22)
            cbar.ax.tick_params(labelsize=22)
    
        # ---- 5) Plot resonances (cross color depends on heatmap) ----------------
        if spectrum.size > 0:
            cross_color = 'red' if plot_heatmap else 'black'
            ax.scatter(np.real(spectrum), np.imag(spectrum),
                       marker='x', color=cross_color, s=40)
    
        # ---- 6) Axes config ------------------------------------------------------
        ax.set_xlim(k_min, k_max)
        ax.set_ylim(im_min, im_max)
    
        ax.set_xlabel("Re(k) [µm$^{-1}$]", fontsize=22)
        ax.set_ylabel("Im(k) [µm$^{-1}$]", fontsize=22)
    
        # Force scientific notation on y-axis (imag is small ~ -1e-4)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_powerlimits((-3, 3))  # force sci notation for small ranges
        ax.yaxis.set_major_formatter(yfmt)
    
        # Ticks font sizes
        ax.tick_params(axis='both', which='major', labelsize=22)
    
        plt.tight_layout()
    
        # ---- 7) Save figure if requested ----------------------------------------
        if save:
            secular_tag = "heatmap" if plot_heatmap else "crossesOnly"
            lam_tag = f"lam{lam_min:.3f}-{lam_max:.3f}"
            fname = f"ResonancesSigma_{secular_tag}_{lam_tag}.{file_format}"
            fig_path = os.path.join(self.save_path, fname)
            fig.savefig(fig_path, format=file_format, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"✓ Figure saved to {fig_path}")
    
        plt.show()


    def plot_length_spectrum(self, L_max=4999, n_points=100000,
                             use_experimental=False,
                             save=True, file_format="pdf",
                             ignore_zero_peak=True,
                             zero_window_um=None):
        """
        Compute and plot the length spectrum from the graph's (quasi-)resonances.
    
        Parameters
        ----------
        L_max : float, optional
            Maximum length to plot (µm). Default is 4999.
        n_points : int, optional
            Number of points used for the ℓ grid. Default is 100000.
        use_experimental : bool
            If True, use self.exp_spectrum (real experimental peaks).
            Otherwise, use self.spectrum (numerical resonances).
        save : bool, optional
            If True, saves the figure in self.save_path.
        file_format : str, optional
            File format for saving the figure ('pdf', 'png', 'svg', etc.).
        ignore_zero_peak : bool, optional
            If True (default), exclude a tiny window near ℓ=0 when computing
            the plot's y-limit so the huge DC peak does not set the ylim.
        zero_window_um : float or None, optional
            Half-open window width (in µm) to exclude at the origin for ylim
            selection only. If None, it defaults to max(100 µm, 50 * Δℓ).
        """
    
        # 1) ---- Select spectrum -------------------------------------------------
        if use_experimental:
            if not hasattr(self, "exp_spectrum"):
                raise AttributeError("use_experimental=True but exp_spectrum is missing. "
                                     "Run find_experimental_resonances() first.")
            k_vals = np.asarray(self.exp_spectrum, dtype=np.complex128)
            if k_vals.size == 0:
                raise ValueError("exp_spectrum is empty.")
            tag = "__exp"
            note = "[LengthSpectrum] Using experimental spectrum."
        else:
            if not hasattr(self, "spectrum"):
                raise AttributeError("No spectrum found. Run find_resonances() first.")
            k_vals = np.asarray(self.spectrum, dtype=np.complex128)
            if k_vals.size == 0:
                raise ValueError("spectrum is empty.")
            tag = ""
            note = "[LengthSpectrum] Using numerical spectrum."
        print(note)
    
        # 2) ---- Build ℓ grid ----------------------------------------------------
        ell = np.linspace(0.0, float(L_max), int(n_points))  # µm
        d_ell = ell[1] - ell[0] if ell.size >= 2 else float(L_max)
    
        # 3) ---- Compute length spectrum  ρ̂(ℓ) ----------------------------------
        # ρ̂(ℓ) = | Σ_j exp[-i * k_j * n_eff(Re(k_j)) * ℓ] |
        rho_hat = np.zeros_like(ell, dtype=np.complex128)
    
        if callable(self.neff):
            n_eff_vec = self.neff(np.real(k_vals))
            if np.isscalar(n_eff_vec):
                n_eff_vec = np.full(k_vals.shape, float(n_eff_vec), dtype=float)
        else:
            n_eff_vec = np.full(k_vals.shape, float(self.neff), dtype=float)
    
        # Memory-friendly accumulation
        for kj, neff_j in zip(k_vals, n_eff_vec):
            rho_hat += np.exp(-1j * kj * neff_j * ell)
    
        rho_hat = np.abs(rho_hat)
    
        # 4) ---- Choose ylim without the ℓ≈0 spike -------------------------------
        if ignore_zero_peak:
            if zero_window_um is None:
                zero_window_um = max(200, 50 * d_ell)  # small window near 0
            mask_ylim = ell > zero_window_um
            if np.any(mask_ylim):
                ymax = float(np.max(rho_hat[mask_ylim]))
            else:
                # Fallback: if grid is too small, just ignore the first point
                ymax = float(np.max(rho_hat[1:])) if rho_hat.size > 1 else float(rho_hat.max())
        else:
            ymax = float(rho_hat.max())
    
        # Guard in case everything is ~0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
    
        # 5) ---- Plot -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(ell, rho_hat, color="black", lw=1.5)
    
        ax.set_xlim(0.0, float(L_max))
        ax.set_ylim(0.0, 1.05 * ymax)
        ax.set_xlabel(r"Length $\ell$ ($\mu$m)", fontsize=22)
        ax.set_ylabel(r"$\hat{\rho}(\ell)$", fontsize=22)
    
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2,
                       labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5,
                       top=True, right=True)
    
        plt.tight_layout()
    
        # 6) ---- Save -------------------------------------------------------------
        if save:
            filename = f"length_spectrum__Lmax{L_max}_N{n_points}{tag}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Length spectrum figure saved to: {filepath}")
    
        plt.show()
        return fig, ax



     
# ========================================================================
#                 Subclass of OpenGraphs: DirectionalGraphs
# ========================================================================
        
class DirectionalGraphs(OpenGraphs):
    """
    Construct an open quantum graph where all vertices are **4-port directional couplers**.

    This subclass of `OpenGraphs` automatically generates the vertex scattering
    matrices from coupling constants. Each vertex is assumed to be a lossless
    directional coupler modeled by a 4×4 unitary scattering matrix. Couplings
    can be provided per-vertex, globally for internal (no lead) vertices, and/or
    globally for external (lead-connected) vertices.

    Parameters
    ----------
    vertex_positions : list of (float, float)
        Cartesian coordinates of the vertices in µm.

    port_map : list of lists
        Connectivity specification. For vertex i and port p:
            port_map[i][p] = (j, q) or None (for an external lead).
        `None` entries indicate open leads.

    lengths : dict or None, optional
        Explicit dictionary mapping bonds to physical lengths:
            {((i, p), (j, q)) : length_in_um}
        If None, lengths are inferred from Euclidean distances between
        vertex positions (internal bonds only).

    vertex_couplings : dict, list, float, or callable, optional
        Coupling constants for each vertex.
        - dict/list: one entry per vertex (float or callable C(k)).
        - float: same constant for all vertices.
        - callable: returns C(k) as a function of wavenumber.

    internal_coupling : float or callable or None, optional
        Default coupling for **internal** vertices (those not connected to a lead),
        used when a given vertex is not specified in `vertex_couplings`.

    external_coupling : float or callable or None, optional
        Default coupling for **external** (lead-connected) vertices, used when a
        given vertex is not specified in `vertex_couplings`.

    neff : float or callable, optional
        Effective refractive index. May be constant or k-dependent.

    ng : float or callable, optional
        Group index. If None, it is derived numerically from neff(k).

    wavelength_range : tuple of (float, float), optional
        Vacuum wavelength range (µm) defining the working spectral window.

    virtual_vertex_spacing : float, sequence, or None, optional
        Spacing used when splitting multiple bonds. If a sequence is given,
        values are cycled across multi-bond groups.

    lead_orientations : dict or None, optional
        Optional dictionary {(vertex, port): angle_deg} for plotting/export.

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
                 internal_coupling=None,
                 external_coupling=None,
                 neff=1.0,
                 ng=None,
                 wavelength_range=(1.48, 1.64),
                 virtual_vertex_spacing=None,
                 lead_orientations=None,
                 name="MyOpenDirectionalGraph",
                 save_path=None):
        """
        Initialize an open directional graph (all vertices = 4-port couplers).  
        Builds vertex scattering matrices automatically from couplings.  
        """
        # Store coupling specifications
        self.vertex_couplings = vertex_couplings
        self.internal_coupling = internal_coupling
        self.external_coupling = external_coupling

        # Build per-vertex scattering matrices from couplings
        vertex_matrices = self._build_vertex_matrices(vertex_positions, port_map)

        # Describe couplings for deterministic save-dir naming
        self.coupling_info = self._get_coupling_description()

        # Delegate to OpenGraphs with the unified save_path logic
        super().__init__(
            vertex_positions=vertex_positions,
            port_map=port_map,
            lengths=lengths,
            vertex_matrices=vertex_matrices,
            internal_matrix=None,     # matrices already built per-vertex
            external_matrix=None,     # matrices already built per-vertex
            neff=neff,
            ng=ng,
            wavelength_range=wavelength_range,
            virtual_vertex_spacing=virtual_vertex_spacing,
            lead_orientations=lead_orientations,
            name=name,
            save_path=save_path
        )


    def _build_vertex_matrices(self, vertex_positions, port_map):
        """
        Builds the vertex_scattering_matrix dictionary from coupling constants.
        """
        n = len(vertex_positions)
        valencies = [len(ports) for ports in port_map]
        matrices = {}

        for v in range(n):
            val = valencies[v]
            if val != 4:
                raise ValueError(f"Only 4-port directional couplers are supported, but vertex {v} has valency {val}.")

            if self.vertex_couplings is not None:
                if v not in self.vertex_couplings:
                    raise ValueError(f"Missing coupling constant for vertex {v} in vertex_couplings.")
                C = self.vertex_couplings[v]
            else:
                # Infer whether vertex is external (connected to a lead)
                is_external = any(conn is None for conn in port_map[v])
                C = self.external_coupling if is_external else self.internal_coupling
                if C is None:
                    raise ValueError(f"Missing coupling constant for vertex {v} (external={is_external}).")

            # Build matrix from coupling
            matrices[v] = self._directional_coupler_matrix(C)

        return matrices

    def _directional_coupler_matrix(self, C):
        """
        Returns a 4x4 directional coupler scattering matrix from coupling constant C.
        Works with float or callable C.
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
        Build a compact string description for internal and external couplings (for save_dir_name).
        Skips Cin or Cext if they are None.
        """
        def coupling_str(C):
            if C is None:
                return None
            if callable(C):
                return getattr(C, '__name__', 'Cfunc')
            return f"{C:.2f}"
    
        parts = []
        cin_str = coupling_str(self.internal_coupling)
        cext_str = coupling_str(self.external_coupling)
    
        if cin_str is not None:
            parts.append(f"Cin={cin_str}")
        if cext_str is not None:
            parts.append(f"Cext={cext_str}")
    
        if self.vertex_couplings is not None:
            return "custom_vertex_couplings"
    
        # Join only the parts that exist
        return "_".join(parts) if parts else "no_coupling"

    
    def plot_graph(self,
                   show_lengths=False,
                   show_port_arrows=True,
                   head_length_frac=0.12,   # head length as fraction of Lmin
                   head_width_frac=0.08,    # head width  as fraction of Lmin
                   head_offset_frac=0.24,   # CENTER distance from node = head_offset_frac * Lmin
                   save_fig=False,
                   file_format="pdf"):
        """
        Plot the open (directional) quantum graph with labeled real nodes, optional edge lengths,
        and head-only arrows à la Dietz on 4-port directional vertices:
          - GREEN for incoming ports (0,1)
          - RED   for outgoing ports (2,3)
    
        Arrowheads are custom triangular patches with:
          - SAME absolute center distance from the node for all ports (equidistant),
          - Size (length, width) controlled by fractions of Lmin.
    
        Leads (conn=None) are dashed black segments of fixed length and also receive heads.
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
                               node_size=700, node_color='black', ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=300, node_color='black', ax=ax)
        node_labels = {i: f"{i}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color='white', font_weight='bold',
                                font_size=18, ax=ax)
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color=EDGE_COLOR)
    
        if show_lengths:
            edge_labels = {k: f"{v:.2f}"
                           for k, v in nx.get_edge_attributes(self.G, 'length').items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=19, ax=ax)
    
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
            We keep a small margin so tip/base don't overshoot very short segments.
            """
            vx, vy = vertex_xy
            nx_, ny_ = neighbor_xy
            dx, dy = nx_ - vx, ny_ - vy
            Lseg = float(np.hypot(dx, dy))
            if Lseg == 0.0:
                return None, None, 0.0
    
            u = np.array([dx, dy]) / Lseg
    
            # Keep head fully on the segment: center must be at least half head_len from both ends
            # Margin factor (>0.5) reduces risk of touching nodes visually
            margin = 0.55
            max_center = max(0.0, (Lseg * 0.5) - margin * (head_len * 0.5))
            eff_offset = min(desired_offset, max_center)
    
            center = np.array([vx, vy]) + eff_offset * u
            return center, u, Lseg
    
        # ------------------------ Port heads (internal) -------------------------
        if show_port_arrows:
            for v in self.G.nodes:
                if v in self.virtual_vertices:
                    continue
    
                xv, yv = pos[v]
                vxy = (xv, yv)
    
                for p in range(4):
                    conn = self.virtual_port_map[v][p]
                    if conn is None:
                        continue  # handled in leads section
    
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
    
        # --------------------- External leads: dashed + heads -------------------
        lead_length = 80.0  # fixed dashed length
    
        for v in self.G.nodes:
            if v in self.virtual_vertices:
                continue
    
            xv, yv = pos[v]
            origin = np.array([xv, yv])
    
            for p in range(4):
                conn = self.virtual_port_map[v][p]
                if conn is not None:
                    continue  # not a lead
    
                if not hasattr(self, "lead_orientations") or (v, p) not in self.lead_orientations:
                    continue
    
                angle_deg = self.lead_orientations[(v, p)]
                theta = np.deg2rad(angle_deg)
                u = np.array([np.cos(theta), np.sin(theta)])
    
                start = origin
                end   = origin + lead_length * u
    
                # Draw lead BELOW arrows
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        linestyle="--", linewidth=2, color=EDGE_COLOR, zorder=2)
    
                if show_port_arrows:
                    # Keep same absolute offset (equidistant) with clamping on the lead segment
                    # Reuse the same safety rule as internal edges
                    Lseg = lead_length
                    margin = 0.55
                    max_center = max(0.0, (Lseg * 0.5) - margin * (head_length * 0.5))
                    eff_offset = min(offset_abs, max_center)
    
                    center = origin + eff_offset * u
    
                    if p in (0, 1):
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




    
    








