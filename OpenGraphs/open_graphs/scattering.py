#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScatteringMixin — Σ-block builders, compute_S_matrix, _U, _sigma_min.
"""

import numpy as np


class ScatteringMixin:

    # -------------------------------------------------------------------------
    # Propagation matrix
    # -------------------------------------------------------------------------

    def _build_propagation_matrix(self, k):
        """Build the diagonal propagation matrix D(k) for all directed bonds."""
        n_eff = self.neff(np.real(k)) if callable(self.neff) else self.neff
        D = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)
        for dbond, idx in self.directed_bond_index.items():
            L = self.directed_lengths[dbond]
            D[idx, idx] = np.exp(1j * k * n_eff * L)
        return D

    # -------------------------------------------------------------------------
    # Σ-blocks
    # -------------------------------------------------------------------------

    def _build_sigma_alpha_alpha(self, k):
        """Build the internal scattering block Σ_{αα}."""
        Sigma_aa = np.zeros((self.n_directed_bonds, self.n_directed_bonds), dtype=complex)
        for dbond_in in self.directed_bonds:
            src, dst  = dbond_in
            v         = dst[0]
            port_in   = dst[1]
            sigma_v   = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
            idx_in = self.directed_bond_index[dbond_in]
            for port_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][port_out]
                if conn is not None:
                    neighbor, n_port = conn
                    dbond_out = ((v, port_out), (neighbor, n_port))
                    idx_out   = self.directed_bond_index[dbond_out]
                    Sigma_aa[idx_out, idx_in] = sigma_v[port_out, port_in]
        return Sigma_aa

    def _build_sigma_alpha_betahat(self, k):
        """Build the scattering block Σ_{α β̂} (lead injection into internal bonds)."""
        Sigma_ab = np.zeros((self.n_directed_bonds, self.n_leads), dtype=complex)
        for lead_idx, (v, p_in) in enumerate(self.leads):
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
            for p_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p_out]
                if conn is not None:
                    neighbor, n_port = conn
                    dbond_out = ((v, p_out), (neighbor, n_port))
                    idx_out   = self.directed_bond_index[dbond_out]
                    Sigma_ab[idx_out, lead_idx] = sigma_v[p_out, p_in]
        return Sigma_ab

    def _build_sigma_beta_alpha(self, k):
        """Build the scattering block Σ_{β α} (internal bonds to outgoing leads)."""
        Sigma_ba = np.zeros((self.n_leads, self.n_directed_bonds), dtype=complex)
        for dbond_in in self.directed_bonds:
            src, dst = dbond_in
            v        = dst[0]
            port_in  = dst[1]
            sigma_v  = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
            idx_in = self.directed_bond_index[dbond_in]
            for p_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p_out]
                if conn is None:
                    try:
                        lead_idx = self.leads.index((v, p_out))
                    except ValueError:
                        raise RuntimeError(f"Lead (v={v}, port={p_out}) not found in self.leads")
                    Sigma_ba[lead_idx, idx_in] = sigma_v[p_out, port_in]
        return Sigma_ba

    def _build_sigma_beta_betahat(self, k):
        """Build the direct lead-to-lead scattering block Σ_{β β̂}."""
        Sigma_bb = np.zeros((self.n_leads, self.n_leads), dtype=complex)
        for lead_in_idx, (v, p_in) in enumerate(self.leads):
            sigma_v = self.vertex_scattering_matrix[v]
            if callable(sigma_v):
                sigma_v = sigma_v(np.real(k))
            for p_out in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p_out]
                if conn is None:
                    lead_out_idx = self.leads.index((v, p_out))
                    Sigma_bb[lead_out_idx, lead_in_idx] = sigma_v[p_out, p_in]
        return Sigma_bb

    # -------------------------------------------------------------------------
    # Public: S-matrix
    # -------------------------------------------------------------------------

    def compute_S_matrix(self, k):
        """
        Compute the scattering matrix S(k) for the open quantum graph.

        Returns
        -------
        S : ndarray (L × L)
        """
        D        = self._build_propagation_matrix(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        Sigma_ab = self._build_sigma_alpha_betahat(k)
        Sigma_ba = self._build_sigma_beta_alpha(k)
        Sigma_bb = self._build_sigma_beta_betahat(k)
        U_tilde  = D @ Sigma_aa
        resolvent = np.linalg.inv(np.eye(self.n_directed_bonds) - U_tilde)
        S = Sigma_bb + Sigma_ba @ resolvent @ D @ Sigma_ab
        return S

    # -------------------------------------------------------------------------
    # Internal helpers for resonance finder
    # -------------------------------------------------------------------------

    def _U(self, k: complex) -> np.ndarray:
        """Return U(k) = D(k) @ Σ_αα(k), shape (2B, 2B)."""
        k        = complex(k)
        Sigma_aa = self._build_sigma_alpha_alpha(k)
        D_k      = self._build_propagation_matrix(k)
        return (D_k.astype(np.complex128, copy=False) @
                Sigma_aa.astype(np.complex128, copy=False))

    def _sigma_min(self, k: complex) -> float:
        """g(k) = σ_min(I − U(k)).  Returns np.inf on failure."""
        try:
            U    = self._U(k)
            A    = np.eye(U.shape[0], dtype=np.complex128) - U
            svals = np.linalg.svd(A, compute_uv=False)
            smin  = float(svals[-1])
            return smin if np.isfinite(smin) else np.inf
        except Exception:
            return np.inf
