#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPlotMixin — plot_graph (base OpenGraphs version) and plot_energy_distribution.
"""

from ...core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


class GraphPlotMixin:

    def plot_graph(self, show_lengths=False, save_fig=False, file_format="pdf"):
        """
        Plot the open quantum graph with labeled real nodes, optional edge length
        annotations, and dashed black lines for external leads.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        pos         = nx.get_node_attributes(self.G, 'pos')
        real_nodes  = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = list(self.virtual_vertices)

        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=900, node_color='black', ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color='black', ax=ax)

        node_labels = {i: f"{i}" for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color='white', font_weight='bold',
                                font_size=20, ax=ax)
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color='black')

        if show_lengths:
            edge_labels = {k: f"{v:.2f}"
                           for k, v in nx.get_edge_attributes(self.G, 'length').items()}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                         font_size=19, ax=ax)

        # External leads
        lead_length = 80.0
        for v in self.G.nodes:
            if v in self.virtual_vertices:
                continue
            xv, yv = pos[v]
            for p in range(self.valencies[v]):
                conn = self.virtual_port_map[v][p]
                if conn is not None:
                    continue
                if not hasattr(self, "lead_orientations") or (v, p) not in self.lead_orientations:
                    continue
                angle_deg = self.lead_orientations[(v, p)]
                theta     = np.deg2rad(angle_deg)
                direction = np.array([np.cos(theta), np.sin(theta)])
                start     = np.array([xv, yv])
                end       = start + lead_length * direction
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        linestyle="--", linewidth=2, color="black")

        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=1.0)

        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            suffix   = "with_lengths" if show_lengths else "no_lengths"
            filename = f"{self.name}_graph_{suffix}.{file_format}"
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            log(f"Graph plot saved to {filepath}")

        show_figure()

    def plot_energy_distribution(self, k, injection_port, show_leads=True,
                                 cmap=None, save=False, save_fig=None, file_format="pdf"):
        """
        Plot the energy distribution on the graph bonds for a given injection port
        and wavenumber.
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        from matplotlib.colors import LinearSegmentedColormap

        # Identify injection port
        if isinstance(injection_port, int):
            v_inj, p_inj    = self.leads[injection_port]
            injection_index = injection_port
        else:
            v_inj, p_inj    = injection_port
            injection_index = self.leads.index((v_inj, p_inj))

        # Compute internal amplitudes
        e_j               = np.zeros((len(self.leads),), dtype=complex)
        e_j[injection_index] = 1.0

        # amplitudes arriving at the end of each directed bond
        a_alpha = self.model.internal_amplitudes(k, e_j)

        # Energy density per undirected bond
        energy_per_bond = np.zeros(self.n_internal_bonds)
        for idx, (a, b) in enumerate(self.updated_bonds):
            fwd_idx = self.directed_bond_index[(a, b)]
            rev_idx = self.directed_bond_index[(b, a)]
            energy_per_bond[idx] = (np.abs(a_alpha[fwd_idx])**2 +
                                    np.abs(a_alpha[rev_idx])**2)

        # entropy per physical bond (independent of the virtual vertices)
        S_bonds, S_norm_, _ = self.localization_measures(a_alpha)

        max_energy   = np.max(energy_per_bond) if np.max(energy_per_bond) > 0 else 1.0
        norm_energies = energy_per_bond / max_energy

        fig, ax = plt.subplots(figsize=(10, 8))
        pos           = nx.get_node_attributes(self.G, 'pos')
        real_nodes    = [i for i in self.G.nodes if i not in self.virtual_vertices]
        virtual_nodes = self.virtual_vertices

        nx.draw_networkx_nodes(self.G, pos, nodelist=real_nodes,
                               node_size=900, node_color='black', ax=ax)
        nx.draw_networkx_nodes(self.G, pos, nodelist=virtual_nodes,
                               node_size=400, node_color='black', ax=ax)

        node_labels = {i: str(i) for i in real_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels,
                                font_color='white', font_weight='bold',
                                font_size=20, ax=ax)

        edge_widths = [6 + 8 * val for val in norm_energies]
        edge_colors = list(norm_energies)
        edgelist    = [(a[0], b[0]) for (a, b) in self.updated_bonds]

        if cmap is None:
            base    = plt.get_cmap("Reds")
            cmap_obj = LinearSegmentedColormap.from_list(
                "Reds_truncated",
                [base(x) for x in np.linspace(0.0, 0.75, 256)]
            )
        elif isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = cmap

        nx.draw_networkx_edges(self.G, pos, ax=ax,
                               edgelist=edgelist,
                               width=[w + 2 for w in edge_widths],
                               edge_color='black')

        edges = nx.draw_networkx_edges(self.G, pos, ax=ax,
                                       edgelist=edgelist,
                                       width=edge_widths,
                                       edge_color=edge_colors,
                                       edge_cmap=cmap_obj,
                                       edge_vmin=0.0,
                                       edge_vmax=1.0)

        # Core-matplotlib inset instead of mpl_toolkits.axes_grid1: a stale
        # system mpl_toolkits can shadow the pip one and break append_axes.
        cax  = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cbar = plt.colorbar(edges, cax=cax)
        cbar.set_label('Energy Density', fontsize=22)
        cbar.ax.tick_params(labelsize=22)

        if show_leads:
            S            = self.compute_S_matrix(k)
            power_matrix = np.abs(S)**2
            j_idx        = injection_index
            arrow_length = 80
            arrow_offset = 5
            node_margin  = 15

            for lead_idx, (v, p) in enumerate(self.leads):
                angle     = self.lead_orientations_rad[(v, p)]
                direction = np.array([np.cos(angle), np.sin(angle)])
                start_pt  = np.array(self.vertex_positions[v])

                outward_start = start_pt + node_margin * direction
                outward_end   = start_pt + (arrow_length + node_margin) * direction

                if lead_idx == j_idx:
                    normal = np.array([-direction[1], direction[0]])
                    incoming_start = outward_end   - arrow_offset * normal
                    incoming_end   = outward_start - arrow_offset * normal
                    ax.add_patch(FancyArrowPatch(incoming_start, incoming_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='green',
                                                 fill=True, clip_on=False))
                    reflection_start = outward_start + arrow_offset * normal
                    reflection_end   = outward_end   + arrow_offset * normal
                    ax.add_patch(FancyArrowPatch(reflection_start, reflection_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='black',
                                                 fill=True, clip_on=False))
                    R_val     = power_matrix[j_idx, j_idx]
                    label_pos = reflection_end + 10.0 * direction + 3.0 * normal
                    ax.text(label_pos[0], label_pos[1], rf"$R = {R_val:.2f}$",
                            fontsize=22, ha='left', va='center', clip_on=False)
                else:
                    ax.add_patch(FancyArrowPatch(outward_start, outward_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='black',
                                                 fill=True, clip_on=False))
                    T_val     = power_matrix[lead_idx, j_idx]
                    label_pos = outward_end + 0.5 * direction
                    ax.text(label_pos[0], label_pos[1],
                            rf"$T_{{{lead_idx}{j_idx}}} = {T_val:.2f}$",
                            fontsize=22, ha='left', va='center', clip_on=False)

        lambda0   = 2 * np.pi / np.real(k)
        if np.iscomplex(k):
            imag_term = np.imag(k) * self.total_length
            title_str = (rf"$\lambda = {lambda0:.3f} \ \mu m, \ "
                         rf"\mathrm{{Im}}[k L_{{tot}}] = {imag_term:.3f}, \ "
                         rf"S_{{\rm norm}} = {S_norm_:.3f}$")
        else:
            imag_term = None
            title_str = (rf"$\lambda_0 = {lambda0:.3f} \ \mu m, \ "
                         rf"S_{{\rm norm}} = {S_norm_:.3f}$")

        ax.set_title(title_str, fontsize=22)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=3.0)

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
            log(f"✓ Energy distribution figure saved to {filepath}")

        show_figure()
