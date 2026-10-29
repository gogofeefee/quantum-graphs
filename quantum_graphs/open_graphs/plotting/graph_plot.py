#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPlotMixin — plot_graph (shared, see quantum_graphs.plotting) and
plot_energy_distribution.
"""

from ...core.config import log, show_figure
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ...plotting.drawing import (add_colorbar, draw_bond_values, draw_vertices, finish_axes,
                                lead_direction, size_scale)
from ...plotting.topology_plot import TopologyPlotMixin


class GraphPlotMixin(TopologyPlotMixin):

    def plot_energy_distribution(self, k, injection_port, show_leads=True,
                                 cmap=None, save=False, save_fig=None, file_format="pdf"):
        """
        Plot the energy distribution on the graph bonds for a given injection port
        and wavenumber.
        """
        if save_fig is not None:          # same keyword as the other modules
            save = save_fig
        leads = list(self.topology.leads)

        # Identify injection port
        if isinstance(injection_port, int):
            injection_index = injection_port
        else:
            injection_index = leads.index(tuple(injection_port))

        # Compute internal amplitudes
        e_j               = np.zeros((len(leads),), dtype=complex)
        e_j[injection_index] = 1.0

        # amplitudes arriving at the end of each directed bond
        a_alpha = self.model.internal_amplitudes(k, e_j)

        # Energy density per bond, and entropy
        energy_per_bond = self.bond_intensities(a_alpha)
        S_bonds, S_norm_, _ = self.localization_measures(a_alpha)

        max_energy   = np.max(energy_per_bond) if np.max(energy_per_bond) > 0 else 1.0
        norm_energies = energy_per_bond / max_energy

        fig, ax = plt.subplots(figsize=(10, 8))
        paths = self.bond_paths
        edges = draw_bond_values(ax, paths, norm_energies, cmap=cmap)
        draw_vertices(ax, self.topology, 0.065 * size_scale(self.topology))
        add_colorbar(ax, edges, 'Energy Density')

        arrow_tips = []
        if show_leads:
            S            = self.compute_S_matrix(k)
            power_matrix = np.abs(S)**2
            j_idx        = injection_index
            arrow_length = 80
            arrow_offset = 5
            node_margin  = 15

            for lead_idx, (v, p) in enumerate(leads):
                direction = lead_direction(self.lead_orientations, (v, p))
                if direction is None:
                    continue
                start_pt  = np.array(self.topology.positions[v])
                arrow_tips.append(start_pt + (arrow_length + node_margin) * direction)

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
                    arrow_tips.append(label_pos + [1.2 * arrow_length, 0.0])   # text room
                    ax.text(label_pos[0], label_pos[1], rf"$R = {R_val:.2f}$",
                            fontsize=22, ha='left', va='center', clip_on=False)
                else:
                    ax.add_patch(FancyArrowPatch(outward_start, outward_end,
                                                 arrowstyle="simple,head_length=15,head_width=12,tail_width=4",
                                                 linewidth=1, color='black',
                                                 fill=True, clip_on=False))
                    T_val     = power_matrix[lead_idx, j_idx]
                    label_pos = outward_end + 0.5 * direction
                    arrow_tips.append(label_pos + [1.2 * arrow_length, 0.0])   # text room
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
        finish_axes(ax, paths, extra_points=arrow_tips)
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
