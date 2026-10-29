"""
TopologyPlotMixin — ``plot_graph``, the topology figure shared by every graph
class (see :mod:`quantum_graphs.plotting.drawing`).
"""

import os

import matplotlib.pyplot as plt

from ..core.config import log, show_figure
from .drawing import draw_graph


class TopologyPlotMixin:

    def plot_graph(self, show_lengths=False, show_port_markers=False, show_labels=True,
                   show_leads=True, vertex_radius_frac=0.065, marker_radius_frac=0.03,
                   vertex_fontsize=20, length_fontsize=16, bond_linewidth=3.0,
                   bond_color="black", ax=None, title=None, figsize=(10, 8), save_fig=False, file_format="pdf",
                   filename_suffix=None):
        """
        Plot the graph: bonds (parallel bonds as arcs, self-loops as teardrops),
        vertices with their labels (1..N, access couplers "A"), and optionally
        the bond lengths, the port markers of 4-port vertices (blue: ports 0, 1;
        orange: ports 2, 3) and the leads (dashed, if ``lead_orientations`` is
        known).

        Sizes are fractions of the shortest bond length. With ``ax`` the graph
        is drawn into that axes (as a background) and not shown. Returns
        ``(fig, ax)``.
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        draw_graph(ax, self.bond_paths, show_lengths=show_lengths,
                   show_port_markers=show_port_markers, show_labels=show_labels,
                   lead_orientations=getattr(self, "lead_orientations", None) if show_leads else None,
                   vertex_radius_frac=vertex_radius_frac, marker_radius_frac=marker_radius_frac,
                   vertex_fontsize=vertex_fontsize, length_fontsize=length_fontsize,
                   bond_linewidth=bond_linewidth, bond_color=bond_color)
        if title:
            ax.set_title(title)
        if created:
            fig.tight_layout(pad=1.0)
        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            tags = ["with_lengths" if show_lengths else "no_lengths"]
            if show_port_markers:
                tags.append("port_markers")
            if filename_suffix:
                tags.append(filename_suffix)
            filepath = os.path.join(self.save_path,
                                    f"{self.name}_graph_{'_'.join(tags)}.{file_format}")
            fig.savefig(filepath, format=file_format, dpi=300, bbox_inches="tight")
            log(f"Graph plot saved to {filepath}")
        if created:
            show_figure()
        return fig, ax
