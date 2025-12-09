#!/usr/bin/env python3
"""
Comprehensive Network Visualization Script for Glacier Mapping and Physics-LSTM

This script creates publication-quality visualizations of:
1. Current architectures (U-Net, Physics Features, Training Pipeline)
2. Background networks (ANN, U-Net, LSTM, CNN) for thesis
3. Physics-LSTM architecture and temporal processing
4. Concept explanations (IoU, Semantic Segmentation)

Author: Generated for Glacier Mapping Thesis
Output: visualization/output/network_visualizations/
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.axes import Axes
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "text.usetex": False,
        "figure.figsize": (12, 8),
        "figure.dpi": 300,
        "axes.linewidth": 1.0,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
    }
)

# Color scheme consistent with glacier mapping
COLORS = {
    "input": "#4FC3F7",  # Cyan
    "conv": "#66BB6A",  # Green
    "lstm": "#FFA726",  # Orange
    "dense": "#EF5350",  # Red
    "physics": "#AB47BC",  # Purple
    "combine": "#FFEE58",  # Yellow
    "output": "#9CCC65",  # Light Green
    "skip": "#42A5F5",  # Blue
    "pool": "#FF7043",  # Deep Orange
    "bg": "#78909C",  # Blue Grey
    "clean_ice": "#0078FF",  # Blue
    "debris": "#C85000",  # Orange
    "intersection": "#4CAF50",  # Green
    "union": "#FFC107",  # Amber
}

class NetworkVisualizer:
    """Main class for creating network architecture visualizations."""

    def __init__(self, output_dir: str = "visualization/output/network_visualizations"):
        self.output_dir = Path(output_dir)
        self.create_output_structure()

    def create_output_structure(self):
        """Create organized output directory structure."""
        dirs = [
            "current_architecture",
            "background_networks",
            "physics_lstm",
            "concepts",
            "formats",
        ]
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory structure created: {self.output_dir}")

    def save_figure(self, fig, filename: str, subdir: str = ""):
        """Save figure in multiple formats."""
        if subdir:
            base_path = self.output_dir / subdir / filename
        else:
            base_path = self.output_dir / filename

        # Save in multiple formats
        for fmt in ["png", "pdf", "svg"]:
            path = base_path.with_suffix(f".{fmt}")
            # fig.tight_layout()
            # fig.savefig(path, dpi=300, format=fmt)
            if fmt == "png":
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            else:
                fig.savefig(path, bbox_inches="tight", pad_inches=0.1, format=fmt)
            print(f"✓ Saved {fmt.upper()}: {path}")

    def draw_layer_box(
        self,
        ax: Axes,
        x: float,
        y: float,
        width: float,
        height: float,
        color: str = "lightblue",
        edge_color: str = "black",
        alpha: float = 0.9,
        depth_offset: float = 0.008,
    ) -> None:
        """Draw a 3D-looking box for neural network layers."""
        x_left = x - width / 2
        y_bottom = y - height / 2

        # Main front face
        front = FancyBboxPatch(
            (x_left, y_bottom),
            width,
            height,
            boxstyle="round,pad=0.002",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(front)

        # Top face (3D effect)
        top_points = np.array(
            [
                [x_left, y_bottom + height],
                [x_left + depth_offset, y_bottom + height + depth_offset],
                [x_left + width + depth_offset, y_bottom + height + depth_offset],
                [x_left + width, y_bottom + height],
            ]
        )
        top = mpatches.Polygon(
            top_points,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.2,
            alpha=alpha * 0.6,
            zorder=2,
        )
        ax.add_patch(top)

        # Right face (3D effect)
        right_points = np.array(
            [
                [x_left + width, y_bottom],
                [x_left + width + depth_offset, y_bottom + depth_offset],
                [x_left + width + depth_offset, y_bottom + height + depth_offset],
                [x_left + width, y_bottom + height],
            ]
        )
        right = mpatches.Polygon(
            right_points,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.2,
            alpha=alpha * 0.4,
            zorder=1,
        )
        ax.add_patch(right)

    def draw_arrow(
        self,
        ax: Axes,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "black",
        linewidth: float = 1.5,
        style: str = "-",
    ) -> None:
        """Draw an arrow between two points."""
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=15,
            linewidth=linewidth,
            color=color,
            linestyle=style,
            zorder=5,
        )
        ax.add_patch(arrow)

    def add_title(self, ax: Axes, title: str, y_pos: float = 0.95) -> None:
        """Add title to axis."""
        ax.text(
            0.5,
            y_pos,
            title,
            ha="center",
            va="top",
            fontsize=14,
            weight="bold",
            transform=ax.transAxes,
        )

    def create_physics_lstm_architecture(self):
        """Create Physics-LSTM architecture diagram based on actual implementation."""
        fig, ax = plt.subplots(figsize=(11.5, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Scale factor: height = neurons * scale
        neuron_scale = 0.0025
        box_width = 0.035
        special_width = 0.10

        # Y positions for two branches
        lstm_y = 0.68
        pinns_y = 0.32

        # Input layer
        input_x = 0.05
        input_y = 0.5
        input_neurons = 7
        input_height = input_neurons * neuron_scale

        self.draw_layer_box(
            ax, input_x, input_y, box_width, 0.25, color=COLORS["input"]
        )
        ax.text(
            input_x,
            input_y + input_height / 2 + 0.15,
            "7",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            input_x,
            input_y - input_height / 2 + .1,
            "Input",
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.text(
            input_x,
            input_y - input_height / 2 ,
            "X,Y,T\nVu,Vv\nP,W.VF",
            ha="center",
            va="top",
            fontsize=6,
            style="italic",
        )

        # LSTM Branch (top)
        x_pos = 0.17

        # LSTM Layer 1 (Bidirectional: 32 units)
        lstm1_neurons = 32
        lstm1_height = lstm1_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, lstm_y, box_width, lstm1_height, color=COLORS["lstm"]
        )
        ax.text(
            x_pos,
            lstm_y + lstm1_height / 2 + 0.015,
            "32",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            lstm_y - lstm1_height / 2 - 0.01,
            "Bi-LSTM\nL1",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax, input_x + box_width / 2, input_y, x_pos - box_width / 2 - 0.005, lstm_y
        )

        x_pos += 0.08

        # LSTM Layer 2 (Bidirectional: 32 units)
        lstm2_neurons = 32
        lstm2_height = lstm2_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, lstm_y, box_width, lstm2_height, color=COLORS["lstm"]
        )
        ax.text(
            x_pos,
            lstm_y + lstm2_height / 2 + 0.015,
            "32",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            lstm_y - lstm2_height / 2 - 0.01,
            "Bi-LSTM\nL2",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            lstm_y,
            x_pos - box_width / 2 - 0.005,
            lstm_y,
        )

        x_pos += 0.08

        # TimeDistributed (64 → 32)
        td_neurons = 32
        td_height = td_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, lstm_y, box_width, td_height, color=COLORS["dense"]
        )
        ax.text(
            x_pos,
            lstm_y + td_height / 2 + 0.015,
            "32",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            lstm_y - td_height / 2 - 0.01,
            "Time\nDistrib",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            lstm_y,
            x_pos - box_width / 2 - 0.005,
            lstm_y,
            linewidth=2,
        )

        x_pos += 0.08

        # Dense Layer 1 (32 → 32, ReLU)
        dense1_neurons = 32
        dense1_height = dense1_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, lstm_y, box_width, dense1_height, color=COLORS["dense"]
        )
        ax.text(
            x_pos,
            lstm_y + dense1_height / 2 + 0.015,
            "32",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            lstm_y - dense1_height / 2 - 0.01,
            "Dense\nReLU",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            lstm_y,
            x_pos - box_width / 2 - 0.005,
            lstm_y,
        )

        x_pos += 0.08

        # Dense Layer 2 (32 → 32, ReLU)
        dense2_neurons = 32
        dense2_height = dense2_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, lstm_y, box_width, dense2_height, color=COLORS["dense"]
        )
        ax.text(
            x_pos,
            lstm_y + dense2_height / 2 + 0.015,
            "32",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            lstm_y - dense2_height / 2 - 0.01,
            "Dense\nReLU",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            lstm_y,
            x_pos - box_width / 2 - 0.005,
            lstm_y,
        )

        x_pos += 0.08

        # LSTM Output (32 → 2)
        lstm_out_neurons = 2
        lstm_out_height = max(lstm_out_neurons * neuron_scale, 0.015)
        self.draw_layer_box(
            ax, x_pos, lstm_y, box_width, lstm_out_height, color=COLORS["output"]
        )
        ax.text(
            x_pos,
            lstm_y + lstm_out_height / 2 + 0.015,
            "2",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            lstm_y - lstm_out_height / 2 - 0.01,
            "Output\n(u,v)",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            lstm_y,
            x_pos - box_width / 2 - 0.005,
            lstm_y,
        )

        lstm_final_x = x_pos

        # PINNs Branch (bottom)
        x_pos = 0.17

        # Dense Layer 1 (7 → 32, tanh)
        pinns1_neurons = 32
        pinns1_height = pinns1_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, pinns_y, box_width, pinns1_height, color=COLORS["physics"]
        )
        ax.text(
            x_pos,
            pinns_y + pinns1_height / 2 + 0.015,
            "32",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y - pinns1_height / 2 - 0.01,
            "Dense\ntanh",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax, input_x + box_width / 2, input_y, x_pos - box_width / 2 - 0.005, pinns_y
        )

        x_pos += 0.08

        # Dense Layer 2 (32 → 64, tanh)
        pinns2_neurons = 64
        pinns2_height = pinns2_neurons * neuron_scale
        self.draw_layer_box(
            ax, x_pos, pinns_y, box_width, pinns2_height, color=COLORS["physics"]
        )
        ax.text(
            x_pos,
            pinns_y + pinns2_height / 2 + 0.015,
            "64",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y - pinns2_height / 2 - 0.01,
            "Dense\ntanh",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            pinns_y,
            x_pos - box_width / 2 - 0.005,
            pinns_y,
        )

        x_pos += 0.08

        # Output (64 → 2: ψ, p)
        pinns_out_neurons = 2
        pinns_out_height = max(pinns_out_neurons * neuron_scale, 0.015)
        self.draw_layer_box(
            ax, x_pos, pinns_y, box_width, pinns_out_height, color=COLORS["physics"]
        )
        ax.text(
            x_pos,
            pinns_y + pinns_out_height / 2 + 0.015,
            "2",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y - pinns_out_height / 2 - 0.01,
            "Outout\n(ψ, p)",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
            pinns_y,
            x_pos - box_width / 2 - 0.005,
            pinns_y,
        )

        x_pos += 0.11

        # Physics Constraints Box
        physics_height = 0.10
        self.draw_layer_box(
            ax, x_pos, pinns_y, special_width, physics_height, color=COLORS["physics"]
        )
        ax.text(
            x_pos,
            pinns_y + physics_height / 2 + 0.015,
            "Autograd Physics",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

        # Physics equations
        ax.text(
            x_pos,
            pinns_y + 0.02,
            r"$u = \frac{\partial\psi}{\partial y}$",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            color="white",
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y - 0.02,
            r"$v = -\frac{\partial\psi}{\partial x}$",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            color="white",
            weight="bold",
        )

        self.draw_arrow(
            ax,
            x_pos - 0.11 + box_width / 2,
            pinns_y,
            x_pos - special_width / 2 - 0.005,
            pinns_y,
        )

        x_pos += 0.13

        # PINNs Final Output (u, v)
        pinns_final_neurons = 2
        pinns_final_height = max(pinns_final_neurons * neuron_scale, 0.015)
        self.draw_layer_box(
            ax, x_pos, pinns_y, box_width, pinns_final_height, color=COLORS["output"]
        )
        ax.text(
            x_pos,
            pinns_y + pinns_final_height / 2 + 0.015,
            "2",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y - pinns_final_height / 2 - 0.01,
            "Output\n(u,v)",
            ha="center",
            va="top",
            fontsize=7,
        )
        self.draw_arrow(
            ax,
            x_pos - 0.13 + special_width / 2,
            pinns_y,
            x_pos - box_width / 2 - 0.005,
            pinns_y,
        )

        pinns_final_x = x_pos

        # Loss function
        loss_x = 0.35
        loss_y = 0.5
        ax.text(
            loss_x,
            loss_y,
            r"Loss: $L_{data}(u,v) + L_{physics}(f_u, f_v)$",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightcyan",
                alpha=0.95,
                edgecolor="darkblue",
                linewidth=2,
            ),
        )

        # Combination layer
        combine_x = 0.72
        combine_y = 0.5
        combine_height = 0.12

        # Final output
        final_x = 0.65
        final_y = combine_y
        final_neurons = 2
        final_height = max(final_neurons * neuron_scale, 0.015)

        self.draw_layer_box(
            ax, final_x, final_y, box_width, final_height, color=COLORS["output"]
        )
        ax.text(
            final_x,
            final_y + final_height / 2 + 0.015,
            "2",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            final_x,
            final_y - final_height / 2 - 0.01,
            "Final\n(u,v)",
            ha="center",
            va="top",
            fontsize=7,
        )

        # Arrows from both branches directly to final output
        self.draw_arrow(
            ax,
            lstm_final_x + box_width / 2,
            lstm_y,
            final_x - box_width / 2 - 0.005,
            final_y + 0.01,
            color="darkgreen",
            linewidth=2.5,
        )
        self.draw_arrow(
            ax,
            pinns_final_x + box_width / 2,
            pinns_y,
            final_x - box_width / 2 - 0.005,
            final_y - 0.01,
            color="darkred",
            linewidth=2.5,
        )

        # Alpha labels centered on arrow lines
        ax.text(
            (lstm_final_x + box_width / 2 + final_x - box_width / 2 - 0.005) / 2 + 0.01,
            (lstm_y + final_y + 0.01) / 2,
            r"$\alpha$",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            weight="bold",
            color="darkgreen",
        )
        ax.text(
            (pinns_final_x + box_width / 2 + final_x - box_width / 2 - 0.005) / 2 + 0.03,
            (pinns_y + final_y - 0.01) / 2,
            r"$1-\alpha$",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            weight="bold",
            color="darkred",
        )
        final_neurons = 2
        final_height = max(final_neurons * neuron_scale, 0.015)

        self.draw_layer_box(
            ax, final_x, final_y, box_width, final_height, color=COLORS["output"]
        )
        ax.text(
            final_x,
            final_y + final_height / 2 + 0.015,
            "2",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            final_x,
            final_y - final_height / 2 - 0.01,
            "Final\n(u,v)",
            ha="center",
            va="top",
            fontsize=7,
        )
        # Single arrow from both branches directly to final output
        # self.draw_arrow(
        #     ax,
        #     (lstm_final_x + pinns_final_x) / 2 + box_width / 2,
        #     (lstm_y + pinns_y) / 2,
        #     final_x - box_width / 2 - 0.02,
        #     final_y,
        #     linewidth=3,
        # )

        # Title
        # ax.add_title("Physics-Informed LSTM Architecture", loc="center")
        fig = plt.gcf()
        # fig.suptitle("Physics-Informed LSTM Architecture", x=0.5, y=0.98, fontsize=14)

        # fig.suptitle("", x=0.5, y=0.98)
        ax.text(
            0.35,
            lstm_y + 0.2,
            "Physics-Informed LSTM Architecture",
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
            color="black",
            # bbox=dict(
            #     boxstyle="round,pad=0.4",
            #     facecolor="white",
            #     alpha=0.9,
            #     edgecolor="darkgreen",
            #     linewidth=2,
            # ),
        )

        # Branch labels
        ax.text(
            0.35,
            lstm_y + 0.10,
            "Data-Driven Branch (LSTM)",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="darkgreen",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.9,
                edgecolor="darkgreen",
                linewidth=2,
            ),
        )

        ax.text(
            0.35,
            pinns_y - 0.15,
            "Physics-Informed Branch (PINNs)",
            ha="center",
            va="top",
            fontsize=11,
            weight="bold",
            color="darkred",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.9,
                edgecolor="darkred",
                linewidth=2,
            ),
        )

        # Learnable parameters
        # ax.text(
        #     0.35,
        #     0.47,
        #     r"Learnable: $\lambda_2$ (viscosity), $w_{lstm}$ (weight)",
        #     ha="center",
        #     va="top",
        #     fontsize=7,
        #     style="italic",
        #     bbox=dict(
        #         boxstyle="round,pad=0.3",
        #         facecolor="lightyellow",
        #         alpha=0.9,
        #         edgecolor="gray",
        #         linewidth=1,
        #     ),
        # )

        self.save_figure(fig, "physics_lstm_architecture", "physics_lstm")
        plt.close(fig)

def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("NETWORK VISUALIZATION SCRIPT")
    print("=" * 60)

    visualizer = NetworkVisualizer()

    print("\n3. Creating Physics-LSTM Visualizations...")
    try:
        visualizer.create_physics_lstm_architecture()
        print("   ✓ Physics-LSTM Architecture")
    except Exception as e:
        print(f"   ✗ Physics-LSTM Architecture: {e}")

    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
