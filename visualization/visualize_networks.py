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

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.axes import Axes
import yaml
from pathlib import Path
from typing import Tuple, Any, List, Optional
import cv2

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import glacier_mapping.utils.visualize as viz
    import glacier_mapping.data.physics as physics
    from glacier_mapping.utils.logging import log

    HAS_GLACIER_MAPPING = True
except ImportError:
    HAS_GLACIER_MAPPING = False
    print(
        "Warning: glacier_mapping modules not available, using fallback implementations"
    )

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available, satellite data visualization disabled")

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

    def create_unet_architecture(self, config_path: str = "configs/train.yaml"):
        """Create U-Net architecture diagram based on actual implementation."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Load config for actual parameters
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_opts = config.get("model_opts", {}).get("args", {})
            net_depth = model_opts.get("net_depth", 4)
            first_channels = model_opts.get("first_channel_output", 32)
            dropout = model_opts.get("dropout", 0.1)
        except:
            net_depth = 4
            first_channels = 32
            dropout = 0.1

        # Calculate channel dimensions
        encoder_channels = [first_channels * (2**i) for i in range(net_depth)]
        decoder_channels = encoder_channels[::-1]

        # Input layer
        input_x, input_y = 0.08, 0.5
        input_height = 0.15
        self.draw_layer_box(
            ax,
            input_x,
            input_y,
            0.04,
            input_height,
            color=COLORS["input"],
            edge_color="black",
        )
        ax.text(
            input_x,
            input_y + input_height / 2 + 0.02,
            "Input",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            input_x,
            input_y - input_height / 2 - 0.02,
            "H×W×C",
            ha="center",
            va="top",
            fontsize=8,
        )

        # Encoder path (left side)
        x_pos = 0.18
        encoder_outputs = []

        for i, channels in enumerate(encoder_channels):
            y_pos = 0.7 - i * 0.12
            height = 0.03 + (channels / max(encoder_channels)) * 0.08

            # ConvBlock
            self.draw_layer_box(
                ax, x_pos, y_pos, 0.04, height, color=COLORS["conv"], edge_color="black"
            )
            ax.text(
                x_pos,
                y_pos + height / 2 + 0.015,
                str(channels),
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )
            ax.text(
                x_pos,
                y_pos - height / 2 - 0.01,
                f"ConvBlock\n{i + 1}",
                ha="center",
                va="top",
                fontsize=7,
            )

            # Arrow from previous
            if i == 0:
                self.draw_arrow(ax, input_x + 0.02, input_y, x_pos - 0.02, y_pos)
            else:
                prev_y = 0.7 - (i - 1) * 0.12
                self.draw_arrow(ax, x_pos + 0.02, prev_y, x_pos - 0.02, y_pos)

            # Pooling arrow
            if i < net_depth - 1:
                pool_y = y_pos - 0.04
                self.draw_arrow(
                    ax,
                    x_pos + 0.02,
                    y_pos,
                    x_pos + 0.02,
                    pool_y,
                    color=COLORS["pool"],
                    linewidth=2,
                )
                ax.text(
                    x_pos + 0.06,
                    pool_y,
                    "MaxPool\n2×2",
                    ha="left",
                    va="center",
                    fontsize=6,
                    color=COLORS["pool"],
                )

            encoder_outputs.append((x_pos, y_pos, height))

        # Middle convolution
        middle_x = x_pos + 0.12
        middle_y = 0.7 - (net_depth - 1) * 0.12
        middle_channels = encoder_channels[-1] * 2
        middle_height = 0.03 + (middle_channels / max(encoder_channels)) * 0.08

        self.draw_layer_box(
            ax,
            middle_x,
            middle_y,
            0.04,
            middle_height,
            color=COLORS["conv"],
            edge_color="black",
        )
        ax.text(
            middle_x,
            middle_y + middle_height / 2 + 0.015,
            str(middle_channels),
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
        ax.text(
            middle_x,
            middle_y - middle_height / 2 - 0.01,
            "Middle\nConv",
            ha="center",
            va="top",
            fontsize=7,
        )

        # Arrow from last encoder
        last_enc_x, last_enc_y, _ = encoder_outputs[-1]
        self.draw_arrow(ax, last_enc_x + 0.02, last_enc_y, middle_x - 0.02, middle_y)

        # Decoder path (right side)
        decoder_x = middle_x + 0.12

        for i, channels in enumerate(decoder_channels):
            y_pos = 0.7 - (net_depth - 1 - i) * 0.12
            height = 0.03 + (channels / max(encoder_channels)) * 0.08

            # UpBlock
            self.draw_layer_box(
                ax,
                decoder_x,
                y_pos,
                0.04,
                height,
                color=COLORS["conv"],
                edge_color="black",
            )
            ax.text(
                decoder_x,
                y_pos + height / 2 + 0.015,
                str(channels),
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )
            ax.text(
                decoder_x,
                y_pos - height / 2 - 0.01,
                f"UpBlock\n{i + 1}",
                ha="center",
                va="top",
                fontsize=7,
            )

            # Arrow from previous
            if i == 0:
                self.draw_arrow(ax, middle_x + 0.02, middle_y, decoder_x - 0.02, y_pos)
            else:
                prev_y = 0.7 - (net_depth - i) * 0.12
                self.draw_arrow(ax, decoder_x + 0.02, prev_y, decoder_x - 0.02, y_pos)

            # Skip connection
            skip_idx = net_depth - 1 - i
            if skip_idx < len(encoder_outputs):
                skip_x, skip_y, skip_height = encoder_outputs[skip_idx]
                self.draw_arrow(
                    ax,
                    skip_x + 0.02,
                    skip_y,
                    decoder_x - 0.02,
                    y_pos,
                    color=COLORS["skip"],
                    linewidth=2,
                    style="--",
                )

        # Final output layer
        output_x = decoder_x + 0.12
        output_y = 0.5
        output_height = 0.08

        self.draw_layer_box(
            ax,
            output_x,
            output_y,
            0.04,
            output_height,
            color=COLORS["output"],
            edge_color="black",
        )
        ax.text(
            output_x,
            output_y + output_height / 2 + 0.015,
            "N",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
        ax.text(
            output_x,
            output_y - output_height / 2 - 0.01,
            "Output\nClasses",
            ha="center",
            va="top",
            fontsize=7,
        )

        # Arrow from last decoder
        last_dec_y = 0.7
        self.draw_arrow(ax, decoder_x + 0.02, last_dec_y, output_x - 0.02, output_y)

        # Title and labels
        self.add_title(ax, "U-Net Architecture for Glacier Segmentation")

        # Add legend
        legend_items = [
            ("Input", COLORS["input"]),
            ("ConvBlock", COLORS["conv"]),
            ("MaxPool", COLORS["pool"]),
            ("UpBlock", COLORS["conv"]),
            ("Skip Connection", COLORS["skip"]),
            ("Output", COLORS["output"]),
        ]

        legend_x = 0.85
        legend_y = 0.8
        ax.text(
            legend_x,
            legend_y + 0.05,
            "Legend",
            ha="left",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

        for i, (label, color) in enumerate(legend_items):
            y_offset = legend_y - i * 0.04
            rect = Rectangle(
                (legend_x, y_offset - 0.01),
                0.015,
                0.02,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                legend_x + 0.02, y_offset, label, ha="left", va="center", fontsize=8
            )

        # Add configuration info
        config_text = (
            f"Depth: {net_depth}\nFirst Channels: {first_channels}\nDropout: {dropout}"
        )
        ax.text(
            0.02,
            0.15,
            config_text,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8),
        )

        self.save_figure(fig, "unet_architecture", "current_architecture")
        plt.close(fig)

    def create_physics_features(self):
        """Create visualization of physics features from satellite data."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Physics Features for Glacier Mapping", fontsize=16, weight="bold")

        # Create sample physics features (since we may not have real data)
        size = 100
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)

        # Flow accumulation (radial pattern)
        center_x, center_y = size // 2, size // 2
        flow = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        flow = np.exp(-flow / 3) * 100

        # TPI (topographic position index)
        tpi = np.sin(X / 2) * np.cos(Y / 2) + np.random.normal(0, 0.1, (size, size))

        # Roughness (texture)
        roughness = np.random.gamma(2, 1, (size, size))
        roughness = cv2.GaussianBlur(roughness, (15, 15), 0)

        # Plan curvature
        curvature = np.gradient(np.gradient(X + np.sin(Y))[0])[1]

        features = [
            (flow, "Flow Accumulation", "plasma"),
            (tpi, "Topographic Position Index (TPI)", "RdBu_r"),
            (roughness, "Surface Roughness", "cividis"),
            (curvature, "Plan Curvature", "Spectral"),
        ]

        for idx, (ax, (data, title, cmap)) in enumerate(zip(axes.flat, features)):
            im = ax.imshow(data, cmap=cmap, aspect="auto")
            ax.set_title(title, fontsize=12, weight="bold")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Value", rotation=270, labelpad=15)

        plt.tight_layout()
        self.save_figure(fig, "physics_features", "current_architecture")
        plt.close(fig)

    def create_training_pipeline(self):
        """Create end-to-end training pipeline diagram."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Pipeline stages
        stages = [
            (0.1, 0.8, "Satellite\nData", COLORS["input"], 0.12, 0.08),
            (0.25, 0.8, "Preprocessing\n& Physics", COLORS["physics"], 0.12, 0.08),
            (0.4, 0.8, "U-Net\nModel", COLORS["conv"], 0.12, 0.08),
            (0.55, 0.8, "Segmentation\nOutput", COLORS["output"], 0.12, 0.08),
            (0.7, 0.8, "Loss\nComputation", COLORS["dense"], 0.12, 0.08),
            (0.85, 0.8, "Metrics\n& Evaluation", COLORS["combine"], 0.12, 0.08),
        ]

        # Draw stages
        for x, y, label, color, width, height in stages:
            self.draw_layer_box(
                ax, x, y, width, height, color=color, edge_color="black"
            )
            ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")

        # Draw arrows
        for i in range(len(stages) - 1):
            x1, y1, _, _, w1, _ = stages[i]
            x2, y2, _, _, _, _ = stages[i + 1]
            self.draw_arrow(ax, x1 + w1 / 2, y1, x2 - w1 / 2, y2, linewidth=2)

        # Add details for each stage
        details = [
            (0.1, 0.6, "• Landsat-7 Imagery\n• 8 Spectral Bands\n• DEM Data"),
            (0.25, 0.6, "• Normalization\n• Physics Features\n• Data Augmentation"),
            (0.4, 0.6, "• Encoder-Decoder\n• Skip Connections\n• Convolutional Layers"),
            (0.55, 0.6, "• Clean Ice Mask\n• Debris Mask\n• Background"),
            (0.7, 0.6, "• Cross-Entropy\n• Class Weights\n• Backpropagation"),
            (0.85, 0.6, "• IoU, Precision\n• Recall, F1-Score\n• Validation"),
        ]

        for x, y, text in details:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        # Add feedback loop
        self.draw_arrow(ax, 0.85, 0.4, 0.4, 0.4, color="red", style="--", linewidth=2)
        ax.text(
            0.6,
            0.35,
            "Backpropagation & Weight Updates",
            ha="center",
            va="center",
            fontsize=8,
            color="red",
            style="italic",
        )

        # Title
        self.add_title(ax, "Glacier Mapping Training Pipeline")

        # Add epoch counter
        ax.text(
            0.5,
            0.15,
            "Epoch Loop: 1 → 200",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        self.save_figure(fig, "training_pipeline", "current_architecture")
        plt.close(fig)

    def create_simple_ann(self):
        """Create simple ANN diagram for thesis background."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Network layers
        layers = [
            (0.15, 0.5, "Input\nLayer", 4, COLORS["input"]),
            (0.4, 0.5, "Hidden\nLayer 1", 6, COLORS["dense"]),
            (0.65, 0.5, "Hidden\nLayer 2", 5, COLORS["dense"]),
            (0.85, 0.5, "Output\nLayer", 3, COLORS["output"]),
        ]

        # Draw layers
        layer_positions = []
        for x, y, label, neurons, color in layers:
            # Draw neurons
            neuron_positions = []
            for i in range(neurons):
                neuron_y = y + (i - neurons / 2 + 0.5) * 0.08
                circle = Circle(
                    (x, neuron_y),
                    0.025,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.add_patch(circle)
                neuron_positions.append((x, neuron_y))

            layer_positions.append(neuron_positions)

            # Add layer label
            ax.text(
                x, y - 0.15, label, ha="center", va="top", fontsize=10, weight="bold"
            )

        # Draw connections
        for i in range(len(layer_positions) - 1):
            current_layer = layer_positions[i]
            next_layer = layer_positions[i + 1]

            for x1, y1 in current_layer:
                for x2, y2 in next_layer:
                    # Draw connection with transparency
                    ax.plot([x1, x2], [y1, y2], "gray", alpha=0.3, linewidth=0.5)

        # Add mathematical notation
        ax.text(
            0.5,
            0.15,
            r"$y = \sigma(W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2)$",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        # Title
        self.add_title(ax, "Simple Artificial Neural Network (ANN)")

        # Add legend
        ax.text(
            0.02,
            0.95,
            "σ = Sigmoid/ReLU activation",
            ha="left",
            va="top",
            fontsize=9,
            style="italic",
        )

        self.save_figure(fig, "simple_ann", "background_networks")
        plt.close(fig)

    def create_standard_unet(self):
        """Create standard U-Net diagram for comparison."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Encoder path
        encoder_sizes = [64, 128, 256, 512]
        decoder_sizes = [256, 128, 64]

        # Input
        input_x, input_y = 0.1, 0.5
        self.draw_layer_box(
            ax, input_x, input_y, 0.03, 0.15, color=COLORS["input"], edge_color="black"
        )
        ax.text(
            input_x,
            input_y + 0.1,
            "Input\n512×512×3",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

        # Encoder
        x_pos = 0.2
        encoder_positions = []
        for i, size in enumerate(encoder_sizes):
            y_pos = 0.5
            height = 0.02 + (size / max(encoder_sizes)) * 0.06

            self.draw_layer_box(
                ax,
                x_pos,
                y_pos,
                0.025,
                height,
                color=COLORS["conv"],
                edge_color="black",
            )
            ax.text(
                x_pos,
                y_pos + height / 2 + 0.02,
                f"{size}×{size}",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
            )
            ax.text(
                x_pos,
                y_pos - height / 2 - 0.02,
                f"Conv2D\n{i + 1}",
                ha="center",
                va="top",
                fontsize=7,
            )

            encoder_positions.append((x_pos, y_pos, height))

            # Arrow and pooling
            if i < len(encoder_sizes) - 1:
                self.draw_arrow(ax, x_pos + 0.0125, y_pos, x_pos + 0.0875, y_pos)
                pool_y = y_pos - 0.03
                self.draw_arrow(
                    ax,
                    x_pos + 0.1,
                    y_pos,
                    x_pos + 0.1,
                    pool_y,
                    color=COLORS["pool"],
                    linewidth=2,
                )
                ax.text(
                    x_pos + 0.13,
                    pool_y,
                    "2×2",
                    ha="left",
                    va="center",
                    fontsize=6,
                    color=COLORS["pool"],
                )
                x_pos += 0.15

        # Bottleneck
        bottleneck_x = x_pos + 0.08
        self.draw_layer_box(
            ax, bottleneck_x, 0.5, 0.025, 0.08, color=COLORS["conv"], edge_color="black"
        )
        ax.text(
            bottleneck_x,
            0.5 + 0.05,
            "1024×1024",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )
        ax.text(
            bottleneck_x, 0.5 - 0.05, "Bottleneck", ha="center", va="top", fontsize=7
        )

        # Decoder
        decoder_x = bottleneck_x + 0.12
        for i, size in enumerate(decoder_sizes):
            y_pos = 0.5
            height = 0.02 + (size / max(encoder_sizes)) * 0.06

            self.draw_layer_box(
                ax,
                decoder_x,
                y_pos,
                0.025,
                height,
                color=COLORS["conv"],
                edge_color="black",
            )
            ax.text(
                decoder_x,
                y_pos + height / 2 + 0.02,
                f"{size}×{size}",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
            )
            ax.text(
                decoder_x,
                y_pos - height / 2 - 0.02,
                f"UpConv\n{i + 1}",
                ha="center",
                va="top",
                fontsize=7,
            )

            # Skip connection
            skip_idx = len(encoder_sizes) - 2 - i
            if skip_idx >= 0:
                skip_x, skip_y, skip_height = encoder_positions[skip_idx]
                self.draw_arrow(
                    ax,
                    skip_x + 0.0125,
                    skip_y,
                    decoder_x - 0.0125,
                    y_pos,
                    color=COLORS["skip"],
                    linewidth=2,
                    style="--",
                )

            # Upsampling arrow
            if i < len(decoder_sizes) - 1:
                self.draw_arrow(
                    ax, decoder_x + 0.0125, y_pos, decoder_x + 0.0875, y_pos
                )
                up_y = y_pos + 0.03
                self.draw_arrow(
                    ax,
                    decoder_x + 0.1,
                    y_pos,
                    decoder_x + 0.1,
                    up_y,
                    color=COLORS["combine"],
                    linewidth=2,
                )
                ax.text(
                    decoder_x + 0.13,
                    up_y,
                    "2×2",
                    ha="left",
                    va="center",
                    fontsize=6,
                    color=COLORS["combine"],
                )
                decoder_x += 0.15

        # Output
        output_x = decoder_x + 0.08
        self.draw_layer_box(
            ax, output_x, 0.5, 0.025, 0.08, color=COLORS["output"], edge_color="black"
        )
        ax.text(
            output_x,
            0.5 + 0.05,
            "512×512×N",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )
        ax.text(output_x, 0.5 - 0.05, "Output\nMask", ha="center", va="top", fontsize=7)

        # Final arrow
        self.draw_arrow(ax, decoder_x + 0.0125, 0.5, output_x - 0.0125, 0.5)

        # Title
        self.add_title(ax, "Standard U-Net Architecture")

        # Add annotations
        ax.text(
            0.5,
            0.15,
            "Contracting Path → Bottleneck → Expansive Path",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )

        self.save_figure(fig, "standard_unet", "background_networks")
        plt.close(fig)

    def create_lstm_cell(self):
        """Create LSTM cell diagram with gates."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # LSTM cell components
        components = {
            "input": (0.1, 0.5, "Input\n$x_t$", COLORS["input"]),
            "prev_hidden": (0.1, 0.3, "Prev Hidden\n$h_{t-1}$", COLORS["lstm"]),
            "prev_cell": (0.1, 0.7, "Prev Cell\n$C_{t-1}$", COLORS["physics"]),
            "forget_gate": (0.3, 0.6, "Forget Gate\n$\sigma$", COLORS["dense"]),
            "input_gate": (0.3, 0.5, "Input Gate\n$\sigma$", COLORS["dense"]),
            "output_gate": (0.3, 0.4, "Output Gate\n$\sigma$", COLORS["dense"]),
            "candidate": (0.3, 0.3, "Candidate\n$\tanh$", COLORS["conv"]),
            "cell_state": (0.5, 0.7, "Cell State\n$C_t$", COLORS["physics"]),
            "hidden_output": (0.7, 0.5, "Hidden Output\n$h_t$", COLORS["output"]),
        }

        # Draw components
        for name, (x, y, label, color) in components.items():
            if "Gate" in name or "Candidate" in name:
                self.draw_layer_box(ax, x, y, 0.08, 0.06, color=color)
            else:
                circle = Circle(
                    (x, y), 0.04, facecolor=color, edgecolor="black", linewidth=1.5
                )
                ax.add_patch(circle)

            ax.text(
                x, y - 0.08, label, ha="center", va="top", fontsize=8, weight="bold"
            )

        # Draw connections
        connections = [
            # Input to gates
            ("input", "forget_gate"),
            ("input", "input_gate"),
            ("input", "output_gate"),
            ("input", "candidate"),
            # Previous hidden to gates
            ("prev_hidden", "forget_gate"),
            ("prev_hidden", "input_gate"),
            ("prev_hidden", "output_gate"),
            ("prev_hidden", "candidate"),
            # Gate operations
            ("forget_gate", "cell_state"),
            ("input_gate", "cell_state"),
            ("candidate", "cell_state"),
            ("prev_cell", "cell_state"),
            # Output
            ("cell_state", "output_gate"),
            ("output_gate", "hidden_output"),
        ]

        for from_comp, to_comp in connections:
            if from_comp in components and to_comp in components:
                x1, y1, _, _ = components[from_comp]
                x2, y2, _, _ = components[to_comp]

                # Adjust for circles vs boxes
                if "Gate" in to_comp or "Candidate" in to_comp:
                    x2 -= 0.04
                elif "Gate" in from_comp or "Candidate" in from_comp:
                    x1 += 0.04

                self.draw_arrow(ax, x1, y1, x2, y2, linewidth=1)

        # Add mathematical equations
        equations = [
            (0.5, 0.5, r"$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$"),
            (0.5, 0.4, r"$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$"),
            (0.5, 0.3, r"$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$"),
            (0.5, 0.2, r"$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$"),
            (0.5, 0.1, r"$h_t = o_t \odot \tanh(C_t)$"),
        ]

        for x, y, eq in equations:
            ax.text(
                x,
                y,
                eq,
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            )

        # Title
        self.add_title(ax, "LSTM Cell Architecture with Gates")

        # Legend
        legend_items = [
            ("σ = Sigmoid", COLORS["dense"]),
            ("⊙ = Element-wise multiply", "gray"),
            ("tanh = Hyperbolic tangent", COLORS["conv"]),
        ]

        legend_y = 0.85
        for i, (text, color) in enumerate(legend_items):
            ax.text(
                0.85,
                legend_y - i * 0.05,
                text,
                ha="left",
                va="center",
                fontsize=8,
                color=color,
            )

        self.save_figure(fig, "lstm_cell", "background_networks")
        plt.close(fig)

    def create_basic_cnn(self):
        """Create basic CNN diagram for thesis background."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # CNN layers
        layers = [
            (0.15, 0.5, "Input\nImage", (32, 32, 3), COLORS["input"]),
            (0.3, 0.5, "Conv\n3×3", (30, 30, 16), COLORS["conv"]),
            (0.45, 0.5, "Pool\n2×2", (15, 15, 16), COLORS["pool"]),
            (0.6, 0.5, "Conv\n3×3", (13, 13, 32), COLORS["conv"]),
            (0.75, 0.5, "FC\nLayer", 128, COLORS["dense"]),
            (0.9, 0.5, "Output\nClasses", 10, COLORS["output"]),
        ]

        # Draw layers
        for i, (x, y, label, shape, color) in enumerate(layers):
            if isinstance(shape, tuple):
                # Convolutional layers - show as 3D boxes
                h, w, c = shape
                if len(shape) == 3:  # 3D tensor
                    # Draw front face
                    rect = Rectangle(
                        (x - 0.03, y - 0.04),
                        0.06,
                        0.08,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)

                    # Draw depth indicator
                    depth_points = [
                        [x - 0.03, y - 0.04],
                        [x - 0.025, y - 0.045],
                        [x + 0.035, y - 0.045],
                        [x + 0.03, y - 0.04],
                    ]
                    depth_patch = mpatches.Polygon(
                        depth_points,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=1,
                        alpha=0.7,
                    )
                    ax.add_patch(depth_patch)

                    # Add channel info
                    ax.text(
                        x,
                        y + 0.06,
                        f"{c} channels",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
                else:  # 2D
                    rect = Rectangle(
                        (x - 0.03, y - 0.03),
                        0.06,
                        0.06,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)
            else:
                # Fully connected layers - show as circles
                circle = Circle(
                    (x, y), 0.04, facecolor=color, edgecolor="black", linewidth=1.5
                )
                ax.add_patch(circle)
                ax.text(x, y + 0.06, str(shape), ha="center", va="bottom", fontsize=7)

            ax.text(
                x, y - 0.08, label, ha="center", va="top", fontsize=8, weight="bold"
            )

            # Draw arrows
            if i < len(layers) - 1:
                next_x = layers[i + 1][0]
                self.draw_arrow(ax, x + 0.04, y, next_x - 0.04, y, linewidth=2)

        # Add feature map progression
        feature_text = (
            "Feature Maps:\n32×32×3 → 30×30×16 → 15×15×16 → 13×13×32 → 128 → 10"
        )
        ax.text(
            0.5,
            0.15,
            feature_text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        # Title
        self.add_title(ax, "Basic Convolutional Neural Network (CNN)")

        # Add annotations
        ax.text(
            0.3,
            0.35,
            "Feature\nExtraction",
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            color=COLORS["conv"],
        )
        ax.text(
            0.45,
            0.35,
            "Spatial\nReduction",
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            color=COLORS["pool"],
        )
        ax.text(
            0.75,
            0.35,
            "Classification",
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            color=COLORS["dense"],
        )

        self.save_figure(fig, "basic_cnn", "background_networks")
        plt.close(fig)

    def create_iou_sets(self):
        """Create IoU explanation using Venn diagram (sets)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Create Venn diagram circles
        circle1 = Circle(
            (0.35, 0.5),
            0.2,
            facecolor=COLORS["input"],
            edgecolor="black",
            linewidth=2,
            alpha=0.6,
        )
        circle2 = Circle(
            (0.45, 0.5),
            0.2,
            facecolor=COLORS["output"],
            edgecolor="black",
            linewidth=2,
            alpha=0.6,
        )
        ax.add_patch(circle1)
        ax.add_patch(circle2)

        # Highlight intersection
        intersection = Circle(
            (0.4, 0.5),
            0.12,
            facecolor=COLORS["intersection"],
            edgecolor="darkgreen",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(intersection)

        # Labels
        ax.text(
            0.3,
            0.5,
            "Ground\nTruth",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="darkblue",
        )
        ax.text(
            0.5,
            0.5,
            "Prediction",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="darkred",
        )
        ax.text(
            0.4,
            0.5,
            "Intersection",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="white",
        )

        # Mathematical formula
        ax.text(
            0.5,
            0.75,
            r"IoU = $\frac{|A \cap B|}{|A \cup B|}$",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
        )

        # Explanation
        explanation = "Where:\nA = Ground Truth pixels\nB = Predicted pixels\n∩ = Intersection\n∪ = Union"
        ax.text(
            0.5,
            0.25,
            explanation,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        # Title
        self.add_title(ax, "Intersection over Union (IoU) - Set Theory View")

        # Add legend
        legend_items = [
            ("Ground Truth Set", COLORS["input"]),
            ("Prediction Set", COLORS["output"]),
            ("Intersection", COLORS["intersection"]),
        ]

        legend_y = 0.85
        for i, (text, color) in enumerate(legend_items):
            circle = Circle(
                (0.75, legend_y - i * 0.08),
                0.02,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(circle)
            ax.text(0.78, legend_y - i * 0.08, text, ha="left", va="center", fontsize=9)

        self.save_figure(fig, "iou_sets", "concepts")
        plt.close(fig)

    def create_iou_shapes(self):
        """Create IoU explanation using geometric shapes (pixel-level)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Create grid
        grid_size = 10
        for i in range(grid_size + 1):
            ax.axhline(y=0.1 + i * 0.06, color="lightgray", linewidth=0.5)
            ax.axvline(x=0.2 + i * 0.06, color="lightgray", linewidth=0.5)

        # Ground truth shape (square)
        gt_pixels = [
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 3),
            (5, 4),
            (5, 5),
        ]
        for x, y in gt_pixels:
            rect = Rectangle(
                (0.2 + x * 0.06, 0.1 + y * 0.06),
                0.06,
                0.06,
                facecolor=COLORS["input"],
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Prediction shape (circle approximation)
        pred_pixels = [
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (3, 3),
            (3, 4),
            (3, 5),
            (5, 3),
            (5, 4),
            (5, 5),
        ]
        for x, y in pred_pixels:
            rect = Rectangle(
                (0.2 + x * 0.06, 0.1 + y * 0.06),
                0.06,
                0.06,
                facecolor=COLORS["output"],
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Highlight intersection
        intersection_pixels = [
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 3),
            (5, 4),
            (5, 5),
        ]
        for x, y in intersection_pixels:
            rect = Rectangle(
                (0.2 + x * 0.06, 0.1 + y * 0.06),
                0.06,
                0.06,
                facecolor=COLORS["intersection"],
                edgecolor="darkgreen",
                linewidth=2,
                alpha=0.9,
            )
            ax.add_patch(rect)

        # Count pixels
        intersection_count = len(intersection_pixels)
        union_count = len(set(gt_pixels + pred_pixels))
        iou_value = intersection_count / union_count

        # Add counts and formula
        ax.text(
            0.5,
            0.75,
            f"Intersection = {intersection_count} pixels",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=COLORS["intersection"], alpha=0.8
            ),
        )

        ax.text(
            0.5,
            0.65,
            f"Union = {union_count} pixels",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["union"], alpha=0.8),
        )

        ax.text(
            0.5,
            0.55,
            f"IoU = {intersection_count}/{union_count} = {iou_value:.3f}",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
        )

        # Title
        self.add_title(ax, "Intersection over Union (IoU) - Pixel-Level View")

        # Legend
        legend_items = [
            ("Ground Truth", COLORS["input"]),
            ("Prediction", COLORS["output"]),
            ("Intersection (Correct)", COLORS["intersection"]),
        ]

        legend_y = 0.35
        for i, (text, color) in enumerate(legend_items):
            rect = Rectangle(
                (0.65, legend_y - i * 0.08),
                0.03,
                0.03,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)
            ax.text(
                0.69, legend_y - i * 0.08, text, ha="left", va="center", fontsize=10
            )

        # Add grid labels
        ax.text(
            0.15,
            0.75,
            "Pixel Grid",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            rotation=90,
        )

        self.save_figure(fig, "iou_shapes", "concepts")
        plt.close(fig)

    def create_semantic_segmentation(self):
        """Create semantic segmentation explanation."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(
            "Semantic Segmentation: Pixel-Wise Classification",
            fontsize=16,
            weight="bold",
        )

        # Create sample scene
        size = 64
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)

        # Input image (simulated glacier scene)
        input_img = np.zeros((size, size, 3))

        # Background (rock/soil)
        input_img[:, :, 0] = 0.6  # R
        input_img[:, :, 1] = 0.4  # G
        input_img[:, :, 2] = 0.2  # B

        # Clean ice (blue-ish)
        ice_mask = (X - 5) ** 2 + (Y - 3) ** 2 < 4
        input_img[ice_mask] = [0.2, 0.5, 0.8]

        # Debris (brown-ish)
        debris_mask = (X - 7) ** 2 + (Y - 7) ** 2 < 3
        input_img[debris_mask] = [0.7, 0.3, 0.1]

        # Ground truth labels
        gt_labels = np.zeros((size, size))
        gt_labels[ice_mask] = 1  # Clean ice
        gt_labels[debris_mask] = 2  # Debris
        # Background remains 0

        # Prediction (with some errors)
        pred_labels = gt_labels.copy()
        # Add some false positives
        pred_labels[20:25, 20:25] = 1  # False positive ice
        # Add some false negatives
        pred_labels[ice_mask][0] = 0  # Miss some ice pixels

        # Plot input image
        axes[0].imshow(input_img)
        axes[0].set_title("Input Image\n(Satellite Data)", fontsize=12, weight="bold")
        axes[0].set_xlabel("Width (pixels)")
        axes[0].set_ylabel("Height (pixels)")

        # Plot ground truth
        gt_colors = np.zeros((size, size, 3))
        gt_colors[gt_labels == 0] = [0.5, 0.5, 0.5]  # Background - gray
        gt_colors[gt_labels == 1] = [0, 0.47, 1]  # Clean ice - blue
        gt_colors[gt_labels == 2] = [0.78, 0.31, 0]  # Debris - orange

        axes[1].imshow(gt_colors)
        axes[1].set_title("Ground Truth\n(Pixel Labels)", fontsize=12, weight="bold")
        axes[1].set_xlabel("Width (pixels)")
        axes[1].set_ylabel("Height (pixels)")

        # Plot prediction
        pred_colors = np.zeros((size, size, 3))
        pred_colors[pred_labels == 0] = [0.5, 0.5, 0.5]  # Background - gray
        pred_colors[pred_labels == 1] = [0, 0.47, 1]  # Clean ice - blue
        pred_colors[pred_labels == 2] = [0.78, 0.31, 0]  # Debris - orange

        axes[2].imshow(pred_colors)
        axes[2].set_title(
            "Model Prediction\n(Segmentation Output)", fontsize=12, weight="bold"
        )
        axes[2].set_xlabel("Width (pixels)")
        axes[2].set_ylabel("Height (pixels)")

        # Add class legend
        legend_elements = [
            ("Background", [0.5, 0.5, 0.5]),
            ("Clean Ice", [0, 0.47, 1]),
            ("Debris", [0.78, 0.31, 0]),
        ]

        for ax in axes:
            handles = []
            for label, color in legend_elements:
                from matplotlib.patches import Patch

                handles.append(Patch(facecolor=color, label=label))
            ax.legend(handles=handles, loc="upper right", fontsize=8)

        # Add explanation text
        fig.text(
            0.5,
            0.02,
            "Each pixel is classified into one of N classes (Background, Clean Ice, Debris)\n"
            + "Unlike classification (single label) or detection (bounding boxes), segmentation assigns labels to every pixel",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        self.save_figure(fig, "semantic_segmentation", "concepts")
        plt.close(fig)

    def create_physics_lstm_architecture(self):
        """Create Physics-LSTM architecture diagram based on actual implementation."""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Scale factor: height = neurons * scale
        neuron_scale = 0.0015
        box_width = 0.035
        special_width = 0.10

        # Y positions for two branches
        lstm_y = 0.68
        pinns_y = 0.32

        # Input layer
        input_x = 0.08
        input_y = 0.5
        input_neurons = 7
        input_height = input_neurons * neuron_scale

        self.draw_layer_box(
            ax, input_x, input_y, box_width, input_height, color=COLORS["input"]
        )
        ax.text(
            input_x,
            input_y + input_height / 2 + 0.015,
            "7",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            input_x,
            input_y - input_height / 2 - 0.01,
            "Input",
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.text(
            input_x,
            input_y - input_height / 2 - 0.025,
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
            "ψ, p",
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
            pinns_y + 0.015,
            r"$u = \frac{\partial\psi}{\partial y}$",
            ha="center",
            va="center",
            fontsize=6.5,
            style="italic",
            color="white",
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y,
            r"$v = -\frac{\partial\psi}{\partial x}$",
            ha="center",
            va="center",
            fontsize=6.5,
            style="italic",
            color="white",
            weight="bold",
        )
        ax.text(
            x_pos,
            pinns_y - 0.015,
            r"N-S: $f_u, f_v \rightarrow 0$",
            ha="center",
            va="center",
            fontsize=6.5,
            style="italic",
            color="white",
            weight="bold",
        )

        self.draw_arrow(
            ax,
            x_pos - 0.08 + box_width / 2,
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
        combine_x = 0.68
        combine_y = 0.5
        combine_height = 0.12

        # Arrows from both branches
        self.draw_arrow(
            ax,
            lstm_final_x + box_width / 2,
            lstm_y,
            combine_x - special_width / 2 - 0.005,
            combine_y + 0.02,
            color="darkgreen",
            linewidth=2.5,
        )
        self.draw_arrow(
            ax,
            pinns_final_x + box_width / 2,
            pinns_y,
            combine_x - special_width / 2 - 0.005,
            combine_y - 0.02,
            color="darkred",
            linewidth=2.5,
        )

        # Combination box
        self.draw_layer_box(
            ax,
            combine_x,
            combine_y,
            special_width,
            combine_height,
            color=COLORS["combine"],
        )
        ax.text(
            combine_x,
            combine_y + combine_height / 2 + 0.015,
            "Weighted Combination",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

        # Combination formula
        ax.text(
            combine_x,
            combine_y + 0.015,
            r"$w_L = \sigma(w_{lstm})$",
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            weight="bold",
        )
        ax.text(
            combine_x,
            combine_y - 0.015,
            r"$out = w_L \cdot LSTM + (1-w_L) \cdot PINNs$",
            ha="center",
            va="center",
            fontsize=6.5,
            style="italic",
            weight="bold",
        )

        # Final output
        final_x = 0.82
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
        self.draw_arrow(
            ax,
            combine_x + special_width / 2,
            combine_y,
            final_x - box_width / 2 - 0.005,
            final_y,
            linewidth=3,
        )

        # Title
        self.add_title(ax, "Physics-Informed LSTM Architecture")

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
            pinns_y - 0.10,
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
        ax.text(
            0.02,
            0.04,
            r"Learnable: $\lambda_2$ (viscosity), $w_{lstm}$ (weight)",
            ha="left",
            va="bottom",
            fontsize=7,
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lightyellow",
                alpha=0.9,
                edgecolor="gray",
                linewidth=1,
            ),
        )

        self.save_figure(fig, "physics_lstm_architecture", "physics_lstm")
        plt.close(fig)

    def create_temporal_sequence(self):
        """Create temporal sequence processing visualization."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Time steps
        time_steps = 3
        sequence_length = 0.6
        step_width = sequence_length / time_steps

        # Input sequence
        for t in range(time_steps):
            x_pos = 0.1 + t * step_width
            y_pos = 0.7

            # Input features
            features = ["X,Y,T", "Vu,Vv", "P,W.VF"]
            for i, feature in enumerate(features):
                feat_y = y_pos + (i - 1) * 0.08
                self.draw_layer_box(
                    ax, x_pos, feat_y, 0.04, 0.04, color=COLORS["input"]
                )
                ax.text(
                    x_pos,
                    feat_y,
                    feature,
                    ha="center",
                    va="center",
                    fontsize=6,
                    weight="bold",
                )

            # Time label
            ax.text(
                x_pos,
                y_pos + 0.2,
                f"t-{time_steps - 1 - t}",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            )

        # Bidirectional LSTM processing
        lstm_x = 0.4
        lstm_y = 0.7

        self.draw_layer_box(ax, lstm_x, lstm_y, 0.08, 0.15, color=COLORS["lstm"])
        ax.text(
            lstm_x,
            lstm_y + 0.1,
            "Bi-LSTM",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            lstm_x,
            lstm_y,
            "Forward\n← →",
            ha="center",
            va="center",
            fontsize=7,
            color="darkgreen",
        )
        ax.text(
            lstm_x,
            lstm_y - 0.05,
            "Backward\n→ ←",
            ha="center",
            va="center",
            fontsize=7,
            color="darkred",
        )

        # Arrows from sequence to LSTM
        for t in range(time_steps):
            x_pos = 0.1 + t * step_width
            self.draw_arrow(
                ax, x_pos + 0.02, lstm_y, lstm_x - 0.04, lstm_y, linewidth=1.5
            )

        # Hidden state output
        hidden_x = 0.6
        hidden_y = 0.7

        self.draw_layer_box(ax, hidden_x, hidden_y, 0.06, 0.08, color=COLORS["dense"])
        ax.text(
            hidden_x,
            hidden_y + 0.06,
            "Hidden\nState",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
        ax.text(
            hidden_x,
            hidden_y,
            "h_t",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )

        self.draw_arrow(
            ax, lstm_x + 0.04, lstm_y, hidden_x - 0.03, hidden_y, linewidth=2
        )

        # Context window explanation
        context_y = 0.4
        ax.text(
            0.5,
            context_y,
            "Sliding Window Approach",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        # Show multiple windows
        window_examples = [
            (0.15, context_y - 0.15, "Window 1"),
            (0.35, context_y - 0.15, "Window 2"),
            (0.55, context_y - 0.15, "Window 3"),
        ]

        for x, y, label in window_examples:
            # Draw window
            for t in range(time_steps):
                wx = x + t * 0.08
                rect = Rectangle(
                    (wx, y - 0.03),
                    0.06,
                    0.06,
                    facecolor=COLORS["input"],
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.6,
                )
                ax.add_patch(rect)

            ax.text(x + 0.08, y - 0.08, label, ha="center", va="top", fontsize=8)

            # Arrow to next window
            if x < 0.55:
                self.draw_arrow(ax, x + 0.24, y, x + 0.16, y, color="gray", style="--")

        # Mathematical notation
        ax.text(
            0.5,
            0.15,
            r"Input: $[x_{t-2}, x_{t-1}, x_t] \rightarrow h_t$",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        # Title
        self.add_title(ax, "Temporal Sequence Processing in Physics-LSTM")

        # Add annotations
        ax.text(
            0.02,
            0.95,
            "Bidirectional: Captures past & future context",
            ha="left",
            va="top",
            fontsize=8,
            style="italic",
            color=COLORS["lstm"],
        )
        ax.text(
            0.02,
            0.90,
            "Window size: 3 timesteps",
            ha="left",
            va="top",
            fontsize=8,
            style="italic",
        )

        self.save_figure(fig, "temporal_sequence", "physics_lstm")
        plt.close(fig)

    def create_architecture_comparison(self):
        """Create side-by-side comparison of U-Net vs Physics-LSTM."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            "Architecture Comparison: U-Net vs Physics-LSTM", fontsize=16, weight="bold"
        )

        # U-Net side
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")

        # U-Net components
        unet_components = [
            (0.15, 0.7, "Input\nImage", COLORS["input"], 0.08, 0.06),
            (0.35, 0.7, "Encoder\n(Contraction)", COLORS["conv"], 0.12, 0.08),
            (0.55, 0.7, "Decoder\n(Expansion)", COLORS["conv"], 0.12, 0.08),
            (0.75, 0.7, "Output\nMask", COLORS["output"], 0.08, 0.06),
        ]

        for x, y, label, color, width, height in unet_components:
            self.draw_layer_box(ax1, x, y, width, height, color=color)
            ax1.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")

        # U-Net arrows
        for i in range(len(unet_components) - 1):
            x1, y1, _, _, w1, _ = unet_components[i]
            x2, _, _, _, _, _ = unet_components[i + 1]
            self.draw_arrow(ax1, x1 + w1 / 2, y1, x2 - w1 / 2, y1, linewidth=2)

        # U-Net characteristics
        ax1.text(
            0.5,
            0.4,
            "Characteristics:",
            ha="center",
            va="top",
            fontsize=11,
            weight="bold",
        )
        ax1.text(
            0.5,
            0.35,
            "• Spatial convolutional layers",
            ha="center",
            va="top",
            fontsize=9,
        )
        ax1.text(0.5, 0.30, "• Skip connections", ha="center", va="top", fontsize=9)
        ax1.text(0.5, 0.25, "• Static image input", ha="center", va="top", fontsize=9)
        ax1.text(
            0.5, 0.20, "• Pixel-wise segmentation", ha="center", va="top", fontsize=9
        )
        ax1.text(
            0.5,
            0.15,
            "• Application: Image segmentation",
            ha="center",
            va="top",
            fontsize=9,
        )

        ax1.text(
            0.5,
            0.05,
            "U-Net Architecture",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        # Physics-LSTM side
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        # Physics-LSTM components
        lstm_components = [
            (0.15, 0.8, "Input\nSequence", COLORS["input"], 0.08, 0.06),
            (0.35, 0.8, "LSTM\nBranch", COLORS["lstm"], 0.10, 0.06),
            (0.35, 0.6, "PINNs\nBranch", COLORS["physics"], 0.10, 0.06),
            (0.55, 0.7, "Weighted\nCombine", COLORS["combine"], 0.10, 0.06),
            (0.75, 0.7, "Output\nVelocity", COLORS["output"], 0.08, 0.06),
        ]

        for x, y, label, color, width, height in lstm_components:
            self.draw_layer_box(ax2, x, y, width, height, color=color)
            ax2.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")

        # Physics-LSTM arrows
        # Input to both branches
        input_x, input_y, _, _, input_w, _ = lstm_components[0]
        lstm_x, lstm_y, _, _, _, _ = lstm_components[1]
        pinns_x, pinns_y, _, _, _, _ = lstm_components[2]

        self.draw_arrow(
            ax2,
            input_x + input_w / 2,
            input_y,
            lstm_x - 0.05,
            lstm_y,
            linewidth=2,
            color="darkgreen",
        )
        self.draw_arrow(
            ax2,
            input_x + input_w / 2,
            input_y,
            pinns_x - 0.05,
            pinns_y,
            linewidth=2,
            color="darkred",
        )

        # Branches to combination
        self.draw_arrow(
            ax2, lstm_x + 0.05, lstm_y, 0.5, 0.7, linewidth=2, color="darkgreen"
        )
        self.draw_arrow(
            ax2, pinns_x + 0.05, pinns_y, 0.5, 0.7, linewidth=2, color="darkred"
        )

        # Combination to output
        self.draw_arrow(ax2, 0.6, 0.7, 0.71, 0.7, linewidth=2)

        # Physics-LSTM characteristics
        ax2.text(
            0.5,
            0.4,
            "Characteristics:",
            ha="center",
            va="top",
            fontsize=11,
            weight="bold",
        )
        ax2.text(
            0.5,
            0.35,
            "• Temporal sequence processing",
            ha="center",
            va="top",
            fontsize=9,
        )
        ax2.text(
            0.5,
            0.30,
            "• Physics-informed constraints",
            ha="center",
            va="top",
            fontsize=9,
        )
        ax2.text(
            0.5, 0.25, "• Dual-branch architecture", ha="center", va="top", fontsize=9
        )
        ax2.text(0.5, 0.20, "• Spatiotemporal input", ha="center", va="top", fontsize=9)
        ax2.text(
            0.5,
            0.15,
            "• Application: Dynamics simulation",
            ha="center",
            va="top",
            fontsize=9,
        )

        ax2.text(
            0.5,
            0.05,
            "Physics-LSTM Architecture",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )

        # Add comparison table at bottom
        comparison_text = (
            "Input: Static Image vs Temporal Sequence\n"
            "Processing: Spatial Convolution vs Temporal Recurrence\n"
            "Constraints: Data-driven only vs Physics-informed\n"
            "Output: Segmentation Mask vs Velocity Field"
        )

        fig.text(
            0.5,
            0.02,
            comparison_text,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        self.save_figure(fig, "architecture_comparison", "concepts")
        plt.close(fig)


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("NETWORK VISUALIZATION SCRIPT")
    print("=" * 60)

    visualizer = NetworkVisualizer()

    print("\n1. Creating Current Architecture Visualizations...")
    try:
        visualizer.create_unet_architecture()
        print("   ✓ U-Net Architecture")
    except Exception as e:
        print(f"   ✗ U-Net Architecture: {e}")

    try:
        visualizer.create_physics_features()
        print("   ✓ Physics Features")
    except Exception as e:
        print(f"   ✗ Physics Features: {e}")

    try:
        visualizer.create_training_pipeline()
        print("   ✓ Training Pipeline")
    except Exception as e:
        print(f"   ✗ Training Pipeline: {e}")

    print("\n2. Creating Background Network Visualizations...")
    try:
        visualizer.create_simple_ann()
        print("   ✓ Simple ANN")
    except Exception as e:
        print(f"   ✗ Simple ANN: {e}")

    try:
        visualizer.create_standard_unet()
        print("   ✓ Standard U-Net")
    except Exception as e:
        print(f"   ✗ Standard U-Net: {e}")

    try:
        visualizer.create_lstm_cell()
        print("   ✓ LSTM Cell")
    except Exception as e:
        print(f"   ✗ LSTM Cell: {e}")

    try:
        visualizer.create_basic_cnn()
        print("   ✓ Basic CNN")
    except Exception as e:
        print(f"   ✗ Basic CNN: {e}")

    print("\n3. Creating Physics-LSTM Visualizations...")
    try:
        visualizer.create_physics_lstm_architecture()
        print("   ✓ Physics-LSTM Architecture")
    except Exception as e:
        print(f"   ✗ Physics-LSTM Architecture: {e}")

    try:
        visualizer.create_temporal_sequence()
        print("   ✓ Temporal Sequence")
    except Exception as e:
        print(f"   ✗ Temporal Sequence: {e}")

    print("\n4. Creating Concept Visualizations...")
    try:
        visualizer.create_iou_sets()
        print("   ✓ IoU as Sets")
    except Exception as e:
        print(f"   ✗ IoU as Sets: {e}")

    try:
        visualizer.create_iou_shapes()
        print("   ✓ IoU with Shapes")
    except Exception as e:
        print(f"   ✗ IoU with Shapes: {e}")

    try:
        visualizer.create_semantic_segmentation()
        print("   ✓ Semantic Segmentation")
    except Exception as e:
        print(f"   ✗ Semantic Segmentation: {e}")

    try:
        visualizer.create_architecture_comparison()
        print("   ✓ Architecture Comparison")
    except Exception as e:
        print(f"   ✗ Architecture Comparison: {e}")

    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
