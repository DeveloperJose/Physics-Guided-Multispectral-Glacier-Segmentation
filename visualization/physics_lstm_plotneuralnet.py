#!/usr/bin/env python3
"""
Physics-LSTM Network Visualization using PlotNeuralNet

This script creates a comprehensive visualization of the Physics-LSTM architecture
with dual-branch design: CNN branch for spatial features and LSTM branch for
temporal dynamics, integrated with physics-informed neural networks.

Author: Glacier Mapping Team
"""

import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "temp_plotneuralnet")
)

from PlotNeuralNet.PyCore.Blocks import Block2ConvPool, BlockRes
from PlotNeuralNet.PyCore.TikzGen import (
    ToBegin,
    ToConnection,
    ToConvConvRelu,
    ToFullyConnected,
    ToCor,
    ToEnd,
    ToGenerate,
    ToHead,
    ToInput,
    ToPool,
    ToSum,
)


def create_physics_lstm_architecture():
    """
    Create the Physics-LSTM architecture with dual-branch design.

    Returns
    -------
    list
        List of PlotNeuralNet components for the architecture
    """

    arch = [
        ToHead("."),
        ToCor(),
        ToBegin(),
        # Input Layer - Multi-temporal satellite imagery
        ToConvConvRelu(
            name="input_sequence",
            sFilter=256,
            nFilter=(3, 3),  # RGB input
            offset="(0,0,0)",
            to="(0,0,0)",
            width=(6, 6),
            height=40,
            depth=40,
            caption="Input Sequence\\$\\mathbf{X}_{t-T:t}$",
        ),
        # CNN Branch - First Conv Block
        ToConvConvRelu(
            name="cnn_conv1",
            sFilter=256,
            nFilter=(64, 64),
            offset="(1.5,0,0)",
            to="(input_sequence-east)",
            width=(3, 3),
            height=35,
            depth=35,
            caption="CNN Conv1\\$\\mathbf{F}_{\\theta}$",
        ),
        ToPool(
            name="cnn_pool1",
            offset="(0.8,0,0)",
            to="(cnn_conv1-east)",
            width=1.5,
            height=28,
            depth=28,
            opacity=0.6,
        ),
        # CNN Branch - Second Conv Block
        ToConvConvRelu(
            name="cnn_conv2",
            sFilter=128,
            nFilter=(128, 128),
            offset="(1.2,0,0)",
            to="(cnn_pool1-east)",
            width=(4, 4),
            height=24,
            depth=24,
            caption="CNN Conv2\\$\\mathbf{G}_{\\phi}$",
        ),
        ToPool(
            name="cnn_pool2",
            offset="(0.8,0,0)",
            to="(cnn_conv2-east)",
            width=1.5,
            height=18,
            depth=18,
            opacity=0.6,
        ),
        # CNN Branch - Third Conv Block (Deeper features)
        ToConvConvRelu(
            name="cnn_conv3",
            sFilter=64,
            nFilter=(256, 256),
            offset="(1.2,0,0)",
            to="(cnn_pool2-east)",
            width=(5, 5),
            height=14,
            depth=14,
            caption="CNN Conv3\\$\\mathbf{H}_{\\psi}$",
        ),
        # ===== LSTM BRANCH (Temporal Dynamics) =====
        # LSTM Branch - Input processing (positioned below CNN branch)
        ToConvConvRelu(
            name="lstm_input",
            sFilter=256,
            nFilter=(32, 32),
            offset="(0,-4,0)",
            to="(input_sequence-south)",
            width=(2, 2),
            height=20,
            depth=20,
            caption="LSTM Input\\$\\mathbf{x}_t$",
        ),
        # LSTM Cell 1
        ToConvConvRelu(
            name="lstm_cell1",
            sFilter=128,
            nFilter=(64, 64),
            offset="(2.5,0,0)",
            to="(lstm_input-east)",
            width=(3, 3),
            height=18,
            depth=18,
            caption="LSTM Cell 1\\$\\mathbf{h}_1$",
        ),
        # LSTM Cell 2
        ToConvConvRelu(
            name="lstm_cell2",
            sFilter=64,
            nFilter=(64, 64),
            offset="(2.5,0,0)",
            to="(lstm_cell1-east)",
            width=(3, 3),
            height=16,
            depth=16,
            caption="LSTM Cell 2\\$\\mathbf{h}_2$",
        ),
        # LSTM Cell 3
        ToConvConvRelu(
            name="lstm_cell3",
            sFilter=32,
            nFilter=(64, 64),
            offset="(2.5,0,0)",
            to="(lstm_cell2-east)",
            width=(3, 3),
            height=14,
            depth=14,
            caption="LSTM Cell 3\\$\\mathbf{h}_3$",
        ),
        # ===== PHYSICS-INFORMED INTEGRATION =====
        # Physics Constraint Layer (Navier-Stokes)
        ToConvConvRelu(
            name="physics_constraint",
            sFilter=32,
            nFilter=(128, 128),
            offset="(1.5,2,0)",
            to="(lstm_cell3-north)",
            width=(4, 4),
            height=12,
            depth=12,
            caption="Physics Layer\\$\\nabla \\cdot \\mathbf{v} = 0$\\$\\frac{\\partial \\mathbf{v}}{\\partial t} + (\\mathbf{v} \\cdot \\nabla)\\mathbf{v} = -\\nabla p + \\nu \\nabla^2 \\mathbf{v}$",
        ),
        # Feature Fusion Layer
        ToConvConvRelu(
            name="feature_fusion",
            sFilter=32,
            nFilter=(256, 256),
            offset="(1.5,1.5,0)",
            to="(physics_constraint-north)",
            width=(6, 6),
            height=10,
            depth=10,
            caption="Fusion Layer\\$\\mathbf{z} = [\\mathbf{f}_{cnn}, \\mathbf{h}_{lstm}, \\mathbf{p}_{physics}]$",
        ),
        # ===== OUTPUT BRANCHES =====
        # Segmentation Output Branch
        ToConvConvRelu(
            name="seg_head",
            sFilter=64,
            nFilter=(128, 128),
            offset="(2,0,0)",
            to="(feature_fusion-east)",
            width=(4, 4),
            height=8,
            depth=8,
            caption="Seg Head\\$\\mathbf{y}_{seg}$",
        ),
        # Velocity Output Branch
        ToConvConvRelu(
            name="vel_head",
            sFilter=64,
            nFilter=(64, 64),
            offset="(0,-2,0)",
            to="(seg_head-south)",
            width=(3, 3),
            height=6,
            depth=6,
            caption="Vel Head\\$\\mathbf{v}_{pred}$",
        ),
        # Final Segmentation Output
        ToConvConvRelu(
            name="seg_output",
            sFilter=128,
            nFilter=(3, 3),  # 3 classes: clean ice, debris ice, background
            offset="(2,0,0)",
            to="(seg_head-east)",
            width=(2, 2),
            height=6,
            depth=6,
            caption="Output\\$\\hat{\\mathbf{y}}$",
        ),
        # ===== CONNECTIONS =====
        # CNN branch connections
        ToConnection("input_sequence", "cnn_conv1"),
        ToConnection("cnn_conv1", "cnn_pool1"),
        ToConnection("cnn_pool1", "cnn_conv2"),
        ToConnection("cnn_conv2", "cnn_pool2"),
        ToConnection("cnn_pool2", "cnn_conv3"),
        # LSTM branch connections
        ToConnection("input_sequence", "lstm_input"),
        ToConnection("lstm_input", "lstm_cell1"),
        ToConnection("lstm_cell1", "lstm_cell2"),
        ToConnection("lstm_cell2", "lstm_cell3"),
        # Physics integration connections
        ToConnection("lstm_cell3", "physics_constraint"),
        ToConnection("physics_constraint", "feature_fusion"),
        ToConnection("cnn_conv3", "feature_fusion"),
        # Output connections
        ToConnection("feature_fusion", "seg_head"),
        ToConnection("seg_head", "vel_head"),
        ToConnection("seg_head", "seg_output"),
        ToEnd(),
    ]

    return arch


def main():
    """
    Main function to generate the Physics-LSTM visualization.
    """
    print("🧠 Generating Physics-LSTM Network Visualization...")

    # Create architecture
    arch = create_physics_lstm_architecture()

    # Generate LaTeX file
    output_name = "physics_lstm_plotneuralnet"
    ToGenerate(arch, f"{output_name}.tex")

    print(f"✅ Generated {output_name}.tex")
    print("📝 To compile: pdflatex physics_lstm_plotneuralnet.tex")
    print("🎨 Features included:")
    print("   • Dual-branch architecture (CNN + LSTM)")
    print("   • Physics-informed constraints (Navier-Stokes)")
    print("   • Multi-task learning (segmentation + velocity)")
    print("   • Mathematical notation in LaTeX")
    print("   • Professional academic styling")


if __name__ == "__main__":
    main()
