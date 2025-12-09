#!/usr/bin/env python3
"""
Professional Physics-LSTM Visualization using PlotNeuralNet

This script demonstrates how to use PlotNeuralNet for creating
publication-quality Physics-Informed LSTM architecture diagrams.

Installation:
    pip install plotneuralnet

Usage:
    python physics_lstm_plotneuralnet.py
"""

import sys
from pathlib import Path

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from plotneuralnet.pycore import tikzeng
    from plotneuralnet.pycore.blocks import *

    PLOTNEURALNET_AVAILABLE = True
except ImportError:
    print("PlotNeuralNet not found. Install with: pip install plotneuralnet")
    PLOTNEURALNET_AVAILABLE = False


def create_physics_lstm_plotneuralnet():
    """Create professional Physics-LSTM diagram using PlotNeuralNet."""
    if not PLOTNEURALNET_AVAILABLE:
        return None

    # Define Physics-LSTM architecture using PlotNeuralNet
    arch = [
        tikzeng.to_head(".."),
        tikzeng.to_cor(),
        # Input layer
        tikzeng.to_input("Input", width=2, height=1.5, depth=1),
        # LSTM Branch (top)
        tikzeng.to_conv(name="lstm1", filters=32, kernel=(3, 3), depth=1),
        tikzeng.to_pool(name="pool1", size=(2, 2), depth=1),
        tikzeng.to_conv(name="lstm2", filters=32, kernel=(3, 3), depth=1),
        tikzeng.to_dense(name="lstm_dense", units=32, depth=1),
        # PINNs Branch (bottom)
        tikzeng.to_conv(name="pinns1", filters=32, kernel=(3, 3), depth=1),
        tikzeng.to_conv(name="pinns2", filters=64, kernel=(3, 3), depth=1),
        # Physics constraints
        tikzeng.to_custom(
            name="physics",
            width=3,
            height=2,
            text=r"$\frac{\partial\psi}{\partial y},\;-\frac{\partial\psi}{\partial x}$",
        ),
        # Weighted combination
        tikzeng.to_join(name="combine", width=2, height=1.5),
        # Output
        tikzeng.to_softmax(name="output", units=2, depth=1),
        tikzeng.to_end(),
    ]

    return arch


def generate_latex_code():
    """Generate complete LaTeX document for Physics-LSTM."""
    if not PLOTNEURALNET_AVAILABLE:
        return None

    # Architecture definition
    arch = create_physics_lstm_plotneuralnet()

    # Generate LaTeX code
    latex_code = tikzeng.to_generate(arch, "physics_lstm_plotneuralnet.tex")

    return latex_code


def create_comparison_diagram():
    """Create U-Net vs Physics-LSTM comparison using PlotNeuralNet."""
    if not PLOTNEURALNET_AVAILABLE:
        return None

    # Side-by-side comparison
    arch = [
        tikzeng.to_head(".."),
        tikzeng.to_cor(),
        # U-Net side
        tikzeng.to_input("U-Net Input", width=2, height=1.5, depth=1),
        tikzeng.to_conv(name="unet_enc1", filters=64, kernel=(3, 3), depth=1),
        tikzeng.to_pool(name="unet_pool1", size=(2, 2), depth=1),
        tikzeng.to_conv(name="unet_enc2", filters=128, kernel=(3, 3), depth=1),
        tikzeng.to_pool(name="unet_pool2", size=(2, 2), depth=1),
        tikzeng.to_conv(name="unet_bottleneck", filters=256, kernel=(3, 3), depth=1),
        tikzeng.to_conv(name="unet_dec1", filters=128, kernel=(3, 3), depth=1),
        tikzeng.to_conv(name="unet_dec2", filters=64, kernel=(3, 3), depth=1),
        tikzeng.to_softmax(name="unet_output", units=3, depth=1),
        # Physics-LSTM side
        tikzeng.to_input("LSTM Input", width=2, height=1.5, depth=1),
        tikzeng.to_conv(name="lstm1", filters=32, kernel=(3, 3), depth=1),
        tikzeng.to_lstm(name="lstm2", units=32, depth=1),
        tikzeng.to_dense(name="lstm_dense", units=32, depth=1),
        tikzeng.to_softmax(name="lstm_output", units=2, depth=1),
        tikzeng.to_end(),
    ]

    return tikzeng.to_generate(arch, "unet_vs_lstm_comparison.tex")


def main():
    """Main function to demonstrate PlotNeuralNet usage."""
    print("=" * 60)
    print("PROFESSIONAL PHYSICS-LSTM VISUALIZATION WITH PLOTNEURALNET")
    print("=" * 60)

    if not PLOTNEURALNET_AVAILABLE:
        print("❌ PlotNeuralNet not available. Install with:")
        print("   pip install plotneuralnet")
        return

    print("✅ PlotNeuralNet available - generating professional diagrams...")

    # Generate Physics-LSTM architecture
    try:
        latex_code = generate_latex_code()
        if latex_code:
            with open("physics_lstm_plotneuralnet.tex", "w") as f:
                f.write(latex_code)
            print("✅ Generated: physics_lstm_plotneuralnet.tex")

            # Compilation instructions
            print("\n📝 To compile the diagram:")
            print("   pdflatex physics_lstm_plotneuralnet.tex")
            print("   # This creates physics_lstm_plotneuralnet.pdf")

    except Exception as e:
        print(f"❌ Error generating Physics-LSTM diagram: {e}")

    # Generate comparison diagram
    try:
        comparison_code = create_comparison_diagram()
        if comparison_code:
            with open("unet_vs_lstm_comparison.tex", "w") as f:
                f.write(comparison_code)
            print("✅ Generated: unet_vs_lstm_comparison.tex")

            print("\n📝 To compile the comparison:")
            print("   pdflatex unet_vs_lstm_comparison.tex")
            print("   # This creates unet_vs_lstm_comparison.pdf")

    except Exception as e:
        print(f"❌ Error generating comparison diagram: {e}")

    print("\n" + "=" * 60)
    print("PLOTNEURALNET ADVANTAGES:")
    print("🎯 Publication-ready quality")
    print("🧮 Mathematical notation support")
    print("🔧 Easy customization")
    print("📚 Academic standard")
    print("🌐 Active development")
    print("=" * 60)


if __name__ == "__main__":
    main()
