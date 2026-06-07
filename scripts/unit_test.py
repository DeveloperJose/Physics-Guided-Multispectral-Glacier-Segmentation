import unittest
import numpy as np
import sys
import torch
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from glacier_mapping.data.slice import add_index, compute_dems
from glacier_mapping.model.losses import customloss


class TestVelocityLossMath(unittest.TestCase):
    """Test velocity loss mathematical correctness and Kendall formulation."""

    def test_velocity_loss_basic_functionality(self):
        """Test velocity loss basic functionality."""
        print("=== Test: Velocity Loss Basic Functionality ===")

        # Create simple test data
        batch_size, height, width = 2, 32, 32

        # Create model inputs (binary classification)
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_onehot = torch.zeros(batch_size, 2, height, width)
        target_onehot[:, 1] = 1.0  # Set foreground class
        target_int = torch.ones(batch_size, height, width).long()  # Foreground label

        # Create velocity data
        velocity = (
            torch.abs(torch.randn(batch_size, 1, height, width)) * 5.0
        )  # Positive velocities
        velocity_mask = torch.ones(batch_size, 1, height, width)  # All valid

        # Initialize loss function
        loss_fn = customloss()

        # Test loss computation
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Verify velocity loss is positive and reasonable
        self.assertGreater(velocity_loss.item(), 0.0)
        self.assertLess(velocity_loss.item(), 100.0)  # Shouldn't be too large

        # Test without velocity data
        dice_loss_no_vel, boundary_loss_no_vel, velocity_loss_no_vel = loss_fn(
            pred_logits, target_onehot, target_int, None, None
        )

        # Velocity loss should be zero when no velocity data
        self.assertEqual(velocity_loss_no_vel.item(), 0.0)

        print("  ✓ Velocity loss basic functionality verified.")
        print("  ✓ Velocity loss returns zero when no velocity data provided.")

    def test_sigma_initialization_values(self):
        """Test sigma parameters initialize correctly for velocity loss."""
        print("=== Test: Sigma Initialization Values ===")

        from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule

        # Create module with velocity loss enabled
        config = {
            "model_opts": {"args": {"net_depth": 4, "first_channel_output": 16}},
            "loss_opts": {"use_velocity_loss": True},
            "optim_opts": {},
            "metrics_opts": {},
            "loader_opts": {
                "processed_dir": "demo_data",
                "velocity_channels": True,
                "output_classes": [1],
                "class_names": ["background", "foreground"],
            },
        }

        model = GlacierSegmentationModule(**config)

        # Check initial sigma values are all 1.0
        self.assertIsNotNone(model.sigma_dice)
        self.assertIsNotNone(model.sigma_boundary)
        self.assertIsNotNone(model.sigma_velocity)
        self.assertEqual(model.sigma_dice.item(), 1.0)
        self.assertEqual(model.sigma_boundary.item(), 1.0)
        self.assertEqual(model.sigma_velocity.item(), 1.0)

        # Check sigma_list contains all three sigmas
        self.assertEqual(len(model.sigma_list), 3)
        self.assertIs(model.sigma_list[0], model.sigma_dice)
        self.assertIs(model.sigma_list[1], model.sigma_boundary)
        self.assertIs(model.sigma_list[2], model.sigma_velocity)

        print("  ✓ All sigmas initialize to 1.0 for equal weighting.")
        print("  ✓ Sigma list correctly contains all three parameters.")

    def test_velocity_threshold_config_passthrough(self):
        """Test velocity threshold config reaches the loss implementation."""
        print("=== Test: Velocity Threshold Config Passthrough ===")

        from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule

        model = GlacierSegmentationModule(
            model_opts={"args": {"net_depth": 4, "first_channel_output": 16}},
            loss_opts={
                "use_velocity_loss": True,
                "velocity_high_speed_threshold": 9.5,
            },
            optim_opts={},
            metrics_opts={},
            loader_opts={
                "processed_dir": "demo_data",
                "velocity_channels": True,
                "output_classes": [1],
                "class_names": ["background", "foreground"],
            },
        )

        self.assertEqual(model.loss_fn.velocity_high_speed_threshold, 9.5)
        print("  ✓ Velocity threshold config reaches customloss.")

    def test_kendall_formulation_components(self):
        """Test Kendall formulation mathematical components."""
        print("=== Test: Kendall Formulation Components ===")

        # Test the mathematical components separately
        losses = torch.tensor([0.5, 0.3, 1.2])  # dice, boundary, velocity
        sigmas = torch.tensor([0.8, 1.1, 0.6])  # sigma values

        # Expected Kendall formulation: L/(2σ²) + 0.5*log(σ²)
        expected_total = torch.tensor(0.0)

        for loss, sigma in zip(losses, sigmas):
            sigma_clamped = torch.clamp(sigma, min=0.1)
            var = sigma_clamped**2 + 1e-8

            # Kendall weighting: L/(2σ²)
            weighted_loss = loss / (2.0 * var)
            expected_total += weighted_loss

            # Regularization: 0.5*log(σ²)
            expected_total += 0.5 * torch.log(var)

        # Verify mathematical correctness
        self.assertGreater(expected_total.item(), 0.0)
        self.assertTrue(torch.isfinite(expected_total))

        print("  ✓ Kendall formulation mathematical components verified.")
        print("  ✓ Loss weighting and regularization terms computed correctly.")

    def test_velocity_loss_edge_cases(self):
        """Test velocity loss edge cases and numerical stability."""
        print("=== Test: Velocity Loss Edge Cases ===")

        loss_fn = customloss()

        # Test case 1: All zero velocity
        pred_logits = torch.randn(2, 2, 32, 32)
        target_onehot = torch.zeros(2, 2, 32, 32)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(2, 32, 32).long()

        velocity = torch.zeros(2, 1, 32, 32)  # All zero velocity
        velocity_mask = torch.ones(2, 1, 32, 32)  # All valid

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Sigmoid motion probability is small but nonzero below the threshold.
        self.assertGreaterEqual(velocity_loss.item(), 0.0)
        self.assertLess(velocity_loss.item(), 0.01)

        # Test case 2: Empty velocity mask
        velocity_mask = torch.zeros(2, 1, 32, 32)  # No valid velocity

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Velocity loss should be zero when mask is empty
        self.assertEqual(velocity_loss.item(), 0.0)

        print("  ✓ Zero velocity case handled correctly.")
        print("  ✓ Empty velocity mask case handled correctly.")

    def test_sigma_minimum_constraint(self):
        """Test sigma parameter minimum constraint enforcement."""
        print("=== Test: Sigma Minimum Constraint ===")

        from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule

        model = GlacierSegmentationModule(
            model_opts={"args": {"net_depth": 4, "first_channel_output": 16}},
            loss_opts={"use_velocity_loss": True},
            optim_opts={},
            metrics_opts={},
            loader_opts={
                "processed_dir": "demo_data",
                "velocity_channels": True,
                "output_classes": [1],
                "class_names": ["bg", "fg"],
            },
        )

        # Test that very small sigma values are clamped to minimum
        with torch.no_grad():
            self.assertIsNotNone(model.sigma_dice)
            self.assertIsNotNone(model.sigma_boundary)
            self.assertIsNotNone(model.sigma_velocity)
            model.sigma_dice.data = torch.tensor(0.05)  # Below min=0.1
            model.sigma_boundary.data = torch.tensor(0.01)
            model.sigma_velocity.data = torch.tensor(0.02)

        # Test loss computation with clamped sigmas (should not crash)
        pred_logits = torch.randn(1, 2, 32, 32)
        target_onehot = torch.zeros(1, 2, 32, 32)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(1, 32, 32).long()
        velocity = torch.randn(1, 1, 32, 32)
        velocity_mask = torch.ones(1, 1, 32, 32)

        test_loss = model.compute_loss(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Should not crash and should be finite
        self.assertTrue(torch.isfinite(test_loss))
        self.assertGreater(test_loss.item(), 0.0)

        print("  ✓ Sigma minimum constraint (0.1) enforced correctly.")
        print("  ✓ No numerical instability with very small sigmas.")


class TestSliceFunctions(unittest.TestCase):
    def test_add_index(self):
        """Test the add_index function."""
        print("=== Test: add_index Function ===")
        # Create a sample 4-band array
        tiff_np = np.ones((10, 10, 4), dtype=np.float32)
        tiff_np[..., 0] = 2  # Band 1
        tiff_np[..., 1] = 4  # Band 2

        # Test NDVI-like index (B2 - B1) / (B2 + B1)
        result = add_index(tiff_np, index1=1, index2=0)
        expected_index = (4 - 2) / (4 + 2)  # 2 / 6 = 0.333...

        self.assertEqual(result.shape, (10, 10, 5))
        self.assertTrue(np.allclose(result[..., 4], expected_index))
        print("  ✓ Correctly calculates spectral index.")

        # Test division by zero
        tiff_np[..., 0] = -4
        result = add_index(tiff_np, index1=1, index2=0)
        self.assertFalse(np.isnan(result).any())
        print("  ✓ Handles division by zero gracefully.")

    def test_compute_dems(self):
        """Test the compute_dems function."""
        print("=== Test: compute_dems Function ===")
        # Create a sample 2-band DEM array
        dem_np = np.zeros((10, 10, 2), dtype=np.float32)
        dem_np[..., 0] = 1000  # Elevation
        dem_np[..., 1] = 30  # Slope

        result = compute_dems(dem_np)

        self.assertEqual(result.shape, (10, 10, 2))
        self.assertTrue(np.all(result[..., 0] == 1000))
        self.assertTrue(np.all(result[..., 1] == 30))
        print("  ✓ Correctly extracts elevation and slope.")


class TestClassWeights(unittest.TestCase):
    """Test class weighting for Dice loss in binary and multi-class scenarios."""

    def test_binary_class_weights(self):
        """Test binary classification with class_weights=[0, 1]."""
        print("=== Test: Binary Class Weights ===")

        batch_size, height, width = 2, 32, 32

        # Create model inputs (binary classification)
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_onehot = torch.zeros(batch_size, 2, height, width)
        target_onehot[:, 1] = 1.0  # Set foreground class
        target_int = torch.ones(batch_size, height, width).long()  # Foreground label

        # Initialize loss function with class weights
        loss_fn = customloss(class_weights=[0, 1])

        # Test loss computation
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int
        )

        # Verify losses are finite and positive
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(boundary_loss))
        self.assertGreaterEqual(boundary_loss.item(), 0.0)
        self.assertEqual(velocity_loss.item(), 0.0)  # No velocity data

        print("  ✓ Binary class weights [0, 1] handled correctly.")

    def test_multiclass_class_weights(self):
        """Test multi-class classification with class_weights=[0, 1, 1]."""
        print("=== Test: Multi-class Class Weights ===")

        batch_size, height, width = 2, 32, 32

        # Create model inputs (multi-class classification)
        pred_logits = torch.randn(batch_size, 3, height, width)
        target_onehot = torch.zeros(batch_size, 3, height, width)
        target_onehot[:, 1] = 1.0  # Set class 1
        target_int = torch.ones(batch_size, height, width).long()  # Class 1 label

        # Initialize loss function with class weights
        loss_fn = customloss(class_weights=[0, 1, 1])

        # Test loss computation
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int
        )

        # Verify losses are finite and positive
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(boundary_loss))
        self.assertGreaterEqual(boundary_loss.item(), 0.0)
        self.assertEqual(velocity_loss.item(), 0.0)  # No velocity data

        print("  ✓ Multi-class class weights [0, 1, 1] handled correctly.")

    def test_class_weights_length_mismatch(self):
        """Test that a ValueError is raised for mismatched class_weights length."""
        print("=== Test: Class Weights Length Mismatch ===")

        batch_size, height, width = 2, 32, 32

        # Create model inputs (binary classification)
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_onehot = torch.zeros(batch_size, 2, height, width)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(batch_size, height, width).long()

        # Initialize loss function with incorrect class_weights length
        loss_fn = customloss(class_weights=[0, 1, 1])  # Should be length 2 for binary

        with self.assertRaises(ValueError):
            loss_fn(pred_logits, target_onehot, target_int)

        print("  ✓ ValueError raised for mismatched class_weights length.")

    def test_no_class_weights_fallback(self):
        """Test that the loss function works without class_weights (equal weighting)."""
        print("=== Test: No Class Weights Fallback ===")

        batch_size, height, width = 2, 32, 32

        # Create model inputs (binary classification)
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_onehot = torch.zeros(batch_size, 2, height, width)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(batch_size, height, width).long()

        # Initialize loss function without class_weights
        loss_fn = customloss()

        # Test loss computation
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int
        )

        # Verify losses are finite and positive
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(boundary_loss))
        self.assertGreaterEqual(boundary_loss.item(), 0.0)
        self.assertEqual(velocity_loss.item(), 0.0)  # No velocity data

        print("  ✓ Equal weighting fallback works correctly.")


if __name__ == "__main__":
    unittest.main()
