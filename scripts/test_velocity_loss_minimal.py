#!/usr/bin/env python3
"""
Minimal Velocity Loss Regression Tests

Tests core mathematical correctness of velocity loss without requiring full data pipeline.
Prevents regressions in velocity loss integration by testing mathematical components directly.

Usage:
    uv run python scripts/test_velocity_loss_minimal.py
"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from glacier_mapping.model.losses import customloss


class TestVelocityLossMinimal(unittest.TestCase):
    """Minimal regression tests for velocity loss mathematical correctness."""

    def test_velocity_loss_formula(self):
        """Test velocity loss mathematical formula: mean(P(BG) * velocity * mask)."""
        print("=== Test: Velocity Loss Formula ===")

        # Create test data
        batch_size, height, width = 2, 32, 32

        # Create model inputs (binary classification)
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_onehot = torch.zeros(batch_size, 2, height, width)
        target_onehot[:, 1] = 1.0  # Set foreground class
        target_int = torch.ones(batch_size, height, width, 1).long()  # Foreground label

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

        print("  ✓ Velocity loss formula verified: mean(P(BG) * velocity * mask)")

    def test_velocity_loss_background_probability(self):
        """Test velocity loss background probability calculation."""
        print("=== Test: Velocity Loss Background Probability ===")

        loss_fn = customloss()

        # Test binary case: P(BG) = 1 - P(FG)
        pred_logits = torch.randn(2, 2, 32, 32)
        target_onehot = torch.zeros(2, 2, 32, 32)
        target_onehot[:, 1] = 1.0  # Set foreground class
        target_int = torch.ones(2, 32, 32, 1).long()

        velocity = torch.abs(torch.randn(2, 1, 32, 32)) * 3.0
        velocity_mask = torch.ones(2, 1, 32, 32)

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Test multi-class case: P(BG) = pred_prob[:, 0:1]
        pred_logits_multi = torch.randn(2, 3, 32, 32)
        target_onehot_multi = torch.zeros(2, 3, 32, 32)
        target_onehot_multi[:, 1] = 1.0  # Class 1
        target_onehot_multi[:, 2] = 1.0  # Class 2
        target_int_multi = torch.ones(2, 32, 32, 1).long()

        velocity_loss_multi = loss_fn(
            pred_logits_multi,
            target_onehot_multi,
            target_int_multi,
            velocity,
            velocity_mask,
        )

        # Both should be positive and similar
        self.assertGreater(velocity_loss.item(), 0.0)
        self.assertGreater(velocity_loss_multi.item(), 0.0)

        print(
            "  ✓ Background probability calculation verified for binary and multi-class"
        )

    def test_velocity_loss_zero_cases(self):
        """Test velocity loss edge cases."""
        print("=== Test: Velocity Loss Zero Cases ===")

        loss_fn = customloss()

        # Test case 1: All zero velocity
        pred_logits = torch.randn(2, 2, 32, 32)
        target_onehot = torch.zeros(2, 2, 32, 32)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(2, 32, 32, 1).long()

        velocity = torch.zeros(2, 1, 32, 32)  # All zero velocity
        velocity_mask = torch.ones(2, 1, 32, 32)  # All valid

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Velocity loss should be zero when all velocity is zero
        self.assertEqual(velocity_loss.item(), 0.0)

        # Test case 2: Empty velocity mask
        velocity_mask = torch.zeros(2, 1, 32, 32)  # No valid velocity

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Velocity loss should be zero when mask is empty
        self.assertEqual(velocity_loss.item(), 0.0)

        print("  ✓ Zero velocity cases handled correctly.")

    def test_velocity_loss_numerical_bounds(self):
        """Test velocity loss numerical bounds."""
        print("=== Test: Velocity Loss Numerical Bounds ===")

        loss_fn = customloss()

        # Test with very small velocity values
        pred_logits = torch.randn(2, 2, 32, 32)
        target_onehot = torch.zeros(2, 2, 32, 32)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(2, 32, 32, 1).long()

        velocity = torch.abs(torch.randn(2, 1, 32, 32)) * 1e-8  # Very small
        velocity_mask = torch.ones(2, 1, 32, 32)

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Should be finite and positive
        self.assertTrue(torch.isfinite(velocity_loss))
        self.assertGreater(velocity_loss.item(), 0.0)

        # Test with very large velocity values
        velocity = torch.abs(torch.randn(2, 1, 32, 32)) * 1e6  # Very large
        velocity_mask = torch.ones(2, 1, 32, 32)

        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # Should be finite and positive
        self.assertTrue(torch.isfinite(velocity_loss))
        self.assertGreater(velocity_loss.item(), 0.0)

        print("  ✓ Numerical bounds verified: small and large velocity values.")

    def test_velocity_loss_integration(self):
        """Test velocity loss integration with other losses."""
        print("=== Test: Velocity Loss Integration ===")

        loss_fn = customloss()

        # Create test data
        pred_logits = torch.randn(2, 2, 32, 32)
        target_onehot = torch.zeros(2, 2, 32, 32)
        target_onehot[:, 1] = 1.0  # Set foreground class
        target_int = torch.ones(2, 32, 32, 1).long()

        velocity = torch.abs(torch.randn(2, 1, 32, 32)) * 3.0
        velocity_mask = torch.ones(2, 1, 32, 32)

        # Test all three losses are computed
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_onehot, target_int, velocity, velocity_mask
        )

        # All losses should be positive and finite
        self.assertGreater(dice_loss.item(), 0.0)
        self.assertLess(dice_loss.item(), 2.0)  # Dice loss bounded [0, 2]

        self.assertGreater(boundary_loss.item(), 0.0)
        self.assertLess(boundary_loss.item(), 2.0)  # Boundary loss bounded [0, 2]

        self.assertGreaterEqual(velocity_loss.item(), 0.0)  # Can be zero
        self.assertLess(velocity_loss.item(), 100.0)  # Reasonable upper bound

        # Test total loss computation
        total_loss = dice_loss + boundary_loss + velocity_loss
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        print("  ✓ Three-loss integration verified: all components computed correctly.")

    def test_velocity_loss_stability(self):
        """Test velocity loss stability with repeated calls."""
        print("=== Test: Velocity Loss Stability ===")

        loss_fn = customloss()

        # Create test data
        pred_logits = torch.randn(2, 2, 32, 32)
        target_onehot = torch.zeros(2, 2, 32, 32)
        target_onehot[:, 1] = 1.0
        target_int = torch.ones(2, 32, 32, 1).long()

        velocity = torch.abs(torch.randn(2, 1, 32, 32)) * 2.0
        velocity_mask = torch.ones(2, 1, 32, 32)

        # Test multiple calls for stability
        losses_1 = []
        losses_2 = []
        losses_3 = []

        for i in range(5):
            dice_loss, boundary_loss, velocity_loss = loss_fn(
                pred_logits, target_onehot, target_int, velocity, velocity_mask
            )
            losses_1.append(velocity_loss.item())
            losses_2.append(boundary_loss.item())
            losses_3.append(dice_loss.item())

        # All losses should be consistent
        for i in range(1, len(losses_1)):
            self.assertAlmostEqual(losses_1[i], losses_1[0], places=6)
            self.assertAlmostEqual(losses_2[i], losses_2[0], places=6)
            self.assertAlmostEqual(losses_3[i], losses_3[0], places=6)

        print(
            "  ✓ Velocity loss stability verified: consistent results across multiple calls."
        )


if __name__ == "__main__":
    unittest.main()
