#!/usr/bin/env python3
"""
Dice loss parity test: current permute()[ignore_mask] vs broadcast-mask rewrite.

Tests:
1. Loss value parity on same batch (binary + multi-class)
2. Gradient parity (backward pass)
3. Performance comparison (timing)
4. Edge cases (empty masks, all-valid masks, single pixel)

Run: uv run python scripts/test_dice_parity.py
"""
import time
import unittest
import torch
import torch.nn as nn

from glacier_mapping.model.losses import customloss, make_onehot


def dice_loss_broadcast(pred_prob, target_prob, ignore_mask_exp, ignore_mask_bool, smooth, class_weights_tensor):
    """Broadcast-mask Dice loss — avoids permute()[ignore_mask]."""
    n, c, h, w = pred_prob.shape
    
    # Valid mask: [N, 1, H, W]
    valid_mask = ignore_mask_exp  # already [N, 1, H, W]
    
    # Weighted elements
    weighted_pred = pred_prob * valid_mask
    weighted_target = target_prob * valid_mask
    weighted_prod = pred_prob * target_prob * valid_mask
    
    # Sum over (batch, height, width) -> [C]
    numerator = 2 * weighted_prod.sum(dim=(0, 2, 3)) + smooth
    denominator = weighted_pred.sum(dim=(0, 2, 3)) + weighted_target.sum(dim=(0, 2, 3)) + smooth
    
    dice_per_class = 1 - numerator / denominator
    
    if class_weights_tensor is not None:
        dice_loss = (dice_per_class * class_weights_tensor).sum()
    else:
        dice_loss = dice_per_class.mean()
    
    return dice_loss


def dice_loss_original(pred_prob, target_prob, ignore_mask_exp, ignore_mask_bool, smooth, class_weights_tensor):
    """Original permute()[ignore_mask] Dice loss."""
    n, c, h, w = pred_prob.shape
    
    pred_flat = pred_prob.permute(0, 2, 3, 1)[ignore_mask_bool]
    targ_flat = target_prob.permute(0, 2, 3, 1)[ignore_mask_bool]
    
    if pred_flat.numel() == 0:
        return pred_prob.new_zeros(())
    
    numerator = 2 * (pred_flat * targ_flat).sum(dim=0) + smooth
    denominator = pred_flat.sum(dim=0) + targ_flat.sum(dim=0) + smooth
    dice_per_class = 1 - numerator / denominator
    
    if class_weights_tensor is not None:
        dice_loss = (dice_per_class * class_weights_tensor).sum()
    else:
        dice_loss = dice_per_class.mean()
    
    return dice_loss


class TestDiceParity(unittest.TestCase):
    
    def setUp(self):
        """Create test tensors."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_binary_dice_loss_value_parity(self):
        """Binary Dice: broadcast-mask vs permute()[ignore_mask]."""
        B, C, H, W = 4, 2, 32, 32
        
        pred = torch.randn(B, C, H, W, device=self.device, requires_grad=True)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        # Add some ignore pixels
        target_int[0, 0, 0] = 255
        target_int[1, -1, -1] = 255
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        loss_orig = dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        loss_broadcast = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        self.assertAlmostEqual(loss_orig.item(), loss_broadcast.item(), places=6,
            msg=f"Binary Dice mismatch: orig={loss_orig.item():.8f}, broadcast={loss_broadcast.item():.8f}")
    
    def test_multiclass_dice_loss_value_parity(self):
        """Multi-class Dice: broadcast-mask vs permute()[ignore_mask]."""
        B, C, H, W = 4, 3, 32, 32
        
        pred = torch.randn(B, C, H, W, device=self.device, requires_grad=True)
        target_int = torch.randint(0, 3, (B, H, W), device=self.device, dtype=torch.uint8)
        # Add some ignore pixels
        target_int[0, 0, 0] = 255
        target_int[2, -1, -1] = 255
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [0, 1, 2], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        loss_orig = dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        loss_broadcast = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        self.assertAlmostEqual(loss_orig.item(), loss_broadcast.item(), places=6,
            msg=f"Multi-class Dice mismatch: orig={loss_orig.item():.8f}, broadcast={loss_broadcast.item():.8f}")
    
    def test_binary_dice_with_class_weights(self):
        """Binary Dice with class weights."""
        B, C, H, W = 4, 2, 32, 32
        
        pred = torch.randn(B, C, H, W, device=self.device)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        target_int[0, 0, 0] = 255
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        class_weights = torch.tensor([0.3, 0.7], device=self.device)
        
        loss_orig = dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, class_weights)
        loss_broadcast = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, class_weights)
        
        self.assertAlmostEqual(loss_orig.item(), loss_broadcast.item(), places=6,
            msg=f"Weighted Dice mismatch: orig={loss_orig.item():.8f}, broadcast={loss_broadcast.item():.8f}")
    
    def test_empty_mask(self):
        """All pixels ignored."""
        B, C, H, W = 2, 2, 16, 16
        
        pred = torch.randn(B, C, H, W, device=self.device)
        target_int = torch.full((B, H, W), 255, device=self.device, dtype=torch.uint8)
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        loss_orig = dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        loss_broadcast = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        self.assertEqual(loss_orig.item(), 0.0, "Original should return 0 for empty mask")
        self.assertEqual(loss_broadcast.item(), 0.0, "Broadcast should return 0 for empty mask")
    
    def test_all_valid_pixels(self):
        """No ignore pixels."""
        B, C, H, W = 4, 2, 32, 32
        
        pred = torch.randn(B, C, H, W, device=self.device)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = torch.ones((B, 1, H, W), device=self.device)
        ignore_mask_bool = torch.ones((B, H, W), device=self.device, dtype=torch.bool)
        
        loss_orig = dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        loss_broadcast = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        self.assertAlmostEqual(loss_orig.item(), loss_broadcast.item(), places=6)
    
    def test_gradient_parity(self):
        """Gradients should match."""
        B, C, H, W = 2, 2, 16, 16
        
        pred = torch.randn(B, C, H, W, device=self.device, requires_grad=True)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        target_int[0, 0, 0] = 255
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        # Original
        loss_orig = dice_loss_original(pred_prob.clone(), target.clone(), ignore_mask_exp.clone(), ignore_mask_bool.clone(), 1.0, None)
        grad_orig = torch.autograd.grad(loss_orig, pred, retain_graph=True, create_graph=True)[0]
        
        # Broadcast
        loss_broadcast = dice_loss_broadcast(pred_prob.clone(), target.clone(), ignore_mask_exp.clone(), ignore_mask_bool.clone(), 1.0, None)
        grad_broadcast = torch.autograd.grad(loss_broadcast, pred, retain_graph=True, create_graph=True)[0]
        
        torch.testing.assert_close(grad_orig, grad_broadcast, rtol=1e-5, atol=1e-5,
            msg="Gradients mismatch")
    
    def test_single_pixel(self):
        """Single valid pixel."""
        B, C, H, W = 2, 2, 4, 4
        
        pred = torch.randn(B, C, H, W, device=self.device)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        # Only one valid pixel
        target_int[:, 1:, :] = 255
        target_int[:, 0, 1:] = 255
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        loss_orig = dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        loss_broadcast = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        self.assertAlmostEqual(loss_orig.item(), loss_broadcast.item(), places=6)
    
    def test_performance_comparison(self):
        """Broadcast-mask should be faster."""
        B, C, H, W = 12, 2, 512, 512  # Typical batch size and resolution
        
        pred = torch.randn(B, C, H, W, device=self.device)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        # 20% ignore pixels (typical for glacier data)
        mask = torch.rand(B, H, W, device=self.device) < 0.2
        target_int[mask] = 255
        
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        # Warmup
        for _ in range(5):
            dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
            dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        # Benchmark original
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(100):
            dice_loss_original(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        orig_time = time.perf_counter() - start
        
        # Benchmark broadcast
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(100):
            dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        broadcast_time = time.perf_counter() - start
        
        print(f"\n=== Performance (100 iterations, BS=12, 512x512) ===")
        print(f"Original permute()[ignore_mask]: {orig_time*10:.3f}ms/batch")
        print(f"Broadcast-mask: {broadcast_time*10:.3f}ms/batch")
        print(f"Speedup: {orig_time/broadcast_time:.2f}x")
        
        # Broadcast should be at least as fast
        self.assertLessEqual(broadcast_time, orig_time * 1.2,
            f"Broadcast should be within 20% of original, got {orig_time/broadcast_time:.2f}x")
    
    def test_full_customloss_dice_component(self):
        """Test that the full customloss Dice component matches."""
        B, C, H, W = 4, 2, 32, 32
        
        pred = torch.randn(B, C, H, W, device=self.device, requires_grad=True)
        target_int = torch.randint(0, 2, (B, H, W), device=self.device, dtype=torch.uint8)
        target_int[0, 0, 0] = 255
        
        # Create loss function
        loss_fn = customloss(
            output_classes=[1],
            velocity_loss_weight=0.05,
            velocity_high_speed_threshold=3.16,
        )
        
        # Get Dice from customloss
        losses = loss_fn(pred, target_int, current_epoch=20)
        dice_from_fn = losses[0]
        
        # Compute expected Dice manually
        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, C, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        ignore_mask_bool = (target_int != 255)
        
        expected_dice = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, ignore_mask_bool, 1.0, None)
        
        self.assertAlmostEqual(dice_from_fn.item(), expected_dice.item(), places=6,
            msg=f"customloss Dice mismatch: fn={dice_from_fn.item():.8f}, expected={expected_dice.item():.8f}")


if __name__ == "__main__":
    unittest.main()
