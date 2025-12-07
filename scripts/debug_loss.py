import torch
import torch.nn as nn
from glacier_mapping.model.losses import customloss


def test_loss():
    print("Testing Custom Loss...")

    # Setup
    B, C, H, W = 2, 3, 64, 64  # Batch, Classes (BG, Clean, Debris), Height, Width
    loss_fn = customloss(masked=True)

    # 1. Perfect Prediction
    print("\nCase 1: Perfect Prediction")
    pred = torch.zeros(B, C, H, W)
    pred[:, 0, :, :] = 10.0  # High logit for BG
    target = torch.zeros(B, C, H, W)
    target[:, 0, :, :] = 1.0  # BG is true
    target_int = torch.zeros(B, H, W).long()  # Class 0

    # Velocity data (Zero velocity for BG)
    velocity = torch.zeros(B, 1, H, W)
    velocity_mask = torch.ones(B, 1, H, W)

    losses = loss_fn(pred, target, target_int, velocity, velocity_mask)
    print(f"Losses (Dice, Bound, Vel): {[l.item() for l in losses]}")
    # Expect all close to 0

    # 2. Total Failure (Predict BG, Truth is Debris)
    print("\nCase 2: Total Failure (Predict BG, Truth is Debris)")
    pred = torch.zeros(B, C, H, W)
    pred[:, 0, :, :] = 10.0  # Predict BG

    target = torch.zeros(B, C, H, W)
    target[:, 2, :, :] = 1.0  # Truth is Debris
    target_int = torch.ones(B, H, W).long() * 2  # Class 2

    # Velocity high (should be debris)
    velocity = torch.ones(B, 1, H, W) * 100.0

    losses = loss_fn(pred, target, target_int, velocity, velocity_mask)
    print(f"Losses (Dice, Bound, Vel): {[l.item() for l in losses]}")
    # Expect Dice high, Vel high (Predicted BG where Vel is high)

    # Check Velocity Term specifically
    # Formula: P(BG) * Vel
    # P(BG) approx 1.0. Vel = 100.
    # Loss should be approx 100?

    # 3. Physics Check (Predict Debris, Truth is Debris, Velocity High)
    print("\nCase 3: Physics Check (Predict Debris, Truth is Debris, Vel High)")
    pred = torch.zeros(B, C, H, W)
    pred[:, 2, :, :] = 10.0  # Predict Debris

    # P(BG) should be low (~0)

    # 4. Real Debris Scenario (2 Channels, Config says [2])
    print("\nCase 4: Real Debris Scenario (2 Channels, Config says [2])")
    # Model outputs 2 channels: 0=BG, 1=Debris
    B, C, H, W = 2, 2, 64, 64
    pred = torch.zeros(B, C, H, W)
    pred[:, 0, :, :] = 10.0  # Predict BG

    target = torch.zeros(B, C, H, W)
    target[:, 1, :, :] = 1.0  # Truth is Debris (Channel 1)
    target_int = torch.ones(B, H, W).long() * 1  # Index 1

    # Init loss with [2] (Simulating the bug)
    loss_fn_bug = customloss(masked=True, foreground_classes=[2])
    try:
        losses = loss_fn_bug(pred, target, target_int, velocity, velocity_mask)
        print(f"Losses (Dice, Bound, Vel): {[l.item() for l in losses]}")
        print("  -> Dice is 0? Bug confirmed.")
    except Exception as e:
        print(f"  -> Error: {e}")

    # Init loss with [1] (The Fix)
    loss_fn_fix = customloss(masked=True, foreground_classes=[1])
    losses = loss_fn_fix(pred, target, target_int, velocity, velocity_mask)
    print(f"Losses with Fix (Dice, Bound, Vel): {[l.item() for l in losses]}")
    print("  -> Dice should be high.")
