import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List  # Added imports


class customloss(nn.Module):
    """
    Custom loss combining Dice and boundary (BF1) losses with learned uncertainty weighting.

    Returns [dice_loss_scalar, boundary_loss_scalar, velocity_loss_scalar] for sigma-weighted combination.
    Velocity loss is optional and will return 0.0 if not enabled or velocity data is not provided.
    Uses target_int==255 as ignore mask.
    """

    def __init__(
        self,
        act=nn.Softmax(dim=1),
        smooth=1.0,
        label_smoothing=0.0,
        theta0=3,
        theta=5,
        verbose=False,
        class_weights: Optional[List[float]] = None,  # New parameter
        velocity_high_speed_threshold: float = 3.16,
        velocity_loss_weight: float = 0.05,
        velocity_loss_warmup_epochs: int = 10,
        velocity_loss_ramp_epochs: int = 10,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.theta0 = theta0
        self.theta = theta
        self.verbose = verbose
        self.class_weights = class_weights  # Store class weights
        self.velocity_high_speed_threshold = velocity_high_speed_threshold
        self.velocity_loss_weight = velocity_loss_weight
        self.velocity_loss_warmup_epochs = velocity_loss_warmup_epochs
        self.velocity_loss_ramp_epochs = velocity_loss_ramp_epochs

        # Flag for single-run debug logging
        self._first_call = True
        # Telemetry for logging loss magnitudes
        self.last_velocity_loss: Optional[torch.Tensor] = None
        self.last_velocity_base: Optional[torch.Tensor] = None
        self.last_velocity_weight: float = 0.0
        self.last_velocity_valid: bool = False

    def forward(
        self,
        pred,
        target,
        target_int,
        velocity=None,
        velocity_mask=None,
        current_epoch: Optional[int] = None,
    ):
        """
        pred:       (N,C,H,W) raw logits
        target:     (N,C,H,W) one-hot labels
        target_int: (N,H,W)   integer labels, 255 = ignore
        velocity:   (N,1,H,W) velocity magnitude (optional)
        velocity_mask: (N,1,H,W) velocity validity mask (optional)
        """
        n, c, h, w = pred.shape
        device = pred.device

        # Reset telemetry
        self.last_velocity_loss = torch.tensor(0.0, device=device)
        self.last_velocity_base = torch.tensor(0.0, device=device)
        self.last_velocity_weight = 0.0
        self.last_velocity_valid = False

        # Log classification type for debugging (only on first call)
        if self._first_call:
            if c == 2:
                print(f"[LOSS DEBUG] Binary classification mode (c={c})")
            elif c == 3:
                print(f"[LOSS DEBUG] Multi-class classification mode (c={c})")
            else:
                print(f"[LOSS DEBUG] Unknown classification mode (c={c})")
            self._first_call = False

        target = target.detach()

        if target_int is not None:
            ignore_mask = target_int != 255
        else:
            ignore_mask = target.sum(dim=1) == 1
        ignore_mask_exp = ignore_mask.unsqueeze(1).float().to(device)

        # Validate channel count
        if c <= 1:
            raise ValueError(
                f"Invalid channel count: {c}. Binary requires 2 channels, multi-class requires 3+ channels."
            )

        # Always train both logits; use softmax for binary too so BG receives gradients
        if c == 2:
            pred_prob = torch.softmax(pred, dim=1)
            target_prob = target
            C_eff = 2
        else:
            pred_prob = self.act(pred)  # Softmax
            target_prob = target
            C_eff = c

        pred_prob = pred_prob * ignore_mask_exp
        if self.label_smoothing > 0 and C_eff > 1:
            target_prob = (
                target_prob * (1 - self.label_smoothing) + self.label_smoothing / C_eff
            )
        # Reapply ignore mask after label smoothing so ignored pixels stay zeroed
        target_prob = target_prob * ignore_mask_exp

        # Prepare optional class weights once to share between dice and boundary losses
        weights_tensor = None
        if self.class_weights is not None:
            if len(self.class_weights) != C_eff:
                raise ValueError(
                    f"Length of class_weights ({len(self.class_weights)}) "
                    f"must match number of effective classes ({C_eff})"
                )
            weights_tensor = torch.tensor(
                self.class_weights, device=device, dtype=pred_prob.dtype
            )

        pred_flat = pred_prob.permute(0, 2, 3, 1)[ignore_mask]
        targ_flat = target_prob.permute(0, 2, 3, 1)[ignore_mask]

        if pred_flat.numel() == 0:
            dice_loss_scalar = torch.tensor(0.0, device=device)
        else:
            numerator = 2 * (pred_flat * targ_flat).sum(dim=0) + self.smooth
            denominator = pred_flat.sum(dim=0) + targ_flat.sum(dim=0) + self.smooth
            dice_per_class = 1 - numerator / denominator

            if weights_tensor is not None:
                dice_loss_scalar = (dice_per_class * weights_tensor).sum()
            else:
                # Fallback to equal weighting if no class_weights are provided
                dice_loss_scalar = dice_per_class.mean()

        pred_b_in = pred_prob
        targ_b_in = target_prob

        gt_b = F.max_pool2d(1 - targ_b_in, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - targ_b_in
        )
        pred_b = F.max_pool2d(1 - pred_b_in, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - pred_b_in
        )

        gt_b_ext = F.max_pool2d(gt_b, self.theta, 1, (self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, 1, (self.theta - 1) // 2)

        gt_b = gt_b * ignore_mask_exp
        pred_b = pred_b * ignore_mask_exp
        gt_b_ext = gt_b_ext * ignore_mask_exp
        pred_b_ext = pred_b_ext * ignore_mask_exp

        gt_b = gt_b.view(n, C_eff, -1)
        pred_b = pred_b.view(n, C_eff, -1)
        gt_b_ext = gt_b_ext.view(n, C_eff, -1)
        pred_b_ext = pred_b_ext.view(n, C_eff, -1)

        P = (pred_b * gt_b_ext).sum(dim=2) / (pred_b.sum(dim=2) + 1e-7)
        R = (pred_b_ext * gt_b).sum(dim=2) / (gt_b.sum(dim=2) + 1e-7)

        BF1 = 2 * P * R / (P + R + 1e-7)
        boundary_loss_unreduced = 1 - BF1

        if weights_tensor is not None:
            boundary_loss = (boundary_loss_unreduced.mean(dim=0) * weights_tensor).sum()
        else:
            boundary_loss = torch.mean(boundary_loss_unreduced)

        # Velocity Consistency Loss
        velocity_loss = torch.tensor(0.0, device=device)
        if velocity is not None and velocity_mask is not None:
            if self._first_call:
                print(f"[LOSS DEBUG] Velocity loss enabled with c={c} channels")

            if C_eff == 2:
                ice_prob = pred_prob[:, 1:2]
                bg_prob = pred_prob[:, 0:1]
            else:
                ice_prob = pred_prob[:, 1:].sum(dim=1, keepdim=True)
                bg_prob = pred_prob[:, :1]

            # Data-driven "Moving Background" Penalty
            # 1. Calculate Probability of Motion based on Physics
            # Sigmoid centered at 3.16 m/yr (BG 75th percentile) with moderate steepness (0.5)
            # This handles the high dynamic range of glacier speeds without exploding gradients.
            p_moving_physics = torch.sigmoid(
                (velocity - self.velocity_high_speed_threshold) * 0.5
            )

            # 2. Penalize Semantic Contradiction:
            # Model predicts Background (bg_prob ~ 1.0) BUT Physics says Moving (p_moving_physics ~ 1.0)
            # Loss is high when BOTH are high.
            per_pixel_penalty = bg_prob * p_moving_physics

            combined_mask = velocity_mask * ignore_mask_exp
            valid_count = combined_mask.sum()
            base_velocity_loss = torch.tensor(0.0, device=device)

            if valid_count > 0:
                base_velocity_loss = (
                    per_pixel_penalty * combined_mask
                ).sum() / valid_count

            if base_velocity_loss > 0:
                current_weight = self._compute_velocity_weight(current_epoch)
                velocity_loss = base_velocity_loss * current_weight

                self.last_velocity_base = base_velocity_loss.detach()
                self.last_velocity_loss = velocity_loss.detach()
                self.last_velocity_weight = float(current_weight)
                self.last_velocity_valid = True
            else:
                velocity_loss = torch.tensor(0.0, device=device)

        return [dice_loss_scalar, boundary_loss, velocity_loss]

    def _compute_velocity_weight(self, current_epoch: Optional[int]) -> float:
        """Compute velocity loss weight based on warmup schedule."""
        if current_epoch is None:
            return self.velocity_loss_weight

        if current_epoch < self.velocity_loss_warmup_epochs:
            return 0.0

        # Calculate progress through ramp period (0.0 to 1.0)
        ramp_progress = (current_epoch - self.velocity_loss_warmup_epochs) / max(
            1, self.velocity_loss_ramp_epochs
        )

        if ramp_progress >= 1.0:
            return self.velocity_loss_weight

        return self.velocity_loss_weight * ramp_progress
