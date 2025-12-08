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
        masked=True,
        theta0=3,
        theta=5,
        alpha=0.9,  # compatibility with old code
        verbose=False,
        class_weights: Optional[List[float]] = None,  # New parameter
        velocity_confidence_threshold: float = 0.6,
        velocity_low_speed_threshold: float = 0.1,
        velocity_loss_weight: float = 0.05,
        velocity_loss_warmup_epochs: int = 10,
        velocity_loss_ramp_epochs: int = 10,
        velocity_loss_clip: Optional[float] = 1.0,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.theta0 = theta0
        self.theta = theta
        self.verbose = verbose
        self.class_weights = class_weights  # Store class weights
        self.velocity_confidence_threshold = velocity_confidence_threshold
        self.velocity_low_speed_threshold = velocity_low_speed_threshold
        self.velocity_loss_weight = velocity_loss_weight
        self.velocity_loss_warmup_epochs = velocity_loss_warmup_epochs
        self.velocity_loss_ramp_epochs = velocity_loss_ramp_epochs
        self.velocity_loss_clip = velocity_loss_clip

        # Flag for single-run debug logging
        self._first_call = True

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

        # Use sigmoid for binary, softmax for multi-class
        if c == 2:  # Binary classification (background + foreground)
            # Construct two-channel probability tensors
            pred_fg = torch.sigmoid(pred[:, 1:2])  # Foreground probability
            pred_bg = 1.0 - pred_fg  # Background probability
            pred_prob = torch.cat([pred_bg, pred_fg], dim=1)  # (N, 2, H, W)

            # Construct two-channel one-hot target
            target_fg = target[:, 1:2]  # Foreground target
            target_bg = 1.0 - target_fg  # Background target
            target_prob = torch.cat([target_bg, target_fg], dim=1)  # (N, 2, H, W)

            C_eff = 2
        else:  # Multi-class classification
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
            else:
                ice_prob = pred_prob[:, 1:].sum(dim=1, keepdim=True)

            confidence_mask = (ice_prob > self.velocity_confidence_threshold).float()
            low_speed_penalty = torch.clamp(
                self.velocity_low_speed_threshold - velocity, min=0.0
            )
            per_pixel_penalty = ice_prob * low_speed_penalty

            if self.velocity_loss_clip is not None:
                per_pixel_penalty = torch.clamp(
                    per_pixel_penalty, max=self.velocity_loss_clip
                )

            combined_mask = confidence_mask * velocity_mask * ignore_mask_exp
            valid_count = combined_mask.sum()
            if valid_count > 0:
                numerator = (per_pixel_penalty * combined_mask).sum()
                base_velocity_loss = numerator / (valid_count + 1e-7)
                weight = self._compute_velocity_weight(current_epoch)
                velocity_loss = base_velocity_loss * weight
            else:
                velocity_loss = torch.tensor(0.0, device=device)

        return [dice_loss_scalar, boundary_loss, velocity_loss]

    def _compute_velocity_weight(self, current_epoch: Optional[int]) -> float:
        if current_epoch is None:
            return self.velocity_loss_weight

        if current_epoch < self.velocity_loss_warmup_epochs:
            return 0.0

        ramp_start = self.velocity_loss_warmup_epochs
        ramp_end = ramp_start + max(1, self.velocity_loss_ramp_epochs)
        if current_epoch < ramp_end:
            progress = (current_epoch - ramp_start + 1) / (
                ramp_end - ramp_start + 1e-7
            )
            return self.velocity_loss_weight * max(0.0, min(1.0, progress))

        return self.velocity_loss_weight
