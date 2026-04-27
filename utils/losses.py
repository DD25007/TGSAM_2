"""
Loss functions for TGSAM-2 medical image segmentation.

Includes:
  - Dice Loss
  - Binary Cross Entropy Loss
  - Combined DiceBCE Loss (paper uses this)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss: 1 - (2*TP) / (2*TP + FP + FN)
    Smooth: small epsilon to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, H, W) or (B, 1, H, W) predictions [0,1]
            target: (B, H, W) or (B, 1, H, W) binary labels {0, 1}
        Returns:
            Scalar loss
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss with optional focal weighting."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, H, W) or (B, 1, H, W) logits
            target: (B, H, W) or (B, 1, H, W) binary labels
        Returns:
            Scalar loss
        """
        return F.binary_cross_entropy_with_logits(pred, target.float())


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss (paper uses this).
    Paper: L = 0.5 * L_dice + 0.5 * L_bce
    """

    def __init__(
        self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, H, W) logits
            target: (B, H, W) binary {0, 1}
        Returns:
            Scalar loss
        """
        dice_loss = self.dice(torch.sigmoid(pred), target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, H, W) logits
            target: (B, H, W) binary {0, 1}
        """
        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        focal = self.alpha * (1 - p) ** self.gamma * ce
        return focal.mean()
