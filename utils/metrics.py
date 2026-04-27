"""
Metrics for medical image segmentation evaluation.

Includes:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU)
  - Sensitivity, Specificity
"""

import torch
import numpy as np
from typing import Tuple


class SegmentationMetrics:
    """Compute segmentation metrics (DSC, IoU, etc.)"""

    @staticmethod
    def dice_similarity_coefficient(
        pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
    ) -> float:
        """
        Dice Similarity Coefficient (DSC).

        DSC = (2*TP) / (2*TP + FP + FN)

        Args:
            pred: (B, H, W) or (B, 1, H, W) binary predictions {0, 1}
            target: (B, H, W) or (B, 1, H, W) binary ground truth {0, 1}
            smooth: Regularization constant

        Returns:
            DSC as float [0, 1]
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()

    @staticmethod
    def intersection_over_union(
        pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
    ) -> float:
        """
        Intersection over Union (IoU).

        IoU = TP / (TP + FP + FN)

        Args:
            pred: (B, H, W) or (B, 1, H, W) binary predictions {0, 1}
            target: (B, H, W) or (B, 1, H, W) binary ground truth {0, 1}
            smooth: Regularization constant

        Returns:
            IoU as float [0, 1]
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    @staticmethod
    def sensitivity_specificity(
        pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Sensitivity (Recall / True Positive Rate) and Specificity (True Negative Rate).

        Args:
            pred: (B, H, W) binary predictions {0, 1}
            target: (B, H, W) binary ground truth {0, 1}

        Returns:
            (sensitivity, specificity) as tuple of floats
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        tp = (pred * target).sum().item()
        fp = (pred * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()
        tn = ((1 - pred) * (1 - target)).sum().item()

        sensitivity = tp / (tp + fn + 1e-8)  # recall
        specificity = tn / (tn + fp + 1e-8)

        return sensitivity, specificity

    @staticmethod
    def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Hausdorff Distance between two binary masks.

        Args:
            pred: (H, W) binary mask {0, 1}
            target: (H, W) binary mask {0, 1}

        Returns:
            Hausdorff distance as float
        """
        try:
            from scipy.spatial.distance import directed_hausdorff
        except ImportError:
            print("scipy not available, returning 0")
            return 0.0

        pred = pred.astype(bool)
        target = target.astype(bool)

        # Get coordinates of True pixels
        coords_pred = np.argwhere(pred)
        coords_target = np.argwhere(target)

        if len(coords_pred) == 0 or len(coords_target) == 0:
            return 0.0

        # Hausdorff distance
        d1 = directed_hausdorff(coords_pred, coords_target)[0]
        d2 = directed_hausdorff(coords_target, coords_pred)[0]

        return max(d1, d2)


def evaluate_batch(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict:
    """
    Evaluate a batch of predictions.

    Args:
        pred: (B, H, W) logits or probabilities
        target: (B, H, W) binary {0, 1}
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary with metrics for each sample and batch average
    """
    # Binarize predictions
    pred_binary = (pred > threshold).long()
    target = target.long()

    metrics = SegmentationMetrics()

    batch_dsc = []
    batch_iou = []
    batch_sens = []
    batch_spec = []

    for i in range(pred_binary.shape[0]):
        dsc = metrics.dice_similarity_coefficient(pred_binary[i], target[i])
        iou = metrics.intersection_over_union(pred_binary[i], target[i])
        sens, spec = metrics.sensitivity_specificity(pred_binary[i], target[i])

        batch_dsc.append(dsc)
        batch_iou.append(iou)
        batch_sens.append(sens)
        batch_spec.append(spec)

    return {
        "dsc_per_sample": batch_dsc,
        "iou_per_sample": batch_iou,
        "sensitivity_per_sample": batch_sens,
        "specificity_per_sample": batch_spec,
        "dsc_mean": np.mean(batch_dsc),
        "iou_mean": np.mean(batch_iou),
        "sensitivity_mean": np.mean(batch_sens),
        "specificity_mean": np.mean(batch_spec),
    }
