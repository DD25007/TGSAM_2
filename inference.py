"""
TGSAM-2 Inference Script

Usage:
    python inference.py --checkpoint outputs/acdc/best_dsc_0.8763.pth \
                       --image_path dataset/ACDC/testing/patient101/patient101_frame01.nii.gz \
                       --text_prompt "Segment the left ventricle of the heart in cardiac MRI"
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from PIL import Image

# Project imports
from utils import SegmentationMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_nifti_volume(nifti_path: str) -> np.ndarray:
    """Load NIfTI medical image volume."""
    nifti = nib.load(nifti_path)
    volume = nifti.get_fdata().astype(np.float32)
    return volume


def load_image(image_path: str) -> np.ndarray:
    """Load single image (PNG, JPG, or NIfTI)."""
    if image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
        volume = load_nifti_volume(image_path)
        # Return middle slice
        if volume.ndim == 3:
            mid_idx = volume.shape[2] // 2
            image = volume[:, :, mid_idx]
        else:
            image = volume
    else:
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0

    return image


def normalize_image(
    image: np.ndarray, percentile_low: float = 0.5, percentile_high: float = 99.5
) -> np.ndarray:
    """Normalize image to [0, 1]."""
    lo = np.percentile(image, percentile_low)
    hi = np.percentile(image, percentile_high)
    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo + 1e-8)
    return image.astype(np.float32)


def resize_image(image: np.ndarray, target: int = 1024) -> np.ndarray:
    """Resize image to target size."""
    pil = Image.fromarray((image * 255).astype(np.uint8))
    pil = pil.resize((target, target), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert image to tensor."""
    t = torch.from_numpy(image).unsqueeze(0)  # 1xHxW
    return t.repeat(3, 1, 1)  # 3xHxW


def main(args):
    """Main inference function."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading TGSAM-2 model...")
    try:
        from model import TGSAM2

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = TGSAM2.from_pretrained(
            sam2_checkpoint="checkpoints/sam2_hiera_small.pt"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load image
    logger.info(f"Loading image from {args.image_path}")
    image = load_image(args.image_path)

    # Preprocess
    image = normalize_image(image)
    image = resize_image(image, target=args.image_size)
    image_tensor = image_to_tensor(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Text prompt
    text_prompt = args.text_prompt
    logger.info(f"Text prompt: {text_prompt}")

    # Inference
    logger.info("Running inference...")
    with torch.no_grad():
        # Wrap single frame as sequence
        output = model(frames=image_tensor, text_prompt=text_prompt)
        # output: (1, 1, H, W)

        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_binary = (pred_prob > args.threshold).astype(np.uint8)

    # Visualize (optional)
    if args.save_output:
        output_dir = Path(args.save_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save prediction
        pred_img = Image.fromarray((pred_binary * 255).astype(np.uint8))
        pred_img.save(output_dir / "prediction.png")

        # Save probability map
        prob_img = Image.fromarray((pred_prob * 255).astype(np.uint8))
        prob_img.save(output_dir / "probability_map.png")

        # Save composite
        image_uint8 = (image * 255).astype(np.uint8)
        composite = np.stack([image_uint8, image_uint8, image_uint8], axis=-1)
        composite[pred_binary > 0, 0] = 255  # Red channel for prediction
        composite_img = Image.fromarray(composite.astype(np.uint8))
        composite_img.save(output_dir / "composite.png")

        logger.info(f"Output saved to {output_dir}")

    # Compute metrics if reference mask provided
    if args.mask_path:
        logger.info(f"Loading mask from {args.mask_path}")
        mask = load_image(args.mask_path)
        mask = resize_image(mask, target=args.image_size)
        mask_tensor = torch.from_numpy((mask > 0.5).astype(np.uint8)).to(device)

        metrics = SegmentationMetrics()
        dsc = metrics.dice_similarity_coefficient(
            torch.from_numpy(pred_binary).to(device), mask_tensor
        )
        iou = metrics.intersection_over_union(
            torch.from_numpy(pred_binary).to(device), mask_tensor
        )
        sens, spec = metrics.sensitivity_specificity(
            torch.from_numpy(pred_binary).to(device), mask_tensor
        )

        logger.info("\n" + "=" * 40)
        logger.info("Metrics")
        logger.info("=" * 40)
        logger.info(f"DSC:         {dsc:.4f}")
        logger.info(f"IoU:         {iou:.4f}")
        logger.info(f"Sensitivity: {sens:.4f}")
        logger.info(f"Specificity: {spec:.4f}")
        logger.info("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGSAM-2 Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--image_path", required=True, help="Path to input image/volume"
    )
    parser.add_argument(
        "--text_prompt", required=True, help="Text description of target"
    )
    parser.add_argument(
        "--mask_path", default=None, help="Path to reference mask (optional)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold"
    )
    parser.add_argument("--image_size", type=int, default=1024, help="Input image size")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument(
        "--save_output", default=None, help="Directory to save output (optional)"
    )

    args = parser.parse_args()
    main(args)
