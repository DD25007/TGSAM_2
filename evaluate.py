"""
TGSAM-2 Evaluation Script

Usage:
    python evaluate.py --config configs/acdc.yaml --checkpoint outputs/acdc/best_dsc_0.8763.pth
"""

import yaml
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
from data import get_dataset, collate_sequences
from utils import SegmentationMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: dict,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on test set."""
    model.eval()

    metrics = SegmentationMetrics()

    all_dsc = []
    all_iou = []
    all_sensitivity = []
    all_specificity = []

    results_per_sample = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"]
            masks = batch["masks"]
            prompts = batch["prompts"]

            for seq_idx, (seq_img, seq_mask, prompt) in enumerate(
                zip(images, masks, prompts)
            ):
                seq_img = seq_img.to(device)  # (T, 3, H, W)
                seq_mask = seq_mask.to(device).float()  # (T, H, W)

                try:
                    # Forward pass
                    outputs = model(frames=seq_img, text_prompt=prompt)
                    # outputs: (T, 1, H, W) or (T, H, W)

                    if outputs.dim() == 4:
                        outputs = outputs.squeeze(1)

                    # Binarize
                    pred_binary = (outputs > threshold).long()

                    # Compute metrics for each frame
                    for frame_idx in range(seq_img.shape[0]):
                        pred_frame = pred_binary[frame_idx]
                        mask_frame = seq_mask[frame_idx]

                        dsc = metrics.dice_similarity_coefficient(
                            pred_frame, mask_frame
                        )
                        iou = metrics.intersection_over_union(pred_frame, mask_frame)
                        sens, spec = metrics.sensitivity_specificity(
                            pred_frame, mask_frame
                        )

                        all_dsc.append(dsc)
                        all_iou.append(iou)
                        all_sensitivity.append(sens)
                        all_specificity.append(spec)

                        results_per_sample.append(
                            {
                                "batch": batch_idx,
                                "sequence": seq_idx,
                                "frame": frame_idx,
                                "dsc": dsc,
                                "iou": iou,
                                "sensitivity": sens,
                                "specificity": spec,
                            }
                        )

                    pbar.set_postfix(
                        {
                            "dsc": np.mean(all_dsc[-10:]),
                            "iou": np.mean(all_iou[-10:]),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error processing sequence {seq_idx}: {e}")
                    continue

    # Compute statistics
    results = {
        "dsc": {
            "mean": np.mean(all_dsc),
            "std": np.std(all_dsc),
            "min": np.min(all_dsc),
            "max": np.max(all_dsc),
        },
        "iou": {
            "mean": np.mean(all_iou),
            "std": np.std(all_iou),
            "min": np.min(all_iou),
            "max": np.max(all_iou),
        },
        "sensitivity": {
            "mean": np.mean(all_sensitivity),
            "std": np.std(all_sensitivity),
        },
        "specificity": {
            "mean": np.mean(all_specificity),
            "std": np.std(all_specificity),
        },
        "num_samples": len(all_dsc),
        "results_per_sample": results_per_sample,
    }

    return results


def save_results(results: dict, output_path: Path):
    """Save evaluation results to file."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    import json

    # Convert numpy values to float for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    results_serializable = convert_to_serializable(results)

    with open(output_path / "results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Save summary report
    with open(output_path / "summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TGSAM-2 Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Number of samples: {results['num_samples']}\n\n")

        f.write("Dice Similarity Coefficient (DSC)\n")
        f.write(f"  Mean: {results['dsc']['mean']:.4f}\n")
        f.write(f"  Std:  {results['dsc']['std']:.4f}\n")
        f.write(f"  Min:  {results['dsc']['min']:.4f}\n")
        f.write(f"  Max:  {results['dsc']['max']:.4f}\n\n")

        f.write("Intersection over Union (IoU)\n")
        f.write(f"  Mean: {results['iou']['mean']:.4f}\n")
        f.write(f"  Std:  {results['iou']['std']:.4f}\n")
        f.write(f"  Min:  {results['iou']['min']:.4f}\n")
        f.write(f"  Max:  {results['iou']['max']:.4f}\n\n")

        f.write("Sensitivity\n")
        f.write(f"  Mean: {results['sensitivity']['mean']:.4f}\n")
        f.write(f"  Std:  {results['sensitivity']['std']:.4f}\n\n")

        f.write("Specificity\n")
        f.write(f"  Mean: {results['specificity']['mean']:.4f}\n")
        f.write(f"  Std:  {results['specificity']['std']:.4f}\n")

    logger.info(f"Results saved to {output_path}")


def main(args):
    """Main evaluation function."""
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # GPU memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("GPU memory cleared")

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = get_dataset(
        name=config["dataset"]["name"],
        root=config["dataset"]["root"],
        split="test",
        image_size=config["dataset"]["image_size"],
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=2,
    )

    logger.info(f"Test set: {len(test_dataset)} samples")

    # Load model
    logger.info("Loading TGSAM-2 model...")
    try:
        from model import TGSAM2

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model = TGSAM2.from_pretrained(
                sam2_checkpoint="checkpoints/sam2_hiera_small.pt"
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        else:
            model = TGSAM2.from_pretrained(
                sam2_checkpoint="checkpoints/sam2_hiera_small.pt"
            )

        model = model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Make sure SAM-2 is installed: pip install -e segment-anything-2/")
        return

    # Evaluate
    logger.info("Starting evaluation...")
    
    # Clear GPU memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    results = evaluate(
        model,
        test_loader,
        device,
        config,
        threshold=config["eval"]["threshold"],
    )
    
    # Clear GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"DSC:   {results['dsc']['mean']:.4f} ± {results['dsc']['std']:.4f}")
    logger.info(f"IoU:   {results['iou']['mean']:.4f} ± {results['iou']['std']:.4f}")
    logger.info(
        f"Sens:  {results['sensitivity']['mean']:.4f} ± {results['sensitivity']['std']:.4f}"
    )
    logger.info(
        f"Spec:  {results['specificity']['mean']:.4f} ± {results['specificity']['std']:.4f}"
    )
    logger.info(f"Samples: {results['num_samples']}")
    logger.info("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TGSAM-2")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument(
        "--output_dir", default="outputs/eval", help="Directory to save results"
    )

    args = parser.parse_args()
    main(args)
