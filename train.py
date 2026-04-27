"""
TGSAM-2 Training Script

Usage:
    python train.py --config configs/acdc.yaml --device cuda:0
    python train.py --config configs/spleen.yaml --organ spleen
"""

import yaml
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Project imports
from data import get_dataset, collate_sequences
from utils import DiceBCELoss, evaluate_batch

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


def setup_directories(config: dict):
    """Create output directories for checkpoints and logs."""
    checkpoint_dir = Path(config["train_config"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    return checkpoint_dir, log_dir


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_dsc = []
    all_iou = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

    for batch_idx, batch in enumerate(pbar):
        images = batch["images"]  # List of (T, 3, H, W) tensors
        masks = batch["masks"]  # List of (T, H, W) tensors
        prompts = batch["prompts"]

        optimizer.zero_grad()

        # Process each sequence in batch
        batch_loss = 0.0
        for seq_img, seq_mask, prompt in zip(images, masks, prompts):
            # Move to device
            seq_img = seq_img.to(device)  # (T, 3, 1024, 1024)
            seq_mask = seq_mask.to(device).float()  # (T, 1024, 1024)

            # Forward pass through TGSAM2
            try:
                # Model expects (B, T, 3, H, W) and texts as List[str]
                result = model(
                    frames=seq_img.unsqueeze(0),  # (1, T, 3, H, W)
                    texts=[prompt],  # List with one text
                    gt_masks=seq_mask.unsqueeze(0).unsqueeze(2),  # (1, T, 1, H, W)
                )
                outputs = result["pred_masks"].squeeze(0)  # (T, 1, H, W)

                # Compute loss over all frames
                loss = criterion(outputs, seq_mask.unsqueeze(1))
                batch_loss += loss

                # Track metrics
                with torch.no_grad():
                    pred_binary = (outputs > 0.5).long()
                    metrics = evaluate_batch(pred_binary.float(), seq_mask.unsqueeze(1))
                    all_dsc.extend(metrics["dsc_per_sample"])
                    all_iou.extend(metrics["iou_per_sample"])

            except Exception as e:
                logger.warning(f"Error processing sequence: {e}")
                import traceback

                logger.warning(f"Traceback: {traceback.format_exc()}")
                continue

        # Backward pass
        if batch_loss > 0:
            (batch_loss / max(len(images), 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            pbar.set_postfix(
                {
                    "loss": batch_loss.item() / max(len(images), 1),
                    "dsc": np.mean(all_dsc[-10:]) if all_dsc else 0,
                    "iou": np.mean(all_iou[-10:]) if all_iou else 0,
                }
            )

            # Periodic GPU memory cleanup (every 5 batches)
            if (batch_idx + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Average metrics
    avg_loss = total_loss / max(len(train_loader), 1)
    avg_dsc = np.mean(all_dsc) if all_dsc else 0.0
    avg_iou = np.mean(all_iou) if all_iou else 0.0

    return {
        "loss": avg_loss,
        "dsc": avg_dsc,
        "iou": avg_iou,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """Validate model on validation set."""
    model.eval()

    total_loss = 0.0
    all_dsc = []
    all_iou = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["images"]
            masks = batch["masks"]
            prompts = batch["prompts"]

            for seq_img, seq_mask, prompt in zip(images, masks, prompts):
                seq_img = seq_img.to(device)
                seq_mask = seq_mask.to(device).float()

                try:
                    # Model expects (B, T, 3, H, W) and texts as List[str]
                    result = model(
                        frames=seq_img.unsqueeze(0),  # (1, T, 3, H, W)
                        texts=[prompt],  # List with one text
                    )
                    outputs = result["pred_masks"].squeeze(0)  # (T, 1, H, W)
                    loss = criterion(outputs, seq_mask.unsqueeze(1))
                    total_loss += loss.item()

                    pred_binary = (outputs > 0.5).long()
                    metrics = evaluate_batch(pred_binary.float(), seq_mask.unsqueeze(1))
                    all_dsc.extend(metrics["dsc_per_sample"])
                    all_iou.extend(metrics["iou_per_sample"])
                except Exception as e:
                    logger.warning(f"Error in validation: {e}")
                    continue

    avg_loss = total_loss / max(len(val_loader), 1)
    avg_dsc = np.mean(all_dsc) if all_dsc else 0.0
    avg_iou = np.mean(all_iou) if all_iou else 0.0

    return {
        "loss": avg_loss,
        "dsc": avg_dsc,
        "iou": avg_iou,
    }


def main(args):
    """Main training loop."""
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

    # Setup directories
    checkpoint_dir, log_dir = setup_directories(config)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = get_dataset(
        name=config["dataset"]["name"],
        root=config["dataset"]["root"],
        split=config["dataset"].get("split", "train"),
        image_size=config["dataset"]["image_size"],
        augment=config["dataset"]["augment"],
    )

    logger.info("Loading validation dataset...")
    val_dataset = get_dataset(
        name=config["dataset"]["name"],
        root=config["dataset"]["root"],
        split=config["dataset"].get("test_split", "test"),
        image_size=config["dataset"]["image_size"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=2,
    )

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")

    # Initialize model
    logger.info("Initializing TGSAM-2 model...")
    try:
        from model import TGSAM2

        model = TGSAM2.from_pretrained(
            sam2_checkpoint="checkpoints/sam2_hiera_small.pt"
        )
        model = model.to(device)
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Make sure SAM-2 is installed: pip install -e segment-anything-2/")
        return

    # Loss & optimizer
    criterion = DiceBCELoss(
        dice_weight=config["loss"]["dice_weight"],
        bce_weight=config["loss"]["bce_weight"],
        smooth=config["loss"]["smooth"],
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
        weight_decay=config["train"]["weight_decay"],
    )

    scheduler = StepLR(
        optimizer,
        step_size=config["train"]["lr_schedule_params"]["step_size"],
        gamma=config["train"]["lr_schedule_params"]["gamma"],
    )

    # Training loop
    logger.info("Starting training...")
    best_dsc = 0.0

    for epoch in range(config["train"]["num_epochs"]):
        # Clear GPU memory at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        # Validate
        if (epoch + 1) % config["train_config"]["validation_frequency"] == 0:
            # Clear GPU memory before validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_metrics = validate(model, val_loader, criterion, device, config)

            logger.info(
                f"Epoch {epoch+1}/{config['train']['num_epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} DSC: {train_metrics['dsc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} DSC: {val_metrics['dsc']:.4f}"
            )

            # Clear GPU memory after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save best checkpoint
            if val_metrics["dsc"] > best_dsc:
                best_dsc = val_metrics["dsc"]
                ckpt_path = checkpoint_dir / f"best_dsc_{best_dsc:.4f}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "metrics": val_metrics,
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")

        # Save periodic checkpoint
        if (epoch + 1) % config["train_config"]["save_frequency"] == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )

        scheduler.step()

    logger.info("Training complete!")
    logger.info(f"Best validation DSC: {best_dsc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TGSAM-2")
    parser.add_argument(
        "--config", default="configs/acdc.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use (cuda, cpu, etc.)"
    )

    args = parser.parse_args()
    main(args)
