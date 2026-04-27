"""
TGSAM-2 Dataset Loaders
Handles all four modalities from the paper (§3.1):
  - ACDC      (MRI, NIfTI .nii.gz)
  - Spleen    (CT,  NIfTI .nii.gz, MSD format)
  - Prostate  (Ultrasound, image frames)
  - CVC-ClinicDB (Endoscopy, PNG frames)

Each dataset returns sequences (video-like streams) of (image, mask, text_prompt).
"""

import os
import glob
import json
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── Text prompts (imported from parent directory) ─────────────────────────────────
import sys
from pathlib import Path

# Add parent directory to path to import text_prompts
sys.path.insert(0, str(Path(__file__).parent.parent))
from text_prompts import TEXT_PROMPTS


# ── Shared helpers ────────────────────────────────────────────────────────────


def normalise(
    volume: np.ndarray, percentile_low: float = 0.5, percentile_high: float = 99.5
) -> np.ndarray:
    """Robust percentile normalisation → [0, 1]."""
    lo = np.percentile(volume, percentile_low)
    hi = np.percentile(volume, percentile_high)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-8)
    return volume.astype(np.float32)


def resize_and_pad(img: np.ndarray, target: int = 1024) -> np.ndarray:
    """Resize HxW image (float32, [0,1]) to targetxtarget with zero-padding."""
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.resize((target, target), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, target: int = 1024) -> np.ndarray:
    """Resize binary mask (nearest neighbour) to targetxtarget."""
    pil = Image.fromarray(mask.astype(np.uint8))
    pil = pil.resize((target, target), Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """HxW float32 [0,1] → 3xHxW float32 tensor."""
    t = torch.from_numpy(frame).unsqueeze(0)  # 1xHxW
    return t.repeat(3, 1, 1)  # 3xHxW


def augment(
    frame: np.ndarray, mask: np.ndarray, image_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """Random flip + rotation data augmentation."""
    if np.random.rand() > 0.5:
        frame = np.fliplr(frame).copy()
        mask = np.fliplr(mask).copy()
    if np.random.rand() > 0.5:
        frame = np.flipud(frame).copy()
        mask = np.flipud(mask).copy()
    angle = np.random.uniform(-15, 15)
    pil_f = Image.fromarray((frame * 255).astype(np.uint8))
    pil_m = Image.fromarray(mask.astype(np.uint8))
    pil_f = TF.rotate(pil_f, angle, interpolation=T.InterpolationMode.BILINEAR)
    pil_m = TF.rotate(pil_m, angle, interpolation=T.InterpolationMode.NEAREST)
    frame = np.array(pil_f, dtype=np.float32) / 255.0
    mask = np.array(pil_m, dtype=np.uint8)
    return frame, mask


# ═══════════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════════


class MedicalSequenceDataset(Dataset):
    """
    Base class.  Each item is one *sequence* (all slices / frames for one
    object in one study), returned as:
        images  : T x 3 x H x W   float32
        masks   : T x H x W       uint8   (binary, 0/1)
        prompt  : str
    """

    def __init__(self, image_size: int = 1024, augment: bool = False):
        self.image_size = image_size
        self.do_augment = augment
        self.samples: List[dict] = []  # filled by subclasses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _pack(
        self, frames: List[np.ndarray], masks: List[np.ndarray], prompt: str
    ) -> dict:
        """Stack lists → tensors and return a dict."""
        imgs = torch.stack([frame_to_tensor(f) for f in frames])  # Tx3xHxW
        msks = torch.from_numpy(np.stack(masks, axis=0).astype(np.uint8))  # TxHxW
        return {"images": imgs, "masks": msks, "prompt": prompt}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ACDC  (MRI cardiac)
# ═══════════════════════════════════════════════════════════════════════════════

ACDC_LABEL_MAP = {
    "lv": (3, "left_ventricle"),
    "rv": (1, "right_ventricle"),
    "myo": (2, "myocardium"),
}


class ACDCDataset(MedicalSequenceDataset):
    """
    ACDC dataset structure (standard challenge layout):
        dataset/ACDC/
            training/
                patient001/
                    patient001_frame01.nii.gz   ← ED frame
                    patient001_frame01_gt.nii.gz
                    patient001_frame12.nii.gz   ← ES frame
                    patient001_frame12_gt.nii.gz
                    Info.cfg
            testing/
                patient101/  ...

    Each NIfTI is a 3-D volume (H x W x D slices).
    We treat each slice as one "frame" in the video stream.
    One Dataset instance covers ONE cardiac structure.
    """

    def __init__(
        self,
        root: str,
        structure: str = "lv",  # "lv" | "rv" | "myo"
        split: str = "training",
        image_size: int = 1024,
        augment: bool = False,
    ):
        super().__init__(image_size, augment)
        assert structure in ACDC_LABEL_MAP, f"Unknown structure: {structure}"
        self.label_value, self.struct_key = ACDC_LABEL_MAP[structure]
        self.prompt = TEXT_PROMPTS["acdc"][self.struct_key]
        self._build_samples(root, split)

    def _build_samples(self, root: str, split: str):
        split_dir = os.path.join(root, split)
        patient_dirs = sorted(glob.glob(os.path.join(split_dir, "patient*")))
        for pdir in patient_dirs:
            # Find all *_gt.nii.gz or *_gt.nii files (one per cardiac phase)
            gt_files = sorted(glob.glob(os.path.join(pdir, "*_gt.nii*")))
            for gt_path in gt_files:
                # Handle both .nii.gz and .nii formats
                if gt_path.endswith(".nii.gz"):
                    img_path = gt_path.replace("_gt.nii.gz", ".nii.gz")
                else:
                    img_path = gt_path.replace("_gt.nii", ".nii")

                if not os.path.exists(img_path):
                    continue
                self.samples.append({"image": img_path, "mask": gt_path})

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_vol = nib.load(s["image"]).get_fdata().astype(np.float32)  # HxWxD
        msk_vol = nib.load(s["mask"]).get_fdata().astype(np.int32)  # HxWxD

        # Reorient to DxHxW  (slices first)
        img_vol = np.transpose(img_vol, (2, 0, 1))
        msk_vol = np.transpose(msk_vol, (2, 0, 1))

        frames, masks = [], []
        for i in range(img_vol.shape[0]):
            sl = normalise(img_vol[i])
            msk = (msk_vol[i] == self.label_value).astype(np.uint8)

            sl = resize_and_pad(sl, self.image_size)
            msk = resize_mask(msk, self.image_size)

            if self.do_augment:
                sl, msk = augment(sl, msk, self.image_size)

            frames.append(sl)
            masks.append(msk)

        return self._pack(frames, masks, self.prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MSD Spleen  (CT)
# ═══════════════════════════════════════════════════════════════════════════════


class SpleenDataset(MedicalSequenceDataset):
    """
    MSD Task09_Spleen layout:
        dataset/spleen/
            imagesTr/spleen_001.nii.gz   (H x W x D)
            labelsTr/spleen_001.nii.gz   (binary: 0 bg, 1 spleen)
            imagesTs/...  (no labels → use for inference only)
    We use a manual train/test split via a JSON file or the first N volumes.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",  # "train" | "test"
        train_ratio: float = 0.73,  # 30 / (30+11) ≈ 0.73
        image_size: int = 1024,
        augment: bool = False,
    ):
        super().__init__(image_size, augment)
        self.prompt = TEXT_PROMPTS["spleen"]["spleen"]
        self._build_samples(root, split, train_ratio)

    def _build_samples(self, root: str, split: str, train_ratio: float):
        img_dir = os.path.join(root, "imagesTr")
        lbl_dir = os.path.join(root, "labelsTr")
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))

        # Check for official dataset.json split
        json_path = os.path.join(root, "dataset.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            # dataset.json lists {"training": [{"image":..,"label":..}, ...]}
            all_pairs = [
                (os.path.join(root, e["image"]), os.path.join(root, e["label"]))
                for e in meta.get("training", [])
            ]
        else:
            all_pairs = [(p, p.replace("imagesTr", "labelsTr")) for p in all_imgs]

        n_train = int(len(all_pairs) * train_ratio)
        pairs = all_pairs[:n_train] if split == "train" else all_pairs[n_train:]

        for img_path, lbl_path in pairs:
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                self.samples.append({"image": img_path, "mask": lbl_path})

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_vol = nib.load(s["image"]).get_fdata().astype(np.float32)
        msk_vol = nib.load(s["mask"]).get_fdata().astype(np.int32)

        # DxHxW
        img_vol = np.transpose(img_vol, (2, 0, 1))
        msk_vol = np.transpose(msk_vol, (2, 0, 1))

        frames, masks = [], []
        for i in range(img_vol.shape[0]):
            sl = normalise(img_vol[i], percentile_low=0.5, percentile_high=99.5)
            msk = (msk_vol[i] > 0).astype(np.uint8)

            # Skip empty slices during training to save compute
            if self.do_augment and msk.sum() == 0:
                continue

            sl = resize_and_pad(sl, self.image_size)
            msk = resize_mask(msk, self.image_size)

            if self.do_augment:
                sl, msk = augment(sl, msk, self.image_size)

            frames.append(sl)
            masks.append(msk)

        if len(frames) == 0:
            # Fallback: return blank sequence (should not happen with real data)
            blank = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            frames, masks = [blank], [blank.astype(np.uint8)]

        return self._pack(frames, masks, self.prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Micro-Ultrasound Prostate  (Ultrasound video)
# ═══════════════════════════════════════════════════════════════════════════════


class ProstateDataset(MedicalSequenceDataset):
    """
    Micro-Ultrasound Prostate Segmentation Dataset (MicroSegNet).
    Actual layout:
        dataset/Micro_Ultrasound_Prostate_Segmentation_Dataset/
            train/
                micro_ultrasound_scans/  microUS_train_01.nii.gz ...
                expert_annotations/      expert_annotation_train_01.nii.gz ...
                non_expert_annotations/  nonexpert_annotation_train_01.nii.gz ...
            test/
                micro_ultrasound_scans/  microUS_test_01.nii.gz ...
                expert_annotations/      expert_annotation_test_01.nii.gz ...
                clinician_annotations/   clinician_annotation_test_01.nii.gz ...

    Data are 3D NIfTI volumes; we extract 2D slices like ACDC/Spleen.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 1024,
        augment: bool = False,
    ):
        super().__init__(image_size, augment)
        self.prompt = TEXT_PROMPTS["prostate"]["prostate"]
        self._build_samples(root, split)

    def _build_samples(self, root: str, split: str):
        split_dir = os.path.join(root, split)

        # Get image and mask directories
        scan_dir = os.path.join(split_dir, "micro_ultrasound_scans")
        # Use expert_annotations for both train and test (experts are available for all)
        annotation_dir = os.path.join(split_dir, "expert_annotations")

        if not os.path.exists(scan_dir):
            return  # No data for this split

        # Get all scan volumes
        scans = sorted(glob.glob(os.path.join(scan_dir, "*.nii.gz")))

        for scan_path in scans:
            # Extract scan identifier (e.g., "train_01" from "microUS_train_01.nii.gz")
            filename = os.path.basename(scan_path)
            identifier = filename.replace("microUS_", "").replace(".nii.gz", "")

            # Find corresponding annotation
            annotation_path = os.path.join(
                annotation_dir, f"expert_annotation_{identifier}.nii.gz"
            )

            if os.path.exists(annotation_path):
                self.samples.append({"image": scan_path, "mask": annotation_path})

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load 3D volumes
        img_vol = nib.load(s["image"]).get_fdata().astype(np.float32)  # HxWxD or DxHxW
        msk_vol = nib.load(s["mask"]).get_fdata().astype(np.int32)  # HxWxD or DxHxW

        # Normalize shapes to DxHxW (slices first)
        if img_vol.shape[2] < img_vol.shape[0]:  # D is smallest → already DxHxW
            pass
        else:  # HxWxD → transpose to DxHxW
            img_vol = np.transpose(img_vol, (2, 0, 1))
            msk_vol = np.transpose(msk_vol, (2, 0, 1))

        frames, masks = [], []
        for i in range(img_vol.shape[0]):
            sl = normalise(img_vol[i], percentile_low=0.5, percentile_high=99.5)
            msk = (msk_vol[i] > 0).astype(np.uint8)

            # Skip empty slices during training
            if self.do_augment and msk.sum() == 0:
                continue

            sl = resize_and_pad(sl, self.image_size)
            msk = resize_mask(msk, self.image_size)

            if self.do_augment:
                sl, msk = augment(sl, msk, self.image_size)

            frames.append(sl)
            masks.append(msk)

        # Ensure we have at least one frame
        if len(frames) == 0:
            blank = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            frames, masks = [blank], [blank.astype(np.uint8)]

        return self._pack(frames, masks, self.prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CVC-ClinicDB  (Endoscopy polyp)
# ═══════════════════════════════════════════════════════════════════════════════


class CVCClinicDBDataset(MedicalSequenceDataset):
    """
    CVC-ClinicDB layout:
        dataset/CVC-ClinicDB/
            train/
                sequences/
                    sequence1/
                        Original/   *.png
                        Ground Truth/ *.png
            test/
                ...

    Alternative flat layout (if downloaded directly):
        CVC-ClinicDB/
            Original/  612 images
            Ground Truth/ 612 masks
    We auto-detect the layout.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train_ratio: float = 0.62,  # 18/(18+11) ≈ 0.62
        image_size: int = 1024,
        augment: bool = False,
    ):
        super().__init__(image_size, augment)
        self.prompt = TEXT_PROMPTS["cvc"]["polyp"]
        self._build_samples(root, split, train_ratio)

    def _build_samples(self, root: str, split: str, train_ratio: float):
        # ── Layout A: split subdirs ──────────────────────────────────────────
        split_dir = os.path.join(root, split)
        if os.path.exists(split_dir):
            seq_dirs = sorted(
                glob.glob(os.path.join(split_dir, "sequences", "*"))
                + glob.glob(os.path.join(split_dir, "*"))
            )
            for sdir in seq_dirs:
                orig_dir = (
                    os.path.join(sdir, "Original")
                    if os.path.exists(os.path.join(sdir, "Original"))
                    else sdir
                )
                gt_dir = (
                    os.path.join(sdir, "Ground Truth")
                    if os.path.exists(os.path.join(sdir, "Ground Truth"))
                    else sdir.replace("Original", "Ground Truth")
                )
                imgs = sorted(
                    glob.glob(os.path.join(orig_dir, "*.png"))
                    + glob.glob(os.path.join(orig_dir, "*.jpg"))
                )
                masks = sorted(
                    glob.glob(os.path.join(gt_dir, "*.png"))
                    + glob.glob(os.path.join(gt_dir, "*.jpg"))
                )
                if len(imgs) > 0 and len(imgs) == len(masks):
                    self.samples.append({"frames": imgs, "masks": masks})
            return

        # ── Layout B: flat Original / Ground Truth ───────────────────────────
        orig_dir = os.path.join(root, "Original")
        gt_dir = os.path.join(root, "Ground Truth")
        if not os.path.exists(orig_dir):
            orig_dir = os.path.join(root, "images")
            gt_dir = os.path.join(root, "masks")

        imgs = sorted(
            glob.glob(os.path.join(orig_dir, "*.png"))
            + glob.glob(os.path.join(orig_dir, "*.tif"))
        )
        masks = sorted(
            glob.glob(os.path.join(gt_dir, "*.png"))
            + glob.glob(os.path.join(gt_dir, "*.tif"))
        )

        # Pair up; group into fake sequences of ~35 frames (≈612/18)
        pairs = list(zip(imgs, masks))
        n_train = int(len(pairs) * train_ratio)
        pairs = pairs[:n_train] if split == "train" else pairs[n_train:]

        seq_len = 35
        for i in range(0, len(pairs), seq_len):
            chunk = pairs[i : i + seq_len]
            self.samples.append(
                {
                    "frames": [p[0] for p in chunk],
                    "masks": [p[1] for p in chunk],
                }
            )

    def _load_image(self, path: str) -> np.ndarray:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        # Collapse to grayscale for single-channel processing
        return img.mean(axis=-1)

    def _load_mask(self, path: str) -> np.ndarray:
        msk = np.array(Image.open(path).convert("L"), dtype=np.float32)
        return (msk > 127).astype(np.uint8)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames, masks = [], []
        for fpath, mpath in zip(s["frames"], s["masks"]):
            fr = self._load_image(fpath)
            msk = self._load_mask(mpath)
            fr = resize_and_pad(fr, self.image_size)
            msk = resize_mask(msk, self.image_size)
            if self.do_augment:
                fr, msk = augment(fr, msk, self.image_size)
            frames.append(fr)
            masks.append(msk)
        return self._pack(frames, masks, self.prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════


def get_dataset(
    name: str,
    root: str,
    split: str = "train",
    image_size: int = 1024,
    augment: bool = False,
    **kwargs,
) -> MedicalSequenceDataset:
    """
    Convenience factory.  `name` can be one of:
        "acdc_lv", "acdc_rv", "acdc_myo",
        "spleen", "prostate", "cvc"
    """
    name = name.lower()
    if name.startswith("acdc"):
        struct = name.split("_")[1] if "_" in name else "lv"
        return ACDCDataset(
            root,
            structure=struct,
            split=split,
            image_size=image_size,
            augment=augment,
            **kwargs,
        )
    elif name == "spleen":
        return SpleenDataset(
            root, split=split, image_size=image_size, augment=augment, **kwargs
        )
    elif name == "prostate":
        return ProstateDataset(
            root, split=split, image_size=image_size, augment=augment, **kwargs
        )
    elif name == "cvc":
        return CVCClinicDBDataset(
            root, split=split, image_size=image_size, augment=augment, **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ── Collate for variable-length sequences ─────────────────────────────────────


def collate_sequences(batch: List[dict]) -> dict:
    """
    Custom collate: sequences have different T, so we keep them as a list.
    Returns:
        images  : list of (Tx3xHxW) tensors
        masks   : list of (TxHxW)   tensors
        prompts : list of str
    """
    return {
        "images": [b["images"] for b in batch],
        "masks": [b["masks"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
    }
