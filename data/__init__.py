"""
TGSAM-2 Dataset Module

Supports:
  - ACDC (MRI cardiac segmentation)
  - MSD Spleen (CT)
  - Micro-Ultrasound Prostate
  - CVC-ClinicDB (Endoscopy polyp)
"""

from .medical_dataset import (
    MedicalSequenceDataset,
    ACDCDataset,
    SpleenDataset,
    ProstateDataset,
    CVCClinicDBDataset,
    get_dataset,
    collate_sequences,
)

__all__ = [
    "MedicalSequenceDataset",
    "ACDCDataset",
    "SpleenDataset",
    "ProstateDataset",
    "CVCClinicDBDataset",
    "get_dataset",
    "collate_sequences",
]
