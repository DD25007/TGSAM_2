# TGSAM-2: Text-Guided Medical Image Segmentation

Implementation of the MICCAI 2024 paper:  
**"TGSAM-2: Text-Guided Medical Image Segmentation using Segment Anything Model 2"**

[[Paper](https://arxiv.org/abs/2502.11093)]

---

## ✨ Overview

TGSAM-2 extends **SAM-2** with text-guided segmentation for medical imaging. Instead of clicking points, users provide natural language descriptions (e.g., "Segment the left ventricle of the heart in cardiac MRI") to segment anatomical structures across video sequences and 3D volumes.

**Key Innovation:** 
- **TCVP** (Text-Conditioned Visual Perception) - Conditions visual features on text via cross-attention
- **TTME** (Text-Tracking Memory Encoder) - Maintains consistent target tracking using text guidance

---

## 📂 Project Structure

```
project/
├── model/                              # ← Core model components
│   ├── __init__.py
│   ├── tgsam2.py                      # Main TGSAM2 model (paper §2)
│   ├── text_prompt_encoder.py         # BiomedBERT encoding (paper Eq. 3)
│   ├── tcvp.py                        # Text-Conditioned Visual Perception (paper Eq. 4)
│   └── ttme.py                        # Text-Tracking Memory Encoder (paper Eq. 5-6)
│
├── data/                               # ← Dataset loaders
│   ├── __init__.py
│   └── medical_dataset.py             # All 4 dataset implementations
│
├── utils/                              # ← Training utilities
│   ├── __init__.py
│   ├── losses.py                      # DiceLoss, BCELoss, DiceBCELoss
│   └── metrics.py                     # DSC, IoU, Sensitivity, Specificity
│
├── configs/                            # ← Training configurations (YAML)
│   ├── acdc.yaml                      # Cardiac MRI
│   ├── spleen.yaml                    # Spleen CT
│   ├── prostate.yaml                  # Prostate Ultrasound
│   └── cvc.yaml                       # Endoscopy Polyp
│
├── checkpoints/                        # ← Pre-trained models
│   └── sam2_hiera_small.pt            # (download required)
│
├── outputs/                            # ← Training outputs
│   ├── acdc/
│   ├── spleen/
│   ├── prostate/
│   └── cvc/
│
├── segment-anything-2/                 # SAM-2 submodule (installed)
│
├── train.py                            # Main training script
├── evaluate.py                         # Evaluation script
├── inference.py                        # Single-sample inference
├── verify_setup.py                     # Setup verification
├── text_prompts.py                     # Medical text descriptions
├── requirements.txt
└── README.md
```

## Datasets

The paper evaluates on 4 datasets from the **Referring Medical Image Sequence Segmentation (RMISS)** benchmark:

| Dataset           | Modality   | Train | Test | Target             |
| ----------------- | ---------- | ----- | ---- | ------------------ |
| ACDC              | MRI        | 100*  | 50*  | LV, RV, Myocardium |
| MSD Spleen        | CT         | 30*   | 11   | Spleen             |
| Micro-US Prostate | Ultrasound | 55    | 20   | Prostate           |
| CVC-ClinicDB      | Endoscopy  | 18*   | 11*  | Colorectal Polyp   |

_*Paper's reported splits (patients or sequences). Actual downloaded datasets may vary slightly._

### Download Links:
- **RMISS benchmark** (with text prompts): https://arxiv.org/abs/2502.11093
  - GitHub: https://github.com/YuanRuntian/Text-Promptable-Propagation
- **ACDC**: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- **MSD Spleen**: http://medicaldecathlon.com/
- **Micro-Ultrasound Prostate**: https://github.com/mdiwebma/microsegnet
- **CVC-ClinicDB**: https://polyp.grand-challenge.org/CVCClinicDB/

---

## 📊 Datasets

The paper evaluates on 4 diverse medical imaging modalities from the **RMISS benchmark**:

|      Dataset      |  Modality  |   Dimension    | Train | Test  |       Target       |
| :---------------: | :--------: | :------------: | :---: | :---: | :----------------: |
|       ACDC        |    MRI     | 3D (2D slices) |  100  |  50   | LV, RV, Myocardium |
|    MSD Spleen     |     CT     | 3D (2D slices) |  30   |  11   |       Spleen       |
| Micro-US Prostate | Ultrasound |    2D video    |  55   |  20   |      Prostate      |
|   CVC-ClinicDB    | Endoscopy  |    2D video    |  18   |  11   |  Colorectal Polyp  |

### Download Links

- **RMISS benchmark** (text prompts): https://github.com/YuanRuntian/Text-Promptable-Propagation
- **ACDC** (Cardiac MRI): https://www.creatis.insa-lyon.fr/Challenge/acdc/
- **MSD Spleen** (CT): http://medicaldecathlon.com/
- **Micro-US Prostate** (Ultrasound): https://github.com/mdiwebma/microsegnet
- **CVC-ClinicDB** (Endoscopy): https://polyp.grand-challenge.org/CVCClinicDB/

### Expected Directory Structure

```
dataset/
├── ACDC/
│   ├── training/
│   │   ├── patient001/
│   │   │   ├── patient001_frame01.nii          ← Image (uncompressed)
│   │   │   ├── patient001_frame01_gt.nii       ← Mask (uncompressed)
│   │   │   ├── patient001_frame12.nii
│   │   │   ├── patient001_frame12_gt.nii
│   │   │   └── Info.cfg
│   │   ├── patient002/
│   │   └── ... (up to patient100)
│   └── testing/
│       ├── patient101/
│       ├── patient102/
│       └── ... (up to patient150)
│
├── spleen/
│   └── Task09_Spleen/
│       ├── imagesTr/          (41 training images .nii.gz)
│       ├── labelsTr/          (41 training labels .nii.gz)
│       ├── imagesTs/          (20 test images .nii.gz)
│       └── dataset.json
│
├── Micro_Ultrasound_Prostate_Segmentation_Dataset/
│   ├── train/
│   │   ├── micro_ultrasound_scans/        (55 .nii.gz volumes)
│   │   │   ├── microUS_train_01.nii.gz
│   │   │   ├── microUS_train_02.nii.gz
│   │   │   └── ... (up to microUS_train_55.nii.gz)
│   │   ├── expert_annotations/            (55 .nii.gz masks)
│   │   │   ├── expert_annotation_train_01.nii.gz
│   │   │   └── ...
│   │   └── non_expert_annotations/        (55 .nii.gz alternate masks)
│   └── test/
│       ├── micro_ultrasound_scans/        (20 .nii.gz volumes)
│       │   ├── microUS_test_01.nii.gz
│       │   └── ...
│       ├── expert_annotations/            (20 .nii.gz masks)
│       ├── clinician_annotations/         (20 .nii.gz alternate masks)
│       ├── master_student_annotations/    (20 .nii.gz alternate masks)
│       └── medical_student_annotations/   (20 .nii.gz alternate masks)
│
└── CVC-ClinicDB/
    ├── Original/              (612 RGB endoscopy images .tif)
    │   ├── 1.tif
    │   ├── 100.tif
    │   ├── 101.tif
    │   └── ...
    └── Ground Truth/          (612 binary masks .tif)
        ├── 1.tif
        ├── 100.tif
        ├── 101.tif
        └── ...
```

### Actual Dataset Statistics

After loading and processing, the datasets provide:

| Dataset          | Train        | Test        | Notes                                       |
| ---------------- | ------------ | ----------- | ------------------------------------------- |
| **ACDC**         | 200 samples  | 100 samples | 100/50 patients → 2/2 slices per patient    |
| **Spleen**       | 29 samples   | 12 samples  | Auto-split (71% train) from 41 volumes      |
| **Prostate**     | 55 samples   | 20 samples  | One 3D volume = one sequence per patient    |
| **CVC-ClinicDB** | 11 sequences | 7 sequences | 612 images grouped into ~35-frame sequences |

---

## ⚙️ Installation & Setup

### 1. Install SAM-2 in Editable Mode

```bash
cd segment-anything-2
pip install -e .
cd ..
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `torch>=2.0.0` - PyTorch
- `transformers>=4.35.0` - HuggingFace (for BiomedBERT)
- `nibabel>=5.0.0` - NIfTI medical images
- `pyyaml>=6.0` - Configuration files

### 3. Download SAM-2 Checkpoint

```bash
mkdir -p checkpoints
wget -P checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
```

---

## 🚀 Quick Start

### Train on ACDC (Cardiac MRI)

```bash
python train.py --config configs/acdc.yaml --device cuda:0
```

### Train on Other Datasets

```bash
# Spleen segmentation (CT)
python train.py --config configs/spleen.yaml

# Prostate segmentation (Ultrasound)
python train.py --config configs/prostate.yaml

# Polyp segmentation (Endoscopy)
python train.py --config configs/cvc.yaml
```

### Evaluate on Test Set

```bash
python evaluate.py \
    --config configs/acdc.yaml \
    --checkpoint outputs/acdc/best_dsc_0.8763.pth \
    --device cuda:0
```

Results saved to `outputs/eval/summary.txt` and `results.json`.

### Single-Sample Inference

```bash
python inference.py \
    --checkpoint outputs/acdc/best_dsc_0.8763.pth \
    --image_path dataset/ACDC/testing/patient101/patient101_frame01.nii.gz \
    --text_prompt "Segment the left ventricle of the heart in cardiac MRI" \
    --save_output results/acdc/patient101/
```

---

## 📋 Configuration

Each config file (YAML) controls:
- **Dataset**: name, root path, image size, augmentation
- **Training**: batch size, epochs, learning rate, LR scheduler
- **Loss**: dice/BCE weights
- **Checkpointing**: save frequency, output directory

Example: `configs/acdc.yaml`

```yaml
dataset:
  name: "acdc"
  root: "dataset/ACDC"
  image_size: 1024
  augment: true

train:
  batch_size: 4
  num_epochs: 50
  learning_rate: 0.0001
  lr_schedule: "steplr"
  lr_schedule_params:
    step_size: 10
    gamma: 0.5
```

---

## 📈 Results (Paper Table 1)

Model achieves SOTA across all modalities:

|   Method    | ACDC DSC  | Spleen DSC | Prostate DSC |  CVC DSC  |
| :---------: | :-------: | :--------: | :----------: | :-------: |
|   nn-UNet   |   86.54   |   86.98    |    89.73     |   80.34   |
|  MedSAM-2   |   86.04   |   87.75    |    91.57     |   84.35   |
| **TGSAM-2** | **87.63** | **89.34**  |  **92.75**   | **85.10** |

Improvements: **+1.59%** (ACDC), **+1.59%** (Spleen), **+1.18%** (Prostate), **+0.75%** (CVC)

---

## 🏗️ Architecture

### Paper Section 2: TGSAM-2 Method

**Text Encoding (§2.1, Eq. 3):**
```
T (BiomedBERT) → T_proj (linear) → T_embed (self-attention) → SAM-2 decoder
```

**Text-Conditioned Visual Perception (§2.2, Eq. 4):**
```
f_N += MHCA(T, f_N)              # Cross-attention at coarsest level
f_N-1 += GELU(DeConv(f_N))       # Upsample with activation
f_N-2 += DeConv(f_N-1)           # Upsample to finer scale
```

**Text-Tracking Memory Encoder (§2.3, Eq. 5-6):**
```
ŷ' = downsampling(ŷ) [p=4 times]
M = text-conditioned conv blocks (q=2 times)
   = Act(PwConv(LN(DwConv(f'_N))) + W·T)
```

---

## 🔧 Training Details

**From paper (§3.2):**
- **Model**: SAM-2 Hiera-Small (46M params, frozen backbone)
- **Text Encoder**: BiomedBERT (frozen)
- **Image Size**: 1024 × 1024
- **Memory Bank Size**: K=4 frames
- **Batch Size**: 4
- **Optimizer**: Adam (lr=1e-4)
- **LR Schedule**: ×0.5 every 10 epochs
- **Loss**: Dice + BCE (0.5 each weight)
- **GPU**: RTX 3090 24GB

---

## 📚 Key References

1. **TGSAM-2 Paper**: [arXiv:2502.11093](https://arxiv.org/abs/2502.11093)
2. **SAM-2**: [Ravi et al., 2024](https://arxiv.org/abs/2408.00714)
3. **BiomedBERT**: [Gu et al., 2021](https://arxiv.org/abs/2007.15828)
4. **RMISS Benchmark**: [Yuan et al., 2025](https://github.com/YuanRuntian/Text-Promptable-Propagation)

---

## 🐛 Troubleshooting

**ImportError: No module named 'sam2'**
```bash
cd segment-anything-2
pip install -e .
cd ..
```

**CUDA Out of Memory**
- Reduce `batch_size` in config (default: 4)
- Reduce `image_size` (default: 1024)

**BiomedBERT download fails**
- Model auto-downloads from HuggingFace Hub
- Requires internet connection on first run
- Cached in `~/.cache/huggingface/`

**Missing datasets**
- Download from official sources (links above)
- Place in `dataset/` directory with correct structure

---

## 📄 License

This implementation is based on the MICCAI 2024 paper. Please cite if you use this code:

```bibtex
@inproceedings{yuan2024tgsam2,
  title={TGSAM-2: Text-Guided Medical Image Segmentation using Segment Anything Model 2},
  author={Yuan, Runtian and Zhou, Ling and Xu, Jilan and Li, Qingqiu and Chen, Mohan and Zhang, Yuejie and Feng, Rui and Zhang, Tao and Gao, Shang},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2024},
  year={2024}
}
```
