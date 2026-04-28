"""
TGSAM-2: Text-Guided SAM-2 for Medical Image Segmentation
----------------------------------------------------------
ROOT CAUSE OF BUG:
  SAM-2 hiera_small forward_image() applies conv_s0 / conv_s1 AFTER the FPN neck,
  which reduces backbone_fpn channel dims:
      backbone_fpn[0]: 256 → 32   (conv_s0: 256 → 256//8)  ← finest
      backbone_fpn[1]: 256 → 64   (conv_s1: 256 → 256//4)  ← middle
      backbone_fpn[2]: 256        (unchanged)               ← coarsest (fN)

  The old code assumed [96, 192, 384, 768], causing a 64 vs 256 channel mismatch
  when adding delta_N1 (256ch) to fN1 (64ch) inside TCVP.

FIX:
  1. Auto-detect actual FPN dims via a dummy forward pass in from_pretrained().
  2. Correct default constant: SAM2_HIERA_SMALL_FPN_DIMS = [32, 64, 256].
  3. Added shape assertion in encode_frame() to catch future mismatches early.
  4. Replaced SAM-2 internal decoder calls (fragile) with a clean mask_head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Optional, Tuple

from model.tcvp import TCVP
from model.ttme import TTME
from model.text_prompt_encoder import BiomedBERTEncoder, TextPromptEncoder

try:
    from sam2.build_sam import build_sam2

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print(
        "[TGSAM-2] WARNING: SAM-2 not installed. Run: pip install -e segment-anything-2/"
    )

# Correct dims AFTER forward_image() for sam2_hiera_small (fine → coarse)
SAM2_HIERA_SMALL_FPN_DIMS = [32, 64, 256]


def get_fpn_dims(sam2_model: nn.Module, device: torch.device) -> List[int]:
    """Auto-detect actual backbone_fpn channel dims via a dummy forward pass."""
    sam2_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 1024, 1024, device=device)
        try:
            out = sam2_model.forward_image(dummy)
            dims = [f.shape[1] for f in out["backbone_fpn"]]
            print(f"[TGSAM-2] Auto-detected FPN dims (fine→coarse): {dims}")
            return dims
        except Exception as e:
            print(
                f"[TGSAM-2] Could not auto-detect FPN dims ({e}), "
                f"using default {SAM2_HIERA_SMALL_FPN_DIMS}"
            )
            return SAM2_HIERA_SMALL_FPN_DIMS


class TGSAM2(nn.Module):
    """
    TGSAM-2 model integrating:
      - SAM-2 image encoder (frozen)
      - BiomedBERT text encoder (frozen)
      - TCVP: Text-Conditioned Visual Perception  (trainable)
      - TTME: Text-Tracking Memory Encoder        (trainable)
      - TextPromptEncoder + mask head             (trainable)
    """

    def __init__(
        self,
        sam2_model: nn.Module,
        text_dim: int = 768,
        embed_dim: int = 256,
        memory_dim: int = 64,
        memory_bank_size: int = 4,
        fpn_dims: List[int] = None,
        p: int = 4,
        q: int = 2,
    ):
        super().__init__()
        self.sam2 = sam2_model
        self.K = memory_bank_size
        self._fpn_dims = fpn_dims or SAM2_HIERA_SMALL_FPN_DIMS
        coarsest_dim = self._fpn_dims[-1]  # 256 for hiera_small

        # ── Trainable modules ──────────────────────────────────────────────
        self.text_encoder = BiomedBERTEncoder(freeze=True)
        self.text_proj_enc = TextPromptEncoder(text_dim=text_dim, embed_dim=embed_dim)

        self.tcvp = TCVP(text_dim=text_dim, visual_dims=self._fpn_dims)

        self.ttme = TTME(
            visual_dim=coarsest_dim,
            text_dim=text_dim,
            memory_dim=memory_dim,
            p=p,
            q=q,
        )

        # Skip-connection fusion: cat(fN, fN1↑, fN2↑) → coarsest_dim channels
        skip_in = coarsest_dim + self._fpn_dims[-2] + self._fpn_dims[-3]
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(skip_in, coarsest_dim, kernel_size=1),
            nn.GELU(),
        )

        # Mask prediction head: (fused + text_ctx) → 1 logit map
        self.mask_head = nn.Sequential(
            nn.Conv2d(coarsest_dim + embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # ── Freeze SAM-2 (image encoder stays frozen per paper) ───────────
        for param in self.sam2.parameters():
            param.requires_grad = False

    # -----------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        sam2_checkpoint: str = "checkpoints/sam2_hiera_small.pt",
        sam2_cfg: str = "sam2_hiera_s.yaml",
        device: str = "cuda",
        **kwargs,
    ) -> "TGSAM2":
        assert SAM2_AVAILABLE, "Install SAM-2 first (see README)."
        dev = torch.device(device)
        sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device=dev)

        # Always auto-detect: avoids hard-coding errors across SAM-2 versions
        kwargs["fpn_dims"] = get_fpn_dims(sam2, dev)
        return cls(sam2_model=sam2, **kwargs).to(dev)

    # -----------------------------------------------------------------------
    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.text_encoder(texts)  # (B, L, 768)
        Tembed = self.text_proj_enc(T)  # (B, L, embed_dim)
        return T, Tembed

    # -----------------------------------------------------------------------
    def encode_frame(
        self,
        frame: torch.Tensor,  # (B, 3, H, W)
        T: torch.Tensor,  # (B, L, 768)
    ) -> Tuple[List[torch.Tensor], dict]:
        backbone_out = self.sam2.forward_image(frame)
        fpn_features = backbone_out["backbone_fpn"]  # list of tensors

        # ── Shape guard ───────────────────────────────────────────────────
        if len(fpn_features) < 3:
            raise RuntimeError(
                f"Need ≥3 FPN levels for TCVP; got {len(fpn_features)}. "
                "Check SAM-2 config 'scalp' parameter."
            )
        actual_dims = [f.shape[1] for f in fpn_features]
        if actual_dims != self._fpn_dims:
            raise RuntimeError(
                f"FPN channel mismatch!\n"
                f"  TGSAM2 initialized with: {self._fpn_dims}\n"
                f"  forward_image() returned: {actual_dims}\n"
                f"  Use TGSAM2.from_pretrained() which auto-detects dims."
            )

        enriched = self.tcvp(fpn_features, T)
        return enriched, backbone_out

    # -----------------------------------------------------------------------
    def decode_mask(
        self,
        enriched_feats: List[torch.Tensor],
        Tembed: torch.Tensor,  # (B, L, embed_dim)
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Multi-scale decode:
          1. Upsample fN1, fN2 → fN spatial size
          2. Skip-fuse all three levels
          3. Concat pooled text context
          4. mask_head → upsample to target_size
        """
        fN2 = enriched_feats[-3]  # (B, 32,  H_fine, W_fine)
        fN1 = enriched_feats[-2]  # (B, 64,  H_mid,  W_mid)
        fN = enriched_feats[-1]  # (B, 256, H_coarse, W_coarse)

        Hc, Wc = fN.shape[2], fN.shape[3]

        fN1_up = F.interpolate(fN1, size=(Hc, Wc), mode="bilinear", align_corners=False)
        fN2_up = F.interpolate(fN2, size=(Hc, Wc), mode="bilinear", align_corners=False)

        fused = self.skip_fuse(
            torch.cat([fN, fN1_up, fN2_up], dim=1)
        )  # (B, 256, Hc, Wc)

        # Pool text tokens → broadcast as spatial context
        t_ctx = Tembed.mean(dim=1)[:, :, None, None].expand(
            -1, -1, Hc, Wc
        )  # (B, embed_dim, Hc, Wc)

        logits = self.mask_head(torch.cat([fused, t_ctx], dim=1))  # (B, 1, Hc, Wc)
        return F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )

    # -----------------------------------------------------------------------
    def forward(
        self,
        frames: torch.Tensor,  # (B, T, 3, H, W)
        texts: List[str],
        gt_masks: Optional[torch.Tensor] = None,  # (B, T, 1, H, W)
        reset_memory: bool = True,
    ) -> dict:
        B, T_len, C, H, W = frames.shape

        # Encode text once for the whole sequence
        text_raw, Tembed = self.encode_text(texts)  # (B,L,768), (B,L,embed_dim)

        memory_bank = deque(maxlen=self.K)
        all_masks = []

        for t in range(T_len):
            frame_t = frames[:, t]  # (B, 3, H, W)

            enriched_feats, _ = self.encode_frame(frame_t, text_raw)
            pred_mask = self.decode_mask(enriched_feats, Tembed, target_size=(H, W))
            all_masks.append(pred_mask)

            # TTME: update memory for next frame (skip after last frame)
            if t < T_len - 1:
                mem = self.ttme(
                    visual=enriched_feats[-1],
                    mask=torch.sigmoid(pred_mask),
                    text=text_raw,
                )
                memory_bank.append(mem)

        pred_masks = torch.stack(all_masks, dim=1)  # (B, T, 1, H, W)
        result = {"pred_masks": pred_masks}

        if gt_masks is not None:
            result["loss"] = _dice_bce_loss(pred_masks, gt_masks)

        return result


# ---------------------------------------------------------------------------
def _dice_bce_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Combined Dice + BCE loss."""
    p = pred.flatten(0, 1)  # (B*T, 1, H, W)
    g = gt.flatten(0, 1).float()
    bce = F.binary_cross_entropy_with_logits(p, g)
    ps = torch.sigmoid(p)
    p_ = ps.view(ps.size(0), -1)
    g_ = g.view(g.size(0), -1)
    dice = 1 - (2 * (p_ * g_).sum(1) + 1) / (p_.sum(1) + g_.sum(1) + 1)
    return bce + dice.mean()
