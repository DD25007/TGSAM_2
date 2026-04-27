"""
Text-Conditioned Visual Perception (TCVP) Module
-------------------------------------------------
Implements Section 2.2 of TGSAM-2.

Architecture:
  fN  += MHCA(T, fN)                     # cross-attention: text as Q, vision as K/V
  fN-1 += GELU(DeConv(fN))              # upsample to finer scale
  fN-2 += DeConv(fN-1)                  # upsample to finest scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention.
    Text features T act as Query.
    Visual features fN act as Key and Value.
    """

    def __init__(
        self, text_dim: int, visual_dim: int, num_heads: int = 8, dropout: float = 0.0
    ):
        super().__init__()
        assert visual_dim % num_heads == 0, "visual_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Project text → Q
        self.to_q = nn.Linear(text_dim, visual_dim)
        # Project visual → K, V
        self.to_k = nn.Linear(visual_dim, visual_dim)
        self.to_v = nn.Linear(visual_dim, visual_dim)

        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text:   (B, L, C_text)   — L text tokens
            visual: (B, H*W, C_vis)  — flattened spatial features

        Returns:
            attended visual features: (B, H*W, C_vis)
        """
        B, L, _ = text.shape
        N = visual.shape[1]

        Q = self.to_q(text)  # (B, L, C_vis)
        K = self.to_k(visual)  # (B, N, C_vis)
        V = self.to_v(visual)  # (B, N, C_vis)

        # Split into heads
        def split_heads(x):
            return rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Attention: Q over K
        attn = torch.einsum("bhld,bhnd->bhln", Q, K) * self.scale  # (B, h, L, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate V with attention weights
        out = torch.einsum("bhln,bhnd->bhld", attn, V)  # (B, h, L, C/h)
        out = rearrange(out, "b h l d -> b l (h d)")  # (B, L, C_vis)

        # Pool over text tokens → one context vector per spatial location
        # We return context broadcast to visual resolution
        # Strategy: average-pool text output then broadcast
        ctx = out.mean(dim=1, keepdim=True).expand(-1, N, -1)  # (B, N, C_vis)
        return self.out_proj(ctx)


class TCVP(nn.Module):
    """
    Text-Conditioned Visual Perception module.

    Given multi-scale visual features F = {f1, f2, ..., fN}  (coarse→fine order,
    fN is the deepest / coarsest) and text embeddings T, this module:

      1. Applies cross-attention between T (query) and fN (key/value)
      2. Upsamples enriched fN to fN-1 via GELU + DeConv
      3. Upsamples fN-1 to fN-2 via DeConv

    Paper notation (coarsest = last level N):
        fN  += MHCA(T, fN)
        fN-1 += GELU(DeConv(fN))
        fN-2 += DeConv(fN-1)
    """

    def __init__(
        self,
        text_dim: int,
        visual_dims: list = None,  # [C_finest, ..., C_coarsest] or None for lazy init
        num_heads: int = 8,
    ):
        """
        Args:
            text_dim:     Dimension of BiomedBERT token embeddings (768 for base).
            visual_dims:  List of channel dims for each FPN level from fine to coarse.
                          If None, will be auto-detected from first forward pass.
            num_heads:    Number of attention heads.
        """
        super().__init__()
        self.text_dim = text_dim
        self.num_heads = num_heads
        self._initialized = False

        # If dims are provided, initialize eagerly
        if visual_dims is not None:
            self._init_modules(visual_dims)
        else:
            # Lazy initialization - will initialize on first forward
            self._lazy_visual_dims = None

    def _init_modules(self, visual_dims):
        """Initialize TCVP modules with specific visual dimensions."""
        self.N = len(visual_dims)
        C_coarsest = visual_dims[-1]  # fN
        C_2nd_coarse = visual_dims[-2]  # fN-1
        C_3rd_coarse = visual_dims[-3]  # fN-2

        # Cross-attention at coarsest level
        self.cross_attn = MultiHeadCrossAttention(
            text_dim=self.text_dim,
            visual_dim=C_coarsest,
            num_heads=self.num_heads,
        )

        # DeConv: fN  → fN-1 scale  (upsample x2, adjust channels)
        self.deconv_N_to_N1 = nn.Sequential(
            nn.ConvTranspose2d(C_coarsest, C_2nd_coarse, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # DeConv: fN-1 → fN-2 scale  (upsample x2, adjust channels)
        self.deconv_N1_to_N2 = nn.ConvTranspose2d(
            C_2nd_coarse, C_3rd_coarse, kernel_size=2, stride=2
        )

        # Layer norms
        self.norm_N = nn.LayerNorm(C_coarsest)
        self.norm_N1 = nn.LayerNorm(C_2nd_coarse)
        self.norm_N2 = nn.LayerNorm(C_3rd_coarse)
        self._initialized = True

    def forward(
        self,
        features: list,  # [f1, f2, ..., fN] fine→coarse, each (B, C_i, H_i, W_i)
        text_embeddings: torch.Tensor,  # (B, L, C_text)
    ) -> list:
        """
        Returns updated feature list [f1, f2, ..., fN] with the last 3 levels modified.
        """
        # Lazy initialization: detect actual FPN dims from first forward
        if not self._initialized:
            actual_dims = [f.shape[1] for f in features]
            print(f"[TCVP] Auto-detecting FPN dims from features: {actual_dims}")
            self._init_modules(actual_dims)

        fN = features[-1]  # (B, C_N, H_N, W_N)   coarsest
        fN1 = features[-2]  # (B, C_N1, H_N1, W_N1)
        fN2 = features[-3]  # (B, C_N2, H_N2, W_N2)

        B, C_N, H_N, W_N = fN.shape

        # Flatten spatial dims for attention
        fN_flat = fN.flatten(2).transpose(1, 2)  # (B, H*W, C_N)
        fN_flat = self.norm_N(fN_flat)

        # Cross-attention: text queries attend to coarsest visual features
        delta = self.cross_attn(text_embeddings, fN_flat)  # (B, H*W, C_N)
        fN_flat = fN_flat + delta
        fN_updated = fN_flat.transpose(1, 2).reshape(B, C_N, H_N, W_N)

        # Propagate to finer levels via transposed convolution
        delta_N1 = self.deconv_N_to_N1(fN_updated)  # (B, C_N1, H_N1, W_N1)
        delta_N2 = self.deconv_N1_to_N2(delta_N1)  # (B, C_N2, H_N2, W_N2)

        # Residual addition
        fN1_updated = fN1 + delta_N1
        fN2_updated = fN2 + delta_N2

        # Replace last 3 levels; leave finer levels unchanged
        updated = features[:-3] + [fN2_updated, fN1_updated, fN_updated]
        return updated
