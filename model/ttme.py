"""
Text-Tracking Memory Encoder (TTME)
------------------------------------
Implements Section 2.3 of TGSAM-2.

Standard SAM-2 memory encoder:
  ŷ's = Act(LN(Conv(ŷs)))          repeated p times
  f'N,s = Conv(ŷ's) + fN,s
  Ms = ConvBlock(f'N,s)

TTME adds text injection:
  Ms = Act(PwConv(LN(DwConv(f'N,s))) + W·T)    repeated q times

Key parameters from paper:
  p = 4  (mask downsampling repeats)
  q = 2  (memory feature repeats with text)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDownsampler(nn.Module):
    """Repeatedly downsample the predicted mask via Conv+LN+Act (p times)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 256, p: int = 4):
        super().__init__()
        self.p = p
        layers = []
        ch = in_ch
        for i in range(p):
            next_ch = out_ch if i == p - 1 else max(in_ch * (2 ** (i + 1)), 16)
            layers.append(nn.Conv2d(ch, next_ch, kernel_size=3, stride=2, padding=1))
            layers.append(
                nn.LayerNorm([next_ch, 1, 1])
            )  # placeholder; applied per-sample
            layers.append(nn.GELU())
            ch = next_ch

        # Use a simpler flat structure so LN can be applied flexibly
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        ch = in_ch
        for i in range(p):
            next_ch = out_ch if i == p - 1 else max(in_ch * (2 ** (i + 1)), 16)
            self.convs.append(
                nn.Conv2d(ch, next_ch, kernel_size=3, stride=2, padding=1)
            )
            self.norms.append(
                nn.GroupNorm(1, next_ch)
            )  # GroupNorm(1) ≈ LayerNorm for spatial
            self.acts.append(nn.GELU())
            ch = next_ch

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """mask: (B, 1, H, W) → (B, out_ch, H/2^p, W/2^p)"""
        x = mask.float()
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            x = act(norm(conv(x)))
        return x


class DepthwiseSeparableBlock(nn.Module):
    """
    One iteration of the TTME memory block:
      out = Act(PwConv(LN(DwConv(x))) + W·T)

    where DwConv = depthwise conv (preserves channels)
          PwConv = pointwise 1x1 conv (maps channels)
          W·T    = linear projection of text features, broadcast spatially
    """

    def __init__(self, visual_dim: int, text_dim: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            visual_dim, visual_dim, kernel_size=3, padding=1, groups=visual_dim
        )
        self.ln = nn.GroupNorm(1, visual_dim)
        self.pw_conv = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, H, W)
        text: (B, L, C_text) → pooled to (B, C)
        """
        # Text projection: pool over L tokens then broadcast spatially
        t = self.text_proj(text.mean(dim=1))  # (B, C)
        t = t[:, :, None, None]  # (B, C, 1, 1)

        out = self.pw_conv(self.ln(self.dw_conv(x))) + t
        return self.act(out)


class TTME(nn.Module):
    """
    Text-Tracking Memory Encoder.

    Replaces SAM-2's default memory encoder for one previous frame.
    Takes visual features fN, predicted mask ŷ, and text T.

    Steps:
      1. Downsample ŷ → ŷ' (p=4 times via Conv+Norm+Act)
      2. Add: f'N = Conv(ŷ') + fN
      3. Iteratively refine with text (q=2 times):
           f'N = Act(PwConv(LN(DwConv(f'N))) + W·T)
      4. Final projection to memory dim

    Args:
        visual_dim: channel dimension of the coarsest FPN feature (fN)
        text_dim:   dimension of text token embeddings
        memory_dim: output channel dim stored in the memory bank
        p:          mask downsampling repeats (paper: 4)
        q:          text-conditioned block repeats (paper: 2)
    """

    def __init__(
        self,
        visual_dim: int = 256,
        text_dim: int = 768,
        memory_dim: int = 64,
        p: int = 4,
        q: int = 2,
    ):
        super().__init__()
        self.q = q

        # Step 1: mask downsampling
        self.mask_downsampler = MaskDownsampler(in_ch=1, out_ch=visual_dim, p=p)

        # Step 2: fuse mask features into visual features
        self.mask_fuse = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)

        # Step 3: q iterations of text-conditioned depthwise blocks
        self.text_blocks = nn.ModuleList(
            [
                DepthwiseSeparableBlock(visual_dim=visual_dim, text_dim=text_dim)
                for _ in range(q)
            ]
        )

        # Step 4: project to memory dim
        self.out_proj = nn.Conv2d(visual_dim, memory_dim, kernel_size=1)

        # Positional embedding (m_pos in paper Fig. 3b)
        # Learned — same spatial size as memory features
        self.m_pos = nn.Parameter(torch.zeros(1, memory_dim, 1, 1))

    def forward(
        self,
        visual: torch.Tensor,  # (B, C, H, W) — coarsest FPN feature fN of a previous frame
        mask: torch.Tensor,  # (B, 1, H_orig, W_orig) — predicted mask for that frame
        text: torch.Tensor,  # (B, L, C_text) — text token embeddings
    ) -> torch.Tensor:
        """Returns memory feature: (B, memory_dim, H_mem, W_mem)."""

        # 1. Downsample mask to match visual spatial size
        mask_ds = self.mask_downsampler(mask)  # (B, C, H, W)

        # 2. Fuse mask into visual features
        fused = self.mask_fuse(mask_ds) + visual  # (B, C, H, W)

        # 3. Iterative text-conditioned refinement
        x = fused
        for block in self.text_blocks:
            x = block(x, text)  # (B, C, H, W)

        # 4. Project to memory dim + positional bias
        mem = self.out_proj(x)  # (B, memory_dim, H, W)
        mem = mem + self.m_pos.expand_as(mem)

        return mem
