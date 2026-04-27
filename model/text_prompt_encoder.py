"""
Text Prompt Encoder
--------------------
Implements Section 2.1 "Prompting SAM-2 with Text".

Converts BiomedBERT token features T into a sparse prompt embedding
compatible with SAM-2's mask decoder, treating text like points/boxes.

Formulation:
    Tproj  = W · T                                             (linear projection)
    Tembed = Softmax(WQ·Tproj · (WK·Tproj)^T / sqrt(D)) · Tproj  (self-attention aggregation)

Tembed is summed with positional encodings and fed to SAM-2's mask decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class BiomedBERTEncoder(nn.Module):
    """
    Wraps microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.
    Returns per-token embeddings T ∈ R^{L x 768}.
    """

    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(self.MODEL_NAME)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: list of B strings (medical descriptions)
        Returns:
            T: (B, L, 768) token embeddings (padded to max length in batch)
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(next(self.encoder.parameters()).device) for k, v in enc.items()}
        out = self.encoder(**enc)
        return out.last_hidden_state  # (B, L, 768)


class TextPromptEncoder(nn.Module):
    """
    Projects T → Tembed compatible with SAM-2's mask decoder sparse prompts.

    Steps (paper Eq. 3):
      1. Tproj = W · T   (C → D)
      2. Self-attention aggregation:
           Tembed = Softmax(WQ·Tproj · (WK·Tproj)^T / sqrt(D)) · Tproj
      3. Add positional encodings (learned, one per token position)
    """

    def __init__(
        self,
        text_dim: int = 768,  # BiomedBERT hidden size
        embed_dim: int = 256,  # SAM-2 decoder dimension D
        max_tokens: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Step 1: linear projection C → D
        self.proj = nn.Linear(text_dim, embed_dim)

        # Step 2: self-attention weights
        self.WQ = nn.Linear(embed_dim, embed_dim, bias=False)
        self.WK = nn.Linear(embed_dim, embed_dim, bias=False)

        # Step 3: positional encoding (one vector per token position)
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, embed_dim) * 0.02)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: (B, L, C_text) — raw BiomedBERT token embeddings
        Returns:
            Tembed: (B, L, D)  — to be passed to SAM-2 mask decoder as sparse prompt
        """
        B, L, _ = T.shape

        # 1. Project text features
        Tproj = self.proj(T)  # (B, L, D)

        # 2. Self-attention aggregation
        Q = self.WQ(Tproj)  # (B, L, D)
        K = self.WK(Tproj)  # (B, L, D)
        scale = self.embed_dim**-0.5
        attn = F.softmax(Q @ K.transpose(-1, -2) * scale, dim=-1)  # (B, L, L)
        Tembed = attn @ Tproj  # (B, L, D)

        # 3. Add positional encodings (truncate/pad to L)
        pos = self.pos_embed[:, :L, :]
        Tembed = Tembed + pos

        return Tembed  # (B, L, D) — treated as sparse prompt tokens
