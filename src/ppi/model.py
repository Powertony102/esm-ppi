import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from .esm_encoder import ESMEncoder
from .attention import CrossAttentionBlock


def masked_mean_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Concatenate mean and max pooling over valid positions.
    """
    mask = mask.unsqueeze(-1).type_as(x)
    sum_embeddings = torch.sum(x * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pool = sum_embeddings / sum_mask

    x_masked = x.masked_fill(mask == 0, -1e9)
    max_pool, _ = torch.max(x_masked, dim=1)
    return torch.cat([mean_pool, max_pool], dim=1)


class SECAIModel(nn.Module):
    """
    Siamese ESM-2 Cross-Attention Interaction model for PPI.
    """

    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        freeze_esm: bool = True,
        num_heads: int = 8,
        lora_r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.1,
        device: Optional[Union[torch.device, str]] = None,
        max_len: int = 1024,
    ):
        super().__init__()
        if isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device
        self.encoder = ESMEncoder(model_name=esm_model_name, device=dev, max_len=max_len)
        hidden_dim = self.encoder.hidden_dim or 1280

        if freeze_esm:
            for p in self.encoder.model.parameters():
                p.requires_grad = False

        self.cross_attn_A2B = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout, lora_r=lora_r, lora_alpha=lora_alpha)
        self.cross_attn_B2A = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout, lora_r=lora_r, lora_alpha=lora_alpha)

        feature_dim = hidden_dim * 2
        fusion_input_dim = feature_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, seqs_a: List[str], seqs_b: List[str]) -> torch.Tensor:
        feat_A, mask_A, feat_B, mask_B = self.encoder.encode_pair_batch(seqs_a, seqs_b)

        pad_mask_A = ~mask_A
        pad_mask_B = ~mask_B

        feat_A_enh = self.cross_attn_A2B(query=feat_A, key_value=feat_B, key_padding_mask=pad_mask_B)
        feat_B_enh = self.cross_attn_B2A(query=feat_B, key_value=feat_A, key_padding_mask=pad_mask_A)

        vec_A = masked_mean_max_pool(feat_A_enh, mask_A)
        vec_B = masked_mean_max_pool(feat_B_enh, mask_B)

        diff = torch.abs(vec_A - vec_B)
        prod = vec_A * vec_B
        fused = torch.cat([vec_A, vec_B, diff, prod], dim=1)

        logits = self.classifier(fused)
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
