import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adapter for linear projections.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.A = nn.Linear(in_features, r, bias=False)
            self.B = nn.Linear(r, out_features, bias=False)
        else:
            self.A = None
            self.B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0:
            return torch.zeros_like(x)
        return self.B(self.A(self.dropout(x))) * self.scaling


class CrossAttentionBlock(nn.Module):
    """
    Multi-head cross attention with optional LoRA on Query/Value projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.lora_q = LoRAAdapter(hidden_dim, hidden_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.lora_v = LoRAAdapter(hidden_dim, hidden_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, qlen, _ = query.shape
        _, klen, _ = key_value.shape

        q = self.q_proj(query) + self.lora_q(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value) + self.lora_v(key_value)

        q = q.view(bsz, qlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, klen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, klen, self.num_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # [b, 1, 1, klen]
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, qlen, self.hidden_dim)
        out = self.o_proj(attn_out)
        out = self.norm(query + self.dropout(out))
        return out

