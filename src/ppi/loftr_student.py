import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentPPIWithLoFTR(nn.Module):
    def __init__(self, vocab_size: int = 26, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.backbone = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.scale = hidden_dim ** -0.5
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def token_features(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)
        feat = self.backbone(emb)
        return feat

    def interaction_map(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(feat_a.transpose(1, 2), feat_b) * self.scale

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor):
        fa = self.token_features(seq_a)
        fb = self.token_features(seq_b)
        inter = self.interaction_map(fa, fb)
        pa = F.adaptive_max_pool1d(fa, 1).squeeze(-1)
        pb = F.adaptive_max_pool1d(fb, 1).squeeze(-1)
        fused = torch.cat([pa, pb], dim=1)
        logits = self.classifier(fused)
        return logits, inter

