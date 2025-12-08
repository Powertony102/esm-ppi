import torch
import torch.nn as nn


class DualBranchStudentPPI(nn.Module):
    def __init__(self, vocab_size=26, embed_dim=64, hidden_dim=128, esm_dim=1280):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.local_branch = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.semantic_branch_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim),
            nn.Conv1d(embed_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.semantic_branch_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, esm_dim),
            nn.LayerNorm(esm_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + esm_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256 * 2, 2)

    def forward_single(self, x: torch.Tensor):
        emb = self.embedding(x).permute(0, 2, 1)
        feat_local = self.local_branch(emb).squeeze(-1)
        sem_conv = self.semantic_branch_conv(emb)
        feat_sem = self.semantic_branch_proj(sem_conv)
        fused = self.fusion(torch.cat([feat_local, feat_sem], dim=1))
        return fused, feat_sem

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor):
        feat_a, sem_a = self.forward_single(seq_a)
        feat_b, sem_b = self.forward_single(seq_b)
        logits = self.classifier(torch.cat([feat_a, feat_b], dim=1))
        return logits, (sem_a, sem_b)
