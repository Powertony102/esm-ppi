import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class StudentPPICNN(nn.Module):
    def __init__(self, vocab_size=26, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            ResidualBlock1D(embed_dim, hidden_dim, kernel_size=3),
            nn.MaxPool1d(2),
            ResidualBlock1D(hidden_dim, hidden_dim * 2, kernel_size=5),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward_one(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        feat = self.encoder(x)
        return feat.squeeze(-1)

    def forward(self, seq_a, seq_b):
        fa = self.forward_one(seq_a)
        fb = self.forward_one(seq_b)
        combined = torch.cat([fa, fb], dim=1)
        logits = self.classifier(combined)
        return logits
