from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class PPISample:
    seq_a: str
    seq_b: str
    label: float


class PPIDataset(Dataset):
    """Simple dataset from in-memory lists or CSV-like tuples."""

    def __init__(self, samples: List[PPISample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str, torch.Tensor]:
        s = self.samples[idx]
        y = torch.tensor([float(s.label)], dtype=torch.float32)
        return s.seq_a, s.seq_b, y

