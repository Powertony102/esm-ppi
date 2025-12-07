from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv

import torch
from torch.utils.data import Dataset


AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))


def sanitize_sequence(seq: str) -> str:
    s = seq.strip().upper()
    return "".join([c if c in AA20 else "X" for c in s])


@dataclass
class PPISample:
    protein_a: Optional[str]
    protein_b: Optional[str]
    seq_a: str
    seq_b: str
    label: Optional[float]


class PPIDataset(Dataset):
    """Simple dataset from in-memory list of PPISample."""

    def __init__(self, samples: List[PPISample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str, Optional[torch.Tensor], Optional[str], Optional[str]]:
        s = self.samples[idx]
        y = None if s.label is None else torch.tensor([float(s.label)], dtype=torch.float32)
        return s.seq_a, s.seq_b, y, s.protein_a, s.protein_b


class KagglePPIDataset(PPIDataset):
    """Dataset that reads Kaggle CSV with columns: protein_A, protein_B, sequence_A, sequence_B, [label]."""

    def __init__(self, csv_path: str, has_label: bool = True):
        samples: List[PPISample] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pa = row.get("protein_A")
                pb = row.get("protein_B")
                sa = sanitize_sequence(row.get("sequence_A", ""))
                sb = sanitize_sequence(row.get("sequence_B", ""))
                label = float(row["label"]) if has_label and "label" in row and row["label"] != "" else None
                samples.append(PPISample(protein_a=pa, protein_b=pb, seq_a=sa, seq_b=sb, label=label))
        super().__init__(samples)


def kaggle_collate(batch):
    seq_a_list = [a for a, b, y, pa, pb in batch]
    seq_b_list = [b for a, b, y, pa, pb in batch]
    labels = [y for a, b, y, pa, pb in batch]
    ids = [(pa, pb) for a, b, y, pa, pb in batch]
    labels_tensor = None
    if labels[0] is not None:
        labels_tensor = torch.stack(labels, dim=0)
    return seq_a_list, seq_b_list, labels_tensor, ids
