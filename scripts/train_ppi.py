import argparse
from typing import List

import torch
from torch.utils.data import DataLoader

from src.ppi.model import SECAIModel, FocalLoss
from src.ppi.data import PPISample, PPIDataset


def parse_args():
    p = argparse.ArgumentParser("Train SE-CAI PPI model")
    p.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze_esm", action="store_true")
    return p.parse_args()


def build_dummy_dataset() -> PPIDataset:
    samples: List[PPISample] = [
        PPISample("MKTIIALSYIFCLVFADYKDDDDK", "MNSNQKQKDGKKKKKKKKKK", 1.0),
        PPISample("ACDEFGHIKLMNPQRSTVWY", "WYVTSRQPONMLKIHGFEDCA", 0.0),
    ]
    return PPIDataset(samples)


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dummy_dataset()
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: list(zip(*batch)))

    model = SECAIModel(esm_model_name=args.esm_model, freeze_esm=args.freeze_esm, device=device).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for seq_a_list, seq_b_list, labels in dl:
            labels = labels.to(device)
            logits = model(seq_a_list, seq_b_list)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"epoch={epoch+1} loss={loss.item():.4f}")

    print("Training finished.")


if __name__ == "__main__":
    train()

