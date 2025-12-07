import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.ppi.model import SECAIModel, FocalLoss
from src.ppi.data import KagglePPIDataset, kaggle_collate


def parse_args():
    p = argparse.ArgumentParser("Train SE-CAI PPI model on Kaggle dataset")
    p.add_argument("--data_dir", default="kaggle_dataset")
    p.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze_esm", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> Tuple[float, float, float, float]:
    probs = torch.sigmoid(logits).detach()
    preds = (probs > threshold).int()
    y = labels.int()
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec_den = tp + fp
    rec_den = tp + fn
    prec = tp / prec_den if prec_den > 0 else 0.0
    rec = tp / rec_den if rec_den > 0 else 0.0
    f1_den = prec + rec
    f1 = 2 * prec * rec / f1_den if f1_den > 0 else 0.0
    return acc, prec, rec, f1


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(args.data_dir, "train.csv")
    valid_csv = os.path.join(args.data_dir, "valid.csv")

    train_ds = KagglePPIDataset(train_csv, has_label=True)
    valid_ds = KagglePPIDataset(valid_csv, has_label=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=kaggle_collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=kaggle_collate)

    model = SECAIModel(esm_model_name=args.esm_model, freeze_esm=args.freeze_esm, device=device).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for seq_a, seq_b, labels, _ in train_dl:
            labels = labels.to(device)
            logits = model(seq_a, seq_b)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"epoch={epoch+1} train_loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            n = 0
            all_logits = []
            all_labels = []
            for seq_a, seq_b, labels, _ in valid_dl:
                labels = labels.to(device)
                logits = model(seq_a, seq_b)
                l = criterion(logits, labels)
                total_loss += l.item() * labels.size(0)
                n += labels.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
            if n:
                val_loss = total_loss / n
                logits_cat = torch.cat(all_logits, dim=0)
                labels_cat = torch.cat(all_labels, dim=0)
                acc, prec, rec, f1 = metrics_from_logits(logits_cat, labels_cat, args.threshold)
                print(f"val_loss={val_loss:.4f} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    train()
