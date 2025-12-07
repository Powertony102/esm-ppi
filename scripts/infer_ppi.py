import argparse
import csv
import os
import torch
from tqdm import tqdm

from src.ppi.model import SECAIModel
from src.ppi.data import KagglePPIDataset, kaggle_collate
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser("Infer PPI using SE-CAI model")
    p.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    p.add_argument("--seq_a")
    p.add_argument("--seq_b")
    p.add_argument("--test_csv")
    p.add_argument("--out_csv", default="submission.csv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--binary", action="store_true")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return p.parse_args()


@torch.no_grad()
def predict_pair(model: SECAIModel, seq_a: str, seq_b: str, device: torch.device, use_bf16: bool) -> float:
    if use_bf16:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model([seq_a], [seq_b])
    else:
        logits = model([seq_a], [seq_b])
    return torch.sigmoid(logits).item()


@torch.no_grad()
def predict_csv(model: SECAIModel, test_csv: str, out_csv: str, batch_size: int, num_workers: int, binary: bool, threshold: float, device: torch.device, use_bf16: bool):
    ds = KagglePPIDataset(test_csv, has_label=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=kaggle_collate)
    rows = []
    bar = tqdm(dl, desc="infer", leave=False)
    for seq_a, seq_b, _, ids in bar:
        if use_bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(seq_a, seq_b)
        else:
            logits = model(seq_a, seq_b)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        if binary:
            preds = [1 if p > threshold else 0 for p in probs]
        else:
            preds = probs
        for (pa, pb), pred in zip(ids, preds):
            rows.append((pa, pb, pred))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_A", "protein_B", "prediction"])
        writer.writerows(rows)


def main():
    args = parse_args()
    device = (
        torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu")
        )
    )
    model = SECAIModel(esm_model_name=args.esm_model, freeze_esm=True, device=device).to(device)
    model.eval()
    use_bf16 = False
    if args.precision == "bf16":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    elif args.precision == "auto":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    print(f"device={device}, precision={'bf16' if use_bf16 else 'fp32'}")

    if args.seq_a and args.seq_b:
        prob = predict_pair(model, args.seq_a, args.seq_b, device, use_bf16)
        pred = int(prob > args.threshold) if args.binary else prob
        print({"prob": prob, "pred": pred})
    elif args.test_csv:
        out_csv = args.out_csv
        predict_csv(model, args.test_csv, out_csv, args.batch_size, args.num_workers, args.binary, args.threshold, device, use_bf16)
        print(f"Wrote predictions to {out_csv}")
    else:
        print("Please specify either --seq_a/--seq_b or --test_csv")


if __name__ == "__main__":
    main()
