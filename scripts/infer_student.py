import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ppi.student_cnn import StudentPPICNN
from src.ppi.student_tokenizer import pad_batch
from src.ppi.data import KagglePPIDataset, kaggle_collate


def parse_args():
    p = argparse.ArgumentParser("Infer with student CNN")
    p.add_argument("--seq_a")
    p.add_argument("--seq_b")
    p.add_argument("--test_csv")
    p.add_argument("--out_csv", default="student_submission.csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    return p.parse_args()


def select_device(dev):
    if dev != "auto":
        d = torch.device(dev)
    else:
        d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if d.type == "cuda" and not torch.cuda.is_available():
        d = torch.device("cpu")
    return d


@torch.no_grad()
def main():
    args = parse_args()
    device = select_device(args.device)
    model = StudentPPICNN().to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    use_bf16 = False
    if args.precision == "bf16":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    elif args.precision == "auto":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    print(f"device={device}, precision={'bf16' if use_bf16 else 'fp32'}")

    if args.seq_a and args.seq_b:
        a_idx, b_idx = pad_batch([args.seq_a], [args.seq_b], args.max_len)
        a_idx = a_idx.to(device)
        b_idx = b_idx.to(device)
        if use_bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(a_idx, b_idx)
        else:
            logits = model(a_idx, b_idx)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        print({"prob": prob, "pred": int(prob > 0.5)})
    elif args.test_csv:
        ds = KagglePPIDataset(args.test_csv, has_label=False)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=kaggle_collate)
        rows = []
        bar = tqdm(dl, desc="student infer", leave=False)
        for seq_a, seq_b, _, ids in bar:
            a_idx, b_idx = pad_batch(seq_a, seq_b, args.max_len)
            a_idx = a_idx.to(device)
            b_idx = b_idx.to(device)
            if use_bf16:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(a_idx, b_idx)
            else:
                logits = model(a_idx, b_idx)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            for (pa, pb), p in zip(ids, probs):
                rows.append((pa, pb, p))
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["protein_A", "protein_B", "prediction"])
            w.writerows(rows)
        print(f"Wrote {args.out_csv}")
    else:
        print("Provide --seq_a/--seq_b or --test_csv")


if __name__ == "__main__":
    main()

