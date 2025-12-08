import argparse
import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ppi.data import KagglePPIDataset, kaggle_collate
from src.ppi.esm_encoder import ESMEncoder
from src.ppi.model import SECAIModel
from src.ppi.d2feat_student import DualBranchStudentPPI
from src.ppi.student_tokenizer import pad_batch
from src.ppi.d2feat_loss import D2FeatDistillationLoss


def parse_args():
    p = argparse.ArgumentParser("Train Dual-Branch Student with D2Feat distillation")
    p.add_argument("--data_dir", default="kaggle_dataset")
    p.add_argument("--teacher_esm_model", default="esm2_t33_650M_UR50D")
    p.add_argument("--teacher_checkpoint", default="outputs/se_cai_last.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--alpha_mse", type=float, default=1.0)
    p.add_argument("--alpha_kl", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=4.0)
    return p.parse_args()


def select_device(dev):
    if dev != "auto":
        d = torch.device(dev)
    else:
        d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if d.type == "cuda" and not torch.cuda.is_available():
        d = torch.device("cpu")
    return d


def masked_mean(x: torch.Tensor, mask: torch.Tensor):
    m = mask.unsqueeze(-1).type_as(x)
    s = (x * m).sum(dim=1)
    d = torch.clamp(m.sum(dim=1), min=1e-9)
    return s / d


def main():
    args = parse_args()
    device = select_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_csv = os.path.join(args.data_dir, "train.csv")
    valid_csv = os.path.join(args.data_dir, "valid.csv")
    train_ds = KagglePPIDataset(train_csv, has_label=True)
    valid_ds = KagglePPIDataset(valid_csv, has_label=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=kaggle_collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=kaggle_collate)

    teacher_logits_model = SECAIModel(esm_model_name=args.teacher_esm_model, freeze_esm=True, device=device).to(device)
    if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
        sd = torch.load(args.teacher_checkpoint, map_location=device)
        sd = sd.get("state_dict", sd)
        teacher_logits_model.load_state_dict(sd, strict=False)
    for p in teacher_logits_model.parameters():
        p.requires_grad = False
    teacher_logits_model.eval()

    teacher_feat_encoder = ESMEncoder(model_name=args.teacher_esm_model, device=device)
    esm_dim = teacher_feat_encoder.hidden_dim or 1280

    student = DualBranchStudentPPI(vocab_size=26, embed_dim=64, hidden_dim=128, esm_dim=esm_dim).to(device)
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)
    criterion = D2FeatDistillationLoss(alpha_mse=args.alpha_mse, alpha_kl=args.alpha_kl, temperature=args.temperature)

    use_bf16 = False
    if args.precision == "bf16":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    elif args.precision == "auto":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    print(f"device={device}, precision={'bf16' if use_bf16 else 'fp32'}")

    for epoch in range(args.epochs):
        student.train()
        bar = tqdm(train_dl, desc=f"d2feat train {epoch+1}", leave=False)
        last_loss = 0.0
        for seq_a, seq_b, labels, _ in bar:
            labels_ce = labels.squeeze(1).long().to(device)
            a_idx, b_idx = pad_batch(seq_a, seq_b, args.max_len)
            a_idx = a_idx.to(device)
            b_idx = b_idx.to(device)
            with torch.no_grad():
                t_logits_bin = teacher_logits_model(seq_a, seq_b)
                t_logits = torch.cat([torch.zeros_like(t_logits_bin), t_logits_bin], dim=1)
                fa, ma, fb, mb = teacher_feat_encoder.encode_pair_batch(seq_a, seq_b)
                t_feat_a = masked_mean(fa, ma)
                t_feat_b = masked_mean(fb, mb)
            optim.zero_grad()
            if use_bf16:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    s_logits, s_feats = student(a_idx, b_idx)
                    loss, _ = criterion(s_logits, s_feats, t_logits, (t_feat_a, t_feat_b), labels_ce)
            else:
                s_logits, s_feats = student(a_idx, b_idx)
                loss, _ = criterion(s_logits, s_feats, t_logits, (t_feat_a, t_feat_b), labels_ce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
            last_loss = loss.item()
            bar.set_postfix({"loss": f"{last_loss:.4f}"})
        print(f"epoch={epoch+1} loss={last_loss:.4f}")

        student.eval()
        with torch.no_grad():
            total = 0.0
            n = 0
            valbar = tqdm(valid_dl, desc=f"d2feat valid {epoch+1}", leave=False)
            for seq_a, seq_b, labels, _ in valbar:
                labels_ce = labels.squeeze(1).long().to(device)
                a_idx, b_idx = pad_batch(seq_a, seq_b, args.max_len)
                a_idx = a_idx.to(device)
                b_idx = b_idx.to(device)
                t_logits_bin = teacher_logits_model(seq_a, seq_b)
                t_logits = torch.cat([torch.zeros_like(t_logits_bin), t_logits_bin], dim=1)
                fa, ma, fb, mb = teacher_feat_encoder.encode_pair_batch(seq_a, seq_b)
                t_feat_a = masked_mean(fa, ma)
                t_feat_b = masked_mean(fb, mb)
                s_logits, s_feats = student(a_idx, b_idx)
                l, _ = criterion(s_logits, s_feats, t_logits, (t_feat_a, t_feat_b), labels_ce)
                total += l.item() * labels_ce.size(0)
                n += labels_ce.size(0)
                valbar.set_postfix({"loss": f"{l.item():.4f}"})
            if n:
                vloss = total / n
                print(f"val_loss={vloss:.4f}")
                metrics_path = os.path.join(args.output_dir, "d2feat_val_metrics.csv")
                write_header = not os.path.exists(metrics_path)
                with open(metrics_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["epoch", "val_loss", "device", "precision"]) 
                    w.writerow([epoch + 1, f"{vloss:.6f}", str(device), "bf16" if use_bf16 else "fp32"]) 

    ckpt = os.path.join(args.output_dir, "d2feat_student_last.pt")
    torch.save({"state_dict": student.state_dict(), "esm_dim": esm_dim}, ckpt)
    print(f"saved={ckpt}")


if __name__ == "__main__":
    main()
