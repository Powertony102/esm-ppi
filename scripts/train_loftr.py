import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ppi.data import KagglePPIDataset, kaggle_collate
from src.ppi.student_tokenizer import pad_batch
from src.ppi.esm_encoder import ESMEncoder
from src.ppi.loftr_student import StudentPPIWithLoFTR
from src.ppi.loftr_loss import LoFTRDistillationLoss


def parse_args():
    p = argparse.ArgumentParser("Train Student-LoFTR with interaction map distillation")
    p.add_argument("--data_dir", default="kaggle_dataset")
    p.add_argument("--teacher_esm_model", default="esm2_t33_650M_UR50D")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--alpha_map", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--loss_type", choices=["mse", "kl"], default="mse")
    return p.parse_args()


def select_device(dev):
    if dev != "auto":
        d = torch.device(dev)
    else:
        d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if d.type == "cuda" and not torch.cuda.is_available():
        d = torch.device("cpu")
    return d


def pad_teacher_map(maps, masks_a, masks_b, la, lb):
    bsz = maps.shape[0]
    out = torch.zeros((bsz, la, lb), device=maps.device, dtype=maps.dtype)
    mask_out = torch.zeros((bsz, la, lb), device=maps.device, dtype=maps.dtype)
    for i in range(bsz):
        la_i = int(masks_a[i].sum().item())
        lb_i = int(masks_b[i].sum().item())
        out[i, :la_i, :lb_i] = maps[i, :la_i, :lb_i]
        mask_out[i, :la_i, :lb_i] = 1.0
    return out, mask_out


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

    student = StudentPPIWithLoFTR().to(device)
    teacher = ESMEncoder(model_name=args.teacher_esm_model, device=device)
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)
    criterion = LoFTRDistillationLoss(alpha_map=args.alpha_map, temperature=args.temperature, loss_type=args.loss_type)

    use_bf16 = False
    if args.precision == "bf16":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    elif args.precision == "auto":
        use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    print(f"device={device}, precision={'bf16' if use_bf16 else 'fp32'}")

    for epoch in range(args.epochs):
        student.train()
        bar = tqdm(train_dl, desc=f"loftr train {epoch+1}", leave=False)
        last_loss = 0.0
        for seq_a, seq_b, labels, _ in bar:
            labels_ce = labels.squeeze(1).long().to(device)
            a_idx, b_idx = pad_batch(seq_a, seq_b, args.max_len)
            a_idx = a_idx.to(device)
            b_idx = b_idx.to(device)
            with torch.no_grad():
                fa, ma, fb, mb = teacher.encode_pair_batch(seq_a, seq_b)
                fa = F.normalize(fa, dim=-1)
                fb = F.normalize(fb, dim=-1)
                t_map = torch.matmul(fa, fb.transpose(1, 2))
                la = a_idx.size(1)
                lb = b_idx.size(1)
                t_map_pad, mask_pad = pad_teacher_map(t_map, ma, mb, la, lb)
            optim.zero_grad()
            if use_bf16:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    s_logits, s_map = student(a_idx, b_idx)
                    mask = (a_idx != 0).float().unsqueeze(2) * (b_idx != 0).float().unsqueeze(1)
                    loss = criterion(s_logits, s_map, t_map_pad, labels_ce, mask)
            else:
                s_logits, s_map = student(a_idx, b_idx)
                mask = (a_idx != 0).float().unsqueeze(2) * (b_idx != 0).float().unsqueeze(1)
                loss = criterion(s_logits, s_map, t_map_pad, labels_ce, mask)
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
            valbar = tqdm(valid_dl, desc=f"loftr valid {epoch+1}", leave=False)
            for seq_a, seq_b, labels, _ in valbar:
                labels_ce = labels.squeeze(1).long().to(device)
                a_idx, b_idx = pad_batch(seq_a, seq_b, args.max_len)
                a_idx = a_idx.to(device)
                b_idx = b_idx.to(device)
                fa, ma, fb, mb = teacher.encode_pair_batch(seq_a, seq_b)
                fa = F.normalize(fa, dim=-1)
                fb = F.normalize(fb, dim=-1)
                t_map = torch.matmul(fa, fb.transpose(1, 2))
                la = a_idx.size(1)
                lb = b_idx.size(1)
                t_map_pad, mask_pad = pad_teacher_map(t_map, ma, mb, la, lb)
                s_logits, s_map = student(a_idx, b_idx)
                mask = (a_idx != 0).float().unsqueeze(2) * (b_idx != 0).float().unsqueeze(1)
                l = criterion(s_logits, s_map, t_map_pad, labels_ce, mask)
                total += l.item() * labels_ce.size(0)
                n += labels_ce.size(0)
                valbar.set_postfix({"loss": f"{l.item():.4f}"})
            if n:
                vloss = total / n
                print(f"val_loss={vloss:.4f}")

    ckpt = os.path.join(args.output_dir, "loftr_student_last.pt")
    torch.save({"state_dict": student.state_dict()}, ckpt)
    print(f"saved={ckpt}")


if __name__ == "__main__":
    main()
