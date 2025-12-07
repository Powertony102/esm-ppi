import argparse
import torch

from src.ppi.model import SECAIModel


def parse_args():
    p = argparse.ArgumentParser("Infer PPI probability for a pair")
    p.add_argument("--seq_a", required=True)
    p.add_argument("--seq_b", required=True)
    p.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SECAIModel(esm_model_name=args.esm_model, freeze_esm=True, device=device).to(device)
    model.eval()
    logits = model([args.seq_a], [args.seq_b])
    prob = torch.sigmoid(logits).item()
    pred = int(prob > args.threshold)
    print({"prob": prob, "pred": pred})


if __name__ == "__main__":
    main()

