import torch
from src.ppi.model import SECAIModel, FocalLoss


def test_forward_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SECAIModel(esm_model_name="esm2_t6_8M_UR50D", device=device)
    model.eval()
    logits = model(["ACDEFGHIKLMNPQRSTVWY"], ["WYVTSRQPONMLKIHGFEDCA"]).to("cpu")
    assert logits.shape == (1, 1)


def test_focal_loss():
    crit = FocalLoss()
    inputs = torch.tensor([[0.0], [1.0]])
    targets = torch.tensor([[0.0], [1.0]])
    loss = crit(inputs, targets)
    assert torch.isfinite(loss)
