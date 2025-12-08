import torch
from src.ppi.student_cnn import StudentPPICNN
from src.ppi.distill import DistillationLoss


def test_student_forward_shape():
    m = StudentPPICNN()
    a = torch.randint(0, 26, (2, 50))
    b = torch.randint(0, 26, (2, 60))
    logits = m(a, b)
    assert logits.shape == (2, 2)


def test_distill_loss():
    crit = DistillationLoss(alpha=0.5, temperature=4.0)
    s = torch.randn(4, 2)
    t = torch.randn(4, 2)
    y = torch.tensor([0, 1, 1, 0])
    loss = crit(s, t, y)
    assert torch.isfinite(loss)

