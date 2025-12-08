import torch
from src.ppi.loftr_student import StudentPPIWithLoFTR
from src.ppi.loftr_loss import LoFTRDistillationLoss


def test_loftr_forward_shapes():
    m = StudentPPIWithLoFTR()
    a = torch.randint(0, 26, (2, 40))
    b = torch.randint(0, 26, (2, 50))
    logits, inter = m(a, b)
    assert logits.shape == (2, 2)
    assert inter.shape == (2, 40, 50)


def test_loftr_masked_loss():
    crit = LoFTRDistillationLoss(alpha_map=10.0)
    logits = torch.randn(2, 2)
    student_map = torch.randn(2, 40, 50)
    teacher_map = torch.randn(2, 40, 50)
    labels = torch.tensor([0, 1])
    mask = torch.ones(2, 40, 50)
    loss = crit(logits, student_map, teacher_map, labels, mask)
    assert torch.isfinite(loss)

