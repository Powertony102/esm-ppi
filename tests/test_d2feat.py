import torch
from src.ppi.d2feat_student import DualBranchStudentPPI
from src.ppi.d2feat_loss import D2FeatDistillationLoss


def test_d2feat_forward():
    m = DualBranchStudentPPI(vocab_size=26, embed_dim=32, hidden_dim=64, esm_dim=128)
    a = torch.randint(0, 26, (3, 50))
    b = torch.randint(0, 26, (3, 60))
    logits, (sem_a, sem_b) = m(a, b)
    assert logits.shape == (3, 2)
    assert sem_a.shape == (3, 128)
    assert sem_b.shape == (3, 128)


def test_d2feat_loss():
    crit = D2FeatDistillationLoss()
    student_logits = torch.randn(4, 2)
    student_feats = (torch.randn(4, 128), torch.randn(4, 128))
    teacher_logits = torch.randn(4, 2)
    teacher_feats = (torch.randn(4, 128), torch.randn(4, 128))
    labels = torch.tensor([0, 1, 1, 0])
    loss, parts = crit(student_logits, student_feats, teacher_logits, teacher_feats, labels)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"ce", "mse", "kl"}

