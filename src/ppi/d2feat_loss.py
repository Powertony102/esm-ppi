import torch
import torch.nn as nn
import torch.nn.functional as F


class D2FeatDistillationLoss(nn.Module):
    def __init__(self, alpha_mse=1.0, alpha_kl=0.5, temperature=4.0):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_kl = alpha_kl
        self.T = temperature
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, student_logits, student_feats, teacher_logits, teacher_feats, labels):
        loss_ce = self.ce(student_logits, labels)
        s0 = F.normalize(student_feats[0], dim=1)
        t0 = F.normalize(teacher_feats[0], dim=1)
        s1 = F.normalize(student_feats[1], dim=1)
        t1 = F.normalize(teacher_feats[1], dim=1)
        loss_mse = self.mse(s0, t0) + self.mse(s1, t1)
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        loss_kl = self.kl(s, t) * (self.T ** 2)
        total = loss_ce + (self.alpha_mse * loss_mse) + (self.alpha_kl * loss_kl)
        return total, {"ce": loss_ce.item(), "mse": loss_mse.item(), "kl": loss_kl.item()}
