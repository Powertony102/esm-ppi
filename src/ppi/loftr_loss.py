import torch
import torch.nn as nn
import torch.nn.functional as F


class LoFTRDistillationLoss(nn.Module):
    def __init__(self, alpha_map: float = 10.0, temperature: float = 4.0, loss_type: str = "mse"):
        super().__init__()
        self.alpha_map = alpha_map
        self.temperature = temperature
        self.loss_type = loss_type
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.mse = nn.MSELoss(reduction="none")
        self.kl = nn.KLDivLoss(reduction="none")

    def forward(self, student_logits: torch.Tensor, student_map: torch.Tensor, teacher_map: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        loss_ce = self.ce(student_logits, labels)
        if self.loss_type == "kl":
            s = F.log_softmax(student_map / self.temperature, dim=-1)
            t = F.softmax(teacher_map / self.temperature, dim=-1)
            map_loss = self.kl(s, t)
        else:
            map_loss = self.mse(student_map, teacher_map)
        map_loss = (map_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        return loss_ce + self.alpha_map * map_loss

