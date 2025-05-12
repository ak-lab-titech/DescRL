import sys

import numpy as np
import torch
import torch.nn.functional as F


class Qwen25VLKDLoss(torch.nn.Module):
    """
    loss for knowledge distillation.
    """
    def forward(
        self,
        teacher_logits: torch.Tensor, # (instr_len-1, batch, vocab_size)
        student_logits: torch.Tensor, # (instr_len-1, batch, vocab_size)
        teacher_logits_mask: torch.Tensor, # (batch, instr_len-1)
    ):
        if teacher_logits_mask is not None:
            teacher_logits_mask = teacher_logits_mask.permute(1, 0)
            teacher_logits = teacher_logits[teacher_logits_mask == 0]
            student_logits = student_logits[teacher_logits_mask == 0]

        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        loss = - (teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss
