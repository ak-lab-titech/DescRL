import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.load_lmdb import PAD_IDX


class VideoLLaMA2KDLoss(torch.nn.Module):
    """
    loss for knowledge distillation.
    """
    def __init__(self, visual_feature_coef, logits_coef):
        super().__init__()
        self.visual_feature_coef = visual_feature_coef
        self.logits_coef = logits_coef
        self.visual_feature_loss_fn = torch.nn.MSELoss()

        # おそらくteacherの方は活性化関数が何も用いられていない。一方、studentはreluを用いている
        self.relu = torch.nn.ReLU(True) 

    def forward(
        self,
        teacher_logits: torch.Tensor, # (instr_len-1, batch, vocab_size)
        student_logits: torch.Tensor, # (instr_len-1, batch, vocab_size)
        teacher_logits_mask: torch.Tensor, # (batch, instr_len-1)
        teacher_visual_features: torch.Tensor, # (seq_len, batch, dim)
        student_visual_features: torch.Tensor, # (seq_len, batch, dim)
        path_mask: torch.Tensor, # (batch, seq_len)
    ):
        if teacher_logits_mask is not None:
            teacher_logits_mask = teacher_logits_mask.permute(1, 0)
            teacher_logits = teacher_logits[teacher_logits_mask == 0]
            student_logits = student_logits[teacher_logits_mask == 0]

        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        logit_loss = - (teacher_probs * student_log_probs).sum(dim=-1).mean()

        if teacher_visual_features is not None:
            if path_mask is not None:
                path_mask = path_mask.permute(1, 0)
                teacher_visual_features = self.relu(teacher_visual_features[path_mask == 0])
                student_visual_features = student_visual_features[path_mask == 0]
            visual_feature_loss = self.visual_feature_loss_fn(student_visual_features, teacher_visual_features)
        else:
            visual_feature_loss = 0.0

        loss = self.visual_feature_coef * visual_feature_loss + self.logits_coef * logit_loss

        info = {
            "visual_feature_loss": visual_feature_loss,
            "logit_loss": logit_loss,
        }
        return loss, info


class R2RTokenizerVideoLLaMA2KDLoss(VideoLLaMA2KDLoss):
    """
    VideoLLaMA2をteacherとして扱いつつ、tokenizerはR2Rを用いた場合のloss
    この時、必ずハードラベルで扱うことになる。
    """
    def __init__(self, visual_feature_coef, logits_coef):
        super().__init__(visual_feature_coef, logits_coef)
        self.logit_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def forward(
        self,
        teacher_label: torch.Tensor, # (instr_len-1, batch)
        student_logits: torch.Tensor, # (instr_len-1, batch, vocab_size)
        teacher_visual_features: torch.Tensor, # (seq_len, batch, dim)
        student_visual_features: torch.Tensor, # (seq_len, batch, dim)
        path_mask: torch.Tensor, # (batch, seq_len)
    ):
        """
        teacher_label: <bos>は含まず、次の形をとる。sentence <eos> <pad> ... <pad>
        """
        logit_loss = self.logit_loss_fn(
            student_logits.reshape(-1, student_logits.shape[-1]),
            teacher_label.reshape(-1),
        )

        if teacher_visual_features is not None:
            if path_mask is not None:
                path_mask = path_mask.permute(1, 0)
                teacher_visual_features = self.relu(teacher_visual_features[path_mask == 0])
                student_visual_features = student_visual_features[path_mask == 0]
            visual_feature_loss = self.visual_feature_loss_fn(student_visual_features, teacher_visual_features)
        else:
            visual_feature_loss = 0.0

        loss = self.visual_feature_coef * visual_feature_loss + self.logits_coef * logit_loss

        info = {
            "visual_feature_loss": visual_feature_loss,
            "logit_loss": logit_loss,
        }
        return loss, info