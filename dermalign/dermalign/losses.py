"""Classification losses for the long tail + the lesion-alignment loss."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted cross-entropy (Menon et al., ICLR 2021):
    train on logits + tau * log(prior); predict with the raw logits."""

    def __init__(self, prior, tau: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        prior = np.clip(np.asarray(prior, dtype=np.float64), 1e-8, None)
        self.register_buffer("adj", tau * torch.log(torch.as_tensor(prior, dtype=torch.float32)))
        self.ls = label_smoothing

    def forward(self, logits, y):
        return F.cross_entropy(logits + self.adj.to(logits.dtype), y, label_smoothing=self.ls)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1)
        logpt = logp.gather(1, y[:, None]).squeeze(1)
        loss = -((1 - logpt.exp()) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[y]
        return loss.mean()


def build_criterion(name: str, train_counts: np.ndarray, prior: np.ndarray, tau: float = 1.0,
                    label_smoothing: float = 0.0, focal_gamma: float = 2.0):
    if name == "la":
        return LogitAdjustedLoss(prior, tau, label_smoothing)
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    w = 1.0 / np.maximum(train_counts, 1)
    w = w / w.sum() * len(w)
    if name == "wce":
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32), label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(focal_gamma)
    raise ValueError(name)


def dilate_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius_px + 1, stride=1, padding=radius_px)


def alignment_loss(att: torch.Tensor, mask: torch.Tensor, mask_valid: torch.Tensor,
                   dilate_frac: float = 0.05) -> torch.Tensor:
    """Attention mass falling outside the (dilated) lesion mask.

    att: B x 1 x h x w, non-negative, sums to 1 per sample.
    mask: B x 1 x H x W in {0, 1}; mask_valid: B (0 for images without a usable mask).
    The mask is dilated by dilate_frac * H pixels so the lesion border and the
    immediately surrounding skin remain 'allowed'.
    """
    if mask_valid.sum() == 0:
        return att.new_zeros(())
    m = dilate_mask(mask.float(), int(round(dilate_frac * mask.shape[-2])))
    m_small = F.adaptive_avg_pool2d(m, att.shape[-2:])  # fraction of lesion per cell
    outside = (att.float() * (1.0 - m_small)).flatten(1).sum(1)
    return (outside * mask_valid).sum() / mask_valid.sum().clamp_min(1.0)


def attention_entropy(att: torch.Tensor) -> torch.Tensor:
    """Normalised entropy of the attention map in [0, 1] (mean over batch)."""
    a = att.float().flatten(1).clamp_min(1e-12)
    return (-(a * a.log()).sum(1) / np.log(a.shape[1])).mean()
