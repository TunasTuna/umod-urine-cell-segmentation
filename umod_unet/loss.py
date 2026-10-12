from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    ce_weight: torch.Tensor | None,
    n_classes: int = 8,
    smooth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted CE + (1 - mean Dice over classes 1–7). Background is in CE only."""
    ce = F.cross_entropy(logits.float(), y, weight=ce_weight, label_smoothing=0.04)
    p = logits.float().softmax(1)
    oh = F.one_hot(y, n_classes).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = (2 * inter + smooth) / (den + smooth)
    alpha, beta = 0.3, 0.7
    fn = (oh * (1 - p)).sum((0, 2, 3))
    fp = ((1 - oh) * p).sum((0, 2, 3))
    tversky = (inter + smooth) / (inter + alpha * fp + beta * fn + smooth)
    rare = tversky[[1, 3, 4]].mean()
    loss = ce + (1 - dice[1:].mean()) + 0.5 * (1 - rare)
    return loss, ce.detach()
