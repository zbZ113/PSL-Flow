from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from psl_flow.evaluation.metrics import normalize_01, sobel_mag


@dataclass
class TeRBLossConfig:
    beta: float = 0.1


class TeRBLoss(nn.Module):
    """TeR-B loss: L1(S_phys, S) + beta * BCE(B, Sobel(S))."""

    def __init__(self, config: TeRBLossConfig | Dict | None = None):
        super().__init__()
        if isinstance(config, dict):
            config = TeRBLossConfig(beta=float(config.get("beta", 0.1)))
        self.config = config or TeRBLossConfig()

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target = outputs["S_01"]
        s_phys = outputs["S_phys"]
        b_logits = outputs["B_edge_logits"]
        b_edge = outputs["B_edge"]
        b_gt = normalize_01(sobel_mag(target))
        loss_ter = F.l1_loss(s_phys, target)
        loss_b = F.binary_cross_entropy_with_logits(b_logits.float(), b_gt.float())
        loss_total = loss_ter + float(self.config.beta) * loss_b
        return {
            "loss_total": loss_total,
            "loss_ter": loss_ter,
            "loss_b": loss_b,
            "b_gt": b_gt,
            "s_phys_mean": s_phys.mean(),
            "s_phys_std": s_phys.std(unbiased=False),
            "b_mean": b_edge.mean(),
            "b_std": b_edge.std(unbiased=False),
        }
