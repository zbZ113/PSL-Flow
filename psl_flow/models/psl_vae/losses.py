from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from psl_flow.models.lpips import LPIPS


def _repeat_to_three(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


@dataclass
class PSLVAELossConfig:
    lambda_f: float = 1.0
    lambda_phys: float = 1.0
    lambda_perc: float = 0.25
    lambda_kl: float = 1e-6

    @classmethod
    def from_dict(cls, payload: Dict | None) -> "PSLVAELossConfig":
        data = asdict(cls())
        if payload:
            for key, value in dict(payload).items():
                if key in data:
                    data[key] = value
        return cls(**data)


class PSLVAELoss(nn.Module):
    def __init__(self, config: PSLVAELossConfig | Dict | None = None):
        super().__init__()
        if config is None:
            config = PSLVAELossConfig()
        elif isinstance(config, dict):
            config = PSLVAELossConfig.from_dict(config)
        self.config = config
        self.perceptual_loss = None
        if float(config.lambda_perc) > 0.0:
            self.perceptual_loss = LPIPS().eval()
            self.perceptual_loss.requires_grad_(False)

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def forward(self, outputs: Dict[str, torch.Tensor], x_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        targets = outputs.get("targets")
        if targets is None:
            raise RuntimeError("PSLVAELoss expects outputs['targets'] from PSLVAE.forward_from_ir().")
        loss_t = F.l1_loss(outputs["T_rad"], targets["T_rad"])
        loss_e = F.l1_loss(outputs["e"], targets["e"])
        loss_r = F.l1_loss(outputs["R_env"], targets["R_env"])
        loss_b = F.l1_loss(outputs["B_edge"], targets["B_edge"])
        loss_delta = F.l1_loss(outputs["delta_res"], targets["delta_res"])
        loss_f = loss_t + loss_e + loss_r + loss_b + loss_delta
        loss_phys = F.l1_loss(outputs["S_phys"], targets["S_phys"])

        if self.perceptual_loss is not None:
            pred_lpips = _repeat_to_three(outputs["y_hat"] * 2.0 - 1.0)
            target_lpips = _repeat_to_three(x_01 * 2.0 - 1.0)
            loss_perc = self.perceptual_loss(pred_lpips, target_lpips).mean()
        else:
            loss_perc = self._zero(outputs["y_hat"])
        posterior = outputs.get("posterior")
        loss_kl = posterior.kl().mean() if posterior is not None else self._zero(outputs["y_hat"])
        loss_total = (
            float(self.config.lambda_f) * loss_f
            + float(self.config.lambda_phys) * loss_phys
            + float(self.config.lambda_perc) * loss_perc
            + float(self.config.lambda_kl) * loss_kl
        )
        return {
            "loss_total": loss_total,
            "loss_f": loss_f,
            "loss_f_t": loss_t,
            "loss_f_e": loss_e,
            "loss_f_r": loss_r,
            "loss_f_b": loss_b,
            "loss_f_delta": loss_delta,
            "loss_phys": loss_phys,
            "loss_perc": loss_perc,
            "loss_kl": loss_kl,
        }
