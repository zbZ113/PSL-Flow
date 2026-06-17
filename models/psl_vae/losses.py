from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from models.generative_models.vqgan_networks.lpips import LPIPS
from models.physics import sobel_mag


def _repeat_to_three(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


@dataclass
class PhysFactorVAELossConfig:
    w_e: float = 1.0
    w_t: float = 1.0
    w_r: float = 1.0
    w_a: float = 0.1
    w_b: float = 0.25
    lambda_factor: float = 1.0
    lambda_phys: float = 1.0
    lambda_delta: float = 1.0
    lambda_img: float = 2.0
    lambda_edge: float = 0.25
    lambda_lpips: float = 0.0
    kl_weight: float = 1e-6

    @classmethod
    def from_dict(cls, payload: Dict | None) -> "PhysFactorVAELossConfig":
        base = cls()
        data = asdict(base)
        if payload is not None:
            for key, value in dict(payload).items():
                if key in data:
                    data[key] = value
        return cls(**data)


class PhysFactorVAELoss(nn.Module):
    def __init__(self, config: PhysFactorVAELossConfig | Dict | None = None):
        super().__init__()
        if config is None:
            config = PhysFactorVAELossConfig()
        elif isinstance(config, dict):
            config = PhysFactorVAELossConfig.from_dict(config)
        self.config = config
        self.perceptual_loss = None
        if float(self.config.lambda_lpips) > 0.0:
            self.perceptual_loss = LPIPS().eval()
            self.perceptual_loss.requires_grad_(False)

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def forward(self, outputs: Dict[str, torch.Tensor], x_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        targets = outputs.get("targets", None)
        if targets is None:
            raise RuntimeError("PSL_VAELoss expects outputs['targets'] from PSL_VAE.forward_from_ir().")

        loss_e = F.l1_loss(outputs["e"], targets["e"])
        loss_t = F.l1_loss(outputs["T_rad"], targets["T_rad"])
        loss_r = F.l1_loss(outputs["R_env"], targets["R_env"])
        loss_a = F.l1_loss(outputs["A"], targets["A"])
        loss_b = F.l1_loss(outputs["B_edge"], targets["B_edge"])
        loss_factor = (
            float(self.config.w_e) * loss_e
            + float(self.config.w_t) * loss_t
            + float(self.config.w_r) * loss_r
            + float(self.config.w_a) * loss_a
            + float(self.config.w_b) * loss_b
        )
        loss_phys = F.l1_loss(outputs["S_phys"], targets["S_phys"])
        loss_delta = F.l1_loss(outputs["delta_res"], targets["delta_res"])
        loss_img = F.l1_loss(outputs["y_hat"], x_01)
        loss_edge = F.l1_loss(sobel_mag(outputs["y_hat"]), sobel_mag(x_01))

        if self.perceptual_loss is not None:
            pred_lpips = _repeat_to_three(outputs["y_hat"] * 2.0 - 1.0)
            target_lpips = _repeat_to_three(x_01 * 2.0 - 1.0)
            loss_lpips = self.perceptual_loss(pred_lpips, target_lpips).mean()
        else:
            loss_lpips = self._zero(outputs["y_hat"])

        posterior = outputs.get("posterior", None)
        loss_kl = posterior.kl().mean() if posterior is not None else self._zero(outputs["y_hat"])

        loss_total = (
            float(self.config.lambda_factor) * loss_factor
            + float(self.config.lambda_phys) * loss_phys
            + float(self.config.lambda_delta) * loss_delta
            + float(self.config.lambda_img) * loss_img
            + float(self.config.lambda_edge) * loss_edge
            + float(self.config.lambda_lpips) * loss_lpips
            + float(self.config.kl_weight) * loss_kl
        )

        return {
            "loss_total": loss_total,
            "loss_factor": loss_factor,
            "loss_factor_e": loss_e,
            "loss_factor_t": loss_t,
            "loss_factor_r": loss_r,
            "loss_factor_a": loss_a,
            "loss_factor_b": loss_b,
            "loss_phys": loss_phys,
            "loss_delta": loss_delta,
            "loss_img": loss_img,
            "loss_edge": loss_edge,
            "loss_lpips": loss_lpips,
            "loss_kl": loss_kl,
        }
