from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from psl_flow.models.terb.physics import LocalEnvEstimator, build_env_basis, clamp_sigmoid
from psl_flow.models.terb.smp_backbone import SMPWrapper


class TeRB(nn.Module):
    """TeR-B teacher."""

    def __init__(
        self,
        *,
        smp_model: str = "Unet",
        smp_encoder: str = "resnet18",
        smp_encoder_weights: str | None = None,
        vnums: int = 4,
        erme_kernel: int = 5,
        lambda_env_init: float = 0.1,
    ):
        super().__init__()
        if int(vnums) != 4:
            raise ValueError("TeRB currently expects vnums=4 to match the existing environment basis.")
        self.vnums = int(vnums)
        self.backbone = SMPWrapper(
            smp_model=smp_model,
            encoder_name=smp_encoder,
            encoder_weights=smp_encoder_weights,
            in_channels=1,
            out_channels=2 + self.vnums + 1,
            import_context="TeRB",
        )
        self.local_env = LocalEnvEstimator(kernel_size=int(erme_kernel))
        init = torch.tensor(float(lambda_env_init)).clamp(1e-4, 1.0 - 1e-4)
        self.lambda_env_logit = nn.Parameter(torch.logit(init))

    def forward(self, s_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, _, _ = self.backbone(s_01)
        if logits.shape[-2:] != s_01.shape[-2:]:
            logits = F.interpolate(logits, size=s_01.shape[-2:], mode="bilinear", align_corners=False)

        h_e = logits[:, 0:1]
        h_t = logits[:, 1:2]
        h_v4 = logits[:, 2 : 2 + self.vnums]
        h_b = logits[:, 2 + self.vnums : 3 + self.vnums]

        e = clamp_sigmoid(h_e, 0.02, 0.98)
        t_rad = F.softplus(h_t)
        v4 = torch.softmax(h_v4, dim=1)
        b_edge = torch.sigmoid(h_b)

        r_env_basis = build_env_basis(v4, s_01)
        r_env_local = self.local_env(s_01).clamp(0.0, 1.0)
        lambda_env = torch.sigmoid(self.lambda_env_logit)
        r_env = (1.0 - lambda_env) * r_env_basis + lambda_env * r_env_local

        s_phys = torch.clamp(e * t_rad + (1.0 - e) * r_env, 0.0, 1.0)
        return {
            "T_rad": t_rad,
            "e": e,
            "R_env": r_env,
            "R_env_basis": r_env_basis,
            "R_env_local": r_env_local,
            "V4": v4,
            "B_edge_logits": h_b,
            "B_edge": b_edge,
            "S_phys": s_phys,
            "S_01": s_01,
            "lambda_env": lambda_env.view(1),
        }
