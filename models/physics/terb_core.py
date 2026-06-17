from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .smp_backbone import SMPWrapper

from .phys_utils import LocalEnvEstimator, build_env_basis, clamp_sigmoid


class TeR_B(nn.Module):
    def __init__(
        self,
        *,
        smp_model: str = "Unet",
        smp_encoder: str = "resnet18",
        smp_encoder_weights: str | None = None,
        vnums: int = 4,
        erme_kernel: int = 5,
        lambda_env_init: float = 0.1,
        a_low_range: Tuple[float, float] = (0.8, 1.2),
    ):
        super().__init__()
        if int(vnums) != 4:
            raise ValueError("TeR-B Net currently expects vnums=4.")
        self.vnums = int(vnums)
        self.a_low_range = (float(a_low_range[0]), float(a_low_range[1]))

        self.backbone = SMPWrapper(
            smp_model=smp_model,
            encoder_name=smp_encoder,
            encoder_weights=smp_encoder_weights,
            in_channels=1,
            out_channels=2 + self.vnums + 1,
            import_context="TeR_B",
        )

        enc_out_ch = getattr(self.backbone.net.encoder, "out_channels", None)
        deep_ch = int(enc_out_ch[-1]) if isinstance(enc_out_ch, (list, tuple)) else 512
        mid_ch = max(32, deep_ch // 4)
        self.a_low_head = nn.Sequential(
            nn.Conv2d(deep_ch, mid_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, 1, 1),
        )
        self.local_env = LocalEnvEstimator(kernel_size=int(erme_kernel))
        init = torch.tensor(float(lambda_env_init)).clamp(1e-4, 1.0 - 1e-4)
        self.lambda_env_logit = nn.Parameter(torch.logit(init))

    def forward(self, s_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, _, feats = self.backbone(s_01)
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

        feat8 = F.adaptive_avg_pool2d(feats[-1], (8, 8))
        a8_raw = self.a_low_head(feat8)
        a_lo, a_hi = self.a_low_range
        a8 = a_lo + (a_hi - a_lo) * torch.sigmoid(a8_raw)
        a = F.interpolate(a8, size=s_01.shape[-2:], mode="bilinear", align_corners=False)

        r_env_basis = build_env_basis(v4, s_01)
        r_env_local = self.local_env(s_01).clamp(0.0, 1.0)
        lambda_env = torch.sigmoid(self.lambda_env_logit)
        r_env = (1.0 - lambda_env) * r_env_basis + lambda_env * r_env_local

        s_scene = e * t_rad + (1.0 - e) * r_env
        s_phys = torch.clamp(a * s_scene, 0.0, 1.0)

        return {
            "e": e,
            "T_rad": t_rad,
            "V4": v4,
            "B_edge_logits": h_b,
            "B_edge": b_edge,
            "A8": a8,
            "A": a,
            "R_env_basis": r_env_basis,
            "R_env_local": r_env_local,
            "R_env": r_env,
            "S_scene": s_scene,
            "S_phys": s_phys,
            "S_01": s_01,
            "lambda_env": lambda_env.view(1),
        }

