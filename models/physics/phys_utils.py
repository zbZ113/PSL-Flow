from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _norm_groups(channels: int, max_groups: int = 8) -> int:
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.n1 = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class CompactEncoder(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.block1 = ResBlock(base_ch, base_ch)
        self.down = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.block2 = ResBlock(base_ch, base_ch)
        self.block3 = ResBlock(base_ch, base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.down(h)
        h = self.block2(h)
        h = self.block3(h)
        return h


class LocalEnvEstimator(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = int(kernel_size)
        self.kernel_logits = nn.Parameter(torch.zeros(1, 1, self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        with torch.no_grad():
            self.kernel_logits[:, :, center, center] = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        kernel = torch.softmax(self.kernel_logits.view(1, -1), dim=1).view_as(self.kernel_logits)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel)


def clamp_sigmoid(x: torch.Tensor, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(x)


def build_env_basis(v4: torch.Tensor, s_01: torch.Tensor) -> torch.Tensor:
    if v4.shape[1] != 4:
        raise ValueError(f"Expected v4 with 4 channels, got {v4.shape}.")
    h, w = s_01.shape[-2:]
    coarse = F.adaptive_avg_pool2d(s_01, (2, 2)).reshape(s_01.shape[0], 1, 4)
    v_flat = v4.reshape(v4.shape[0], 4, h * w)
    env = torch.matmul(coarse, v_flat).view(v4.shape[0], 1, h, w)
    return env.clamp(0.0, 1.0)


def split_proxy_tensor(raw: torch.Tensor, a_low_range: Tuple[float, float] = (0.8, 1.2)) -> Dict[str, torch.Tensor]:
    if raw.shape[1] != 5:
        raise ValueError(f"Expected 5 proxy channels, got {raw.shape}.")
    a_lo, a_hi = float(a_low_range[0]), float(a_low_range[1])
    e = clamp_sigmoid(raw[:, 0:1], 0.02, 0.98)
    t_rad = F.softplus(raw[:, 1:2])
    r_env = F.softplus(raw[:, 2:3])
    a = a_lo + (a_hi - a_lo) * torch.sigmoid(raw[:, 3:4])
    b_edge = torch.sigmoid(raw[:, 4:5])
    s_scene = e * t_rad + (1.0 - e) * r_env
    s_phys = torch.clamp(a * s_scene, 0.0, 1.0)
    return {
        "e": e,
        "T_rad": t_rad,
        "R_env": r_env,
        "A": a,
        "B_edge": b_edge,
        "S_scene": s_scene,
        "S_phys": s_phys,
    }


def build_lowres_targets(teacher_out: Dict[str, torch.Tensor], size: int = 32) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    out["e"] = F.adaptive_avg_pool2d(teacher_out["e"], (size, size))
    out["T_rad"] = F.adaptive_avg_pool2d(teacher_out["T_rad"], (size, size))
    out["R_env"] = F.adaptive_avg_pool2d(teacher_out["R_env"], (size, size))
    out["A"] = F.adaptive_avg_pool2d(teacher_out["A"], (size, size))
    out["B_edge"] = F.adaptive_max_pool2d(teacher_out["B_edge"], (size, size))
    out["S_01"] = F.adaptive_avg_pool2d(teacher_out["S_01"], (size, size))
    out["S_phys"] = F.adaptive_avg_pool2d(teacher_out["S_phys"], (size, size))
    out["Y_phys32"] = torch.cat(
        [out["e"], out["T_rad"], out["R_env"], out["A"], out["B_edge"]],
        dim=1,
    )
    return out


def ramp_weight(base: float, current_epoch: int, warmup_epochs: int) -> float:
    base = float(base)
    warmup_epochs = int(warmup_epochs)
    if base <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return base
    progress = min(1.0, float(current_epoch + 1) / float(warmup_epochs))
    return base * progress


def set_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    module.requires_grad_(requires_grad)
    if not requires_grad:
        module.eval()
