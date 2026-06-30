from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LocalEnvEstimator(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = int(kernel_size)
        self.kernel_logits = nn.Parameter(torch.zeros(1, 1, self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        with torch.no_grad():
            self.kernel_logits[:, :, center, center] = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        kernel = torch.softmax(self.kernel_logits.view(1, -1), dim=1).reshape(self.kernel_logits.shape)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel)


def clamp_sigmoid(x: torch.Tensor, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    return float(low) + (float(high) - float(low)) * torch.sigmoid(x)


def build_env_basis(v4: torch.Tensor, s_01: torch.Tensor) -> torch.Tensor:
    if v4.shape[1] != 4:
        raise ValueError(f"Expected V4 with 4 channels, got {v4.shape}.")
    h, w = s_01.shape[-2:]
    coarse = F.adaptive_avg_pool2d(s_01, (2, 2)).reshape(s_01.shape[0], 1, 4)
    v_flat = v4.reshape(v4.shape[0], 4, h * w)
    env = torch.matmul(coarse, v_flat).view(v4.shape[0], 1, h, w)
    return env.clamp(0.0, 1.0)
