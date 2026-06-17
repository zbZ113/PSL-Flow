from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().flatten(1).mean(dim=1)


def cosine_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = pred.flatten(1)
    t = target.flatten(1)
    p = p / (p.norm(dim=1, keepdim=True) + eps)
    t = t / (t.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (p * t).sum(dim=1)


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    dtype = x.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), ky)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def normalize_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    x_min = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    x_max = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def tv_weighted(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    wx = weight[..., :, 1:]
    wy = weight[..., 1:, :]
    return (dx * wx).mean() + (dy * wy).mean()


def ssim_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} vs {target.shape}")
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")

    pad = window_size // 2

    def _avg(x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(x, kernel_size=window_size, stride=1)

    mu_x = _avg(pred)
    mu_y = _avg(target)
    sigma_x = _avg(pred * pred) - mu_x * mu_x
    sigma_y = _avg(target * target) - mu_y * mu_y
    sigma_xy = _avg(pred * target) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.flatten(1).mean(dim=1)
