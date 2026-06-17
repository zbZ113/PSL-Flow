from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _norm_groups(channels: int, max_groups: int = 32) -> int:
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass
class LatentPosterior:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        return self.mean + std * torch.randn_like(std)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        kl = torch.exp(self.logvar) + self.mean.pow(2) - 1.0 - self.logvar
        return 0.5 * kl.flatten(1).sum(dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop: float = 0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout(float(drop)) if float(drop) > 0 else nn.Identity()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.drop(h)
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class Downsample2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        latent_ch: int,
        *,
        base_ch: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        blocks_per_level: int = 2,
        drop: float = 0.0,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ):
        super().__init__()
        mults = tuple(int(mult) for mult in channel_mult)
        if not mults:
            raise ValueError("channel_mult must not be empty.")

        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.downsample_factor = 2 ** max(0, len(mults) - 1)

        current = int(base_ch) * mults[0]
        self.stem = nn.Conv2d(int(in_ch), current, 3, padding=1)
        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for index, mult in enumerate(mults):
            out_ch = int(base_ch) * int(mult)
            blocks = []
            for block_index in range(int(blocks_per_level)):
                block_in = current if block_index == 0 else out_ch
                blocks.append(ResBlock(block_in, out_ch, drop=drop))
            self.levels.append(nn.Sequential(*blocks))
            current = out_ch
            if index != len(mults) - 1:
                next_ch = int(base_ch) * int(mults[index + 1])
                self.downsamples.append(Downsample2d(current, next_ch))
                current = next_ch

        self.mid = nn.Sequential(
            ResBlock(current, current, drop=drop),
            ResBlock(current, current, drop=drop),
        )
        self.mean_head = nn.Conv2d(current, int(latent_ch), 3, padding=1)
        self.logvar_head = nn.Conv2d(current, int(latent_ch), 3, padding=1)

    def forward(self, x: torch.Tensor) -> LatentPosterior:
        h = self.stem(x)
        for index, level in enumerate(self.levels):
            h = level(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)
        h = self.mid(h)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp_(self.logvar_min, self.logvar_max)
        return LatentPosterior(mean=mean, logvar=logvar)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        latent_ch: int,
        out_ch: int,
        *,
        base_ch: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        blocks_per_level: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        mults = tuple(int(mult) for mult in channel_mult)
        if not mults:
            raise ValueError("channel_mult must not be empty.")

        dec_mults = tuple(reversed(mults))
        current = int(base_ch) * dec_mults[0]
        self.in_proj = nn.Conv2d(int(latent_ch), current, 3, padding=1)
        self.levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for index, mult in enumerate(dec_mults):
            level_ch = int(base_ch) * int(mult)
            blocks = []
            for block_index in range(int(blocks_per_level)):
                block_in = current if block_index == 0 else level_ch
                blocks.append(ResBlock(block_in, level_ch, drop=drop))
            self.levels.append(nn.Sequential(*blocks))
            current = level_ch
            if index != len(dec_mults) - 1:
                next_ch = int(base_ch) * int(dec_mults[index + 1])
                self.upsamples.append(Upsample2d(current, next_ch))
                current = next_ch

        self.out_norm = nn.GroupNorm(_norm_groups(current), current)
        self.out_conv = nn.Conv2d(current, int(out_ch), 3, padding=1)

    def forward(self, z: torch.Tensor, output_size: Tuple[int, int] | None = None) -> torch.Tensor:
        h = self.in_proj(z)
        for index, level in enumerate(self.levels):
            h = level(h)
            if index < len(self.upsamples):
                h = self.upsamples[index](h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        if output_size is not None and h.shape[-2:] != tuple(output_size):
            h = F.interpolate(h, size=tuple(output_size), mode="bilinear", align_corners=False)
        return h


def split_physics_tensor(
    raw: torch.Tensor,
    *,
    a_low_range: Tuple[float, float] = (0.8, 1.2),
    emissivity_range: Tuple[float, float] = (0.02, 0.98),
) -> Dict[str, torch.Tensor]:
    if raw.shape[1] != 5:
        raise ValueError(f"Expected 5 output channels, got {raw.shape}.")

    a_lo, a_hi = float(a_low_range[0]), float(a_low_range[1])
    e_lo, e_hi = float(emissivity_range[0]), float(emissivity_range[1])

    e = e_lo + (e_hi - e_lo) * torch.sigmoid(raw[:, 0:1])
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
