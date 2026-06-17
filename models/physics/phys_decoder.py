from __future__ import annotations

from typing import Tuple

from torch import nn
import torch.nn.functional as F

from .phys_utils import ResBlock, split_proxy_tensor


class PhysDecoder(nn.Module):
    def __init__(
        self,
        map_ch: int = 64,
        p_token_num: int = 4,
        p_token_dim: int = 64,
        a_low_range: Tuple[float, float] = (0.8, 1.2),
    ):
        super().__init__()
        self.map_ch = int(map_ch)
        self.p_token_num = int(p_token_num)
        self.p_token_dim = int(p_token_dim)
        self.a_low_range = (float(a_low_range[0]), float(a_low_range[1]))

        self.token_proj = nn.Sequential(
            nn.Linear(self.p_token_num * self.p_token_dim, self.map_ch),
            nn.SiLU(),
            nn.Linear(self.map_ch, self.map_ch),
        )
        self.block1 = ResBlock(self.map_ch, self.map_ch)
        self.conv = nn.Conv2d(self.map_ch, self.map_ch, 3, padding=1)
        self.block2 = ResBlock(self.map_ch, self.map_ch)
        self.head = nn.Conv2d(self.map_ch, 5, 1)

    def forward(self, p_map, p_token):
        token_feat = self.token_proj(p_token.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        h = p_map + token_feat
        h = self.block1(h)
        h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False)
        h = F.silu(self.conv(h))
        h = self.block2(h)
        raw = self.head(h)
        out = split_proxy_tensor(raw, a_low_range=self.a_low_range)
        out["raw"] = raw
        return out
