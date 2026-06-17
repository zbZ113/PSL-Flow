from __future__ import annotations

from torch import nn

from .phys_utils import CompactEncoder


class PhysCPEN(nn.Module):
    def __init__(self, in_ch: int = 5, base_ch: int = 64, p_token_num: int = 4, p_token_dim: int = 64):
        super().__init__()
        self.base_ch = int(base_ch)
        self.p_token_num = int(p_token_num)
        self.p_token_dim = int(p_token_dim)
        self.encoder = CompactEncoder(int(in_ch), base_ch=self.base_ch)
        self.map_head = nn.Conv2d(self.base_ch, self.base_ch, 1)
        self.token_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.base_ch, self.p_token_num * self.p_token_dim),
        )

    def forward(self, y_phys32):
        h = self.encoder(y_phys32)
        p_map = self.map_head(h)
        p_token = self.token_head(h).view(y_phys32.shape[0], self.p_token_num, self.p_token_dim)
        return p_map, p_token
