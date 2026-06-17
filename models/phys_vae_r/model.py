from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .modules import ConvDecoder, ConvEncoder, ResBlock, split_physics_tensor


def _set_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    module.requires_grad_(requires_grad)
    if not requires_grad:
        module.eval()


@dataclass
class PhysVAERConfig:
    in_channels: int = 1
    phys_latent_channels: int = 4
    res_latent_channels: int = 4
    base_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    blocks_per_level: int = 2
    dropout: float = 0.0
    a_low_range: Tuple[float, float] = (0.8, 1.2)
    emissivity_range: Tuple[float, float] = (0.02, 0.98)
    residual_scale: float = 0.25

    @classmethod
    def from_dict(cls, payload: Dict) -> "PhysVAERConfig":
        base = cls()
        data = asdict(base)
        for key, value in dict(payload).items():
            if key in data:
                data[key] = value
        return cls(**data)


def pack_joint_latent(z_phys: torch.Tensor, z_res: torch.Tensor) -> torch.Tensor:
    if z_phys.shape[0] != z_res.shape[0] or z_phys.shape[-2:] != z_res.shape[-2:]:
        raise ValueError(f"Cannot pack latent tensors with shapes {z_phys.shape} and {z_res.shape}.")
    return torch.cat([z_phys, z_res], dim=1)


def split_joint_latent(
    z_joint: torch.Tensor,
    *,
    phys_channels: int,
    res_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected = int(phys_channels) + int(res_channels)
    if z_joint.shape[1] != expected:
        raise ValueError(
            f"Joint latent channel mismatch: expected {expected}, got {z_joint.shape[1]}."
        )
    return torch.split(z_joint, [int(phys_channels), int(res_channels)], dim=1)


class PhysicsDecoder(nn.Module):
    def __init__(self, config: PhysVAERConfig):
        super().__init__()
        self.config = config
        self.backbone = ConvDecoder(
            latent_ch=int(config.phys_latent_channels),
            out_ch=int(config.base_channels),
            base_ch=int(config.base_channels),
            channel_mult=config.channel_mult,
            blocks_per_level=int(config.blocks_per_level),
            drop=float(config.dropout),
        )
        self.head = nn.Conv2d(int(config.base_channels), 5, 1)

    def forward(self, z_phys: torch.Tensor, output_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        feat = self.backbone(z_phys, output_size=output_size)
        raw = self.head(feat)
        out = split_physics_tensor(
            raw,
            a_low_range=self.config.a_low_range,
            emissivity_range=self.config.emissivity_range,
        )
        out["raw"] = raw
        out["feat"] = feat
        return out


class ResidualRenderer(nn.Module):
    def __init__(self, config: PhysVAERConfig):
        super().__init__()
        self.config = config
        hidden = int(config.base_channels)
        self.backbone = ConvDecoder(
            latent_ch=int(config.res_latent_channels),
            out_ch=hidden,
            base_ch=int(config.base_channels),
            channel_mult=config.channel_mult,
            blocks_per_level=int(config.blocks_per_level),
            drop=float(config.dropout),
        )
        self.fuse = nn.Conv2d(hidden + 2, hidden, 3, padding=1)
        self.block1 = ResBlock(hidden, hidden, drop=float(config.dropout))
        self.block2 = ResBlock(hidden, hidden, drop=float(config.dropout))
        self.out = nn.Conv2d(hidden, int(config.in_channels), 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        s_phys: torch.Tensor,
        b_edge: torch.Tensor,
        z_res: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        feat = self.backbone(z_res, output_size=output_size)
        if feat.shape[-2:] != s_phys.shape[-2:]:
            feat = F.interpolate(feat, size=s_phys.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([feat, s_phys, b_edge], dim=1)
        h = F.silu(self.fuse(h))
        h = self.block1(h)
        h = self.block2(h)
        return float(self.config.residual_scale) * torch.tanh(self.out(h))


class PhysVAER(nn.Module):
    """Physics-dominant VAE with a residual lane.

    The module expects infrared inputs in [0, 1]. The physics lane reconstructs
    a proxy physical base image first, then the residual lane compensates only
    the unexplained appearance residual.
    """

    def __init__(self, config: PhysVAERConfig | Dict | None = None):
        super().__init__()
        if config is None:
            config = PhysVAERConfig()
        elif isinstance(config, dict):
            config = PhysVAERConfig.from_dict(config)
        self.config = config

        self.phys_encoder = ConvEncoder(
            in_ch=int(config.in_channels),
            latent_ch=int(config.phys_latent_channels),
            base_ch=int(config.base_channels),
            channel_mult=config.channel_mult,
            blocks_per_level=int(config.blocks_per_level),
            drop=float(config.dropout),
        )
        self.res_encoder = ConvEncoder(
            in_ch=int(config.in_channels),
            latent_ch=int(config.res_latent_channels),
            base_ch=int(config.base_channels),
            channel_mult=config.channel_mult,
            blocks_per_level=int(config.blocks_per_level),
            drop=float(config.dropout),
        )
        self.phys_decoder = PhysicsDecoder(config)
        self.res_renderer = ResidualRenderer(config)
        self.downsample_factor = int(self.phys_encoder.downsample_factor)

    @property
    def phys_channels(self) -> int:
        return int(self.config.phys_latent_channels)

    @property
    def res_channels(self) -> int:
        return int(self.config.res_latent_channels)

    @property
    def joint_channels(self) -> int:
        return self.phys_channels + self.res_channels

    def pack_joint_latent(self, z_phys: torch.Tensor, z_res: torch.Tensor) -> torch.Tensor:
        return pack_joint_latent(z_phys, z_res)

    def split_joint_latent(self, z_joint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return split_joint_latent(
            z_joint,
            phys_channels=self.phys_channels,
            res_channels=self.res_channels,
        )

    def _default_output_size(self, z: torch.Tensor) -> Tuple[int, int]:
        return (
            int(z.shape[-2]) * self.downsample_factor,
            int(z.shape[-1]) * self.downsample_factor,
        )

    def encode_phys(self, x_01: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        posterior = self.phys_encoder(x_01)
        z_phys = posterior.sample() if sample else posterior.mode()
        return {
            "posterior_phys": posterior,
            "z_phys": z_phys,
        }

    def encode_residual(self, residual_01: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        posterior = self.res_encoder(residual_01)
        z_res = posterior.sample() if sample else posterior.mode()
        return {
            "posterior_res": posterior,
            "z_res": z_res,
        }

    def encode_joint_latent(
        self,
        x_01: torch.Tensor,
        *,
        sample: bool = False,
        sample_phys: bool | None = None,
        sample_res: bool | None = None,
        detach_base_for_residual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.forward(
            x_01,
            sample=sample,
            sample_phys=sample_phys,
            sample_res=sample_res,
            detach_base_for_residual=detach_base_for_residual,
        )
        return {
            "z_phys": encoded["z_phys"],
            "z_res": encoded["z_res"],
            "z_joint": encoded["z_joint"],
            "posterior_phys": encoded["posterior_phys"],
            "posterior_res": encoded["posterior_res"],
        }

    def decode_latents(
        self,
        *,
        z_joint: torch.Tensor | None = None,
        z_phys: torch.Tensor | None = None,
        z_res: torch.Tensor | None = None,
        output_size: Tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if z_joint is not None:
            if z_phys is not None or z_res is not None:
                raise ValueError("Pass either z_joint or (z_phys, z_res), not both.")
            z_phys, z_res = self.split_joint_latent(z_joint)
        if z_phys is None or z_res is None:
            raise ValueError("Both z_phys and z_res are required for decoding.")
        if output_size is None:
            output_size = self._default_output_size(z_phys)

        phys = self.phys_decoder(z_phys, output_size=output_size)
        delta = self.res_renderer(phys["S_phys"], phys["B_edge"], z_res, output_size=output_size)
        x_hat = torch.clamp(phys["S_phys"] + delta, 0.0, 1.0)
        outputs = {
            "z_phys": z_phys,
            "z_res": z_res,
            "z_joint": self.pack_joint_latent(z_phys, z_res),
            "delta": delta,
            "x_hat": x_hat,
            "x_base": phys["S_phys"],
        }
        outputs.update(phys)
        return outputs

    def set_stage(self, stage: str) -> None:
        stage = str(stage).lower()
        if stage in {"phys", "physics"}:
            _set_requires_grad(self.phys_encoder, True)
            _set_requires_grad(self.phys_decoder, True)
            _set_requires_grad(self.res_encoder, False)
            _set_requires_grad(self.res_renderer, False)
        elif stage in {"res", "residual"}:
            _set_requires_grad(self.phys_encoder, False)
            _set_requires_grad(self.phys_decoder, False)
            _set_requires_grad(self.res_encoder, True)
            _set_requires_grad(self.res_renderer, True)
        elif stage in {"joint", "all"}:
            _set_requires_grad(self.phys_encoder, True)
            _set_requires_grad(self.phys_decoder, True)
            _set_requires_grad(self.res_encoder, True)
            _set_requires_grad(self.res_renderer, True)
        else:
            raise ValueError(f"Unsupported stage={stage}. Expected phys, res, or joint.")

    def forward(
        self,
        x_01: torch.Tensor,
        *,
        sample: bool = True,
        sample_phys: bool | None = None,
        sample_res: bool | None = None,
        detach_base_for_residual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if sample_phys is None:
            sample_phys = bool(sample)
        if sample_res is None:
            sample_res = bool(sample)

        encoded_phys = self.encode_phys(x_01, sample=bool(sample_phys))
        phys = self.phys_decoder(encoded_phys["z_phys"], output_size=tuple(x_01.shape[-2:]))

        base_for_residual = phys["S_phys"].detach() if detach_base_for_residual else phys["S_phys"]
        residual_input = x_01 - base_for_residual

        encoded_res = self.encode_residual(residual_input, sample=bool(sample_res))
        delta = self.res_renderer(
            phys["S_phys"],
            phys["B_edge"],
            encoded_res["z_res"],
            output_size=tuple(x_01.shape[-2:]),
        )
        x_hat = torch.clamp(phys["S_phys"] + delta, 0.0, 1.0)

        outputs = {
            "posterior_phys": encoded_phys["posterior_phys"],
            "posterior_res": encoded_res["posterior_res"],
            "z_phys": encoded_phys["z_phys"],
            "z_res": encoded_res["z_res"],
            "z_joint": self.pack_joint_latent(encoded_phys["z_phys"], encoded_res["z_res"]),
            "residual_input": residual_input,
            "delta": delta,
            "x_hat": x_hat,
            "x_base": phys["S_phys"],
        }
        outputs.update(phys)
        return outputs
