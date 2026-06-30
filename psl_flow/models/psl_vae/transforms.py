from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


def _to_zero_one(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


class GaussianBlur2d(nn.Module):
    def __init__(self, kernel_size: int = 9, sigma: float = 2.0):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            self.register_buffer("kernel", torch.ones(1, 1, 1, 1), persistent=False)
            self.pad = 0
            return
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * float(sigma) ** 2))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        self.register_buffer("kernel", kernel.view(1, 1, kernel_size, kernel_size), persistent=False)
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad <= 0:
            return x
        weight = self.kernel.to(device=x.device, dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        return torch.nn.functional.conv2d(
            torch.nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect"),
            weight,
            groups=x.shape[1],
        )


@dataclass
class PSLFactorTransformConfig:
    temperature_log_scale: float = 4.0
    env_log_scale: float = 4.0
    delta_clip: float = 0.25
    residual_gate_power: float = 1.0
    residual_blur_kernel: int = 9
    residual_blur_sigma: float = 2.0
    residual_use_highpass: bool = False

    @classmethod
    def from_dict(cls, payload: Dict | None) -> "PSLFactorTransformConfig":
        data = asdict(cls())
        if payload:
            for key, value in dict(payload).items():
                if key in data:
                    data[key] = value
        return cls(**data)


class PSLFactorTransform(nn.Module):
    """Build and decode the paper five-channel stack [T, e, R_env, B, Delta]."""

    def __init__(self, config: PSLFactorTransformConfig | Dict | None = None):
        super().__init__()
        if config is None:
            config = PSLFactorTransformConfig()
        elif isinstance(config, dict):
            config = PSLFactorTransformConfig.from_dict(config)
        self.config = config
        self.blur = GaussianBlur2d(config.residual_blur_kernel, config.residual_blur_sigma)

    @property
    def stack_channels(self) -> int:
        return 5

    def _normalize_temperature(self, t_rad: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.temperature_log_scale), 1e-6)
        return _to_zero_one(torch.log1p(torch.clamp(t_rad, min=0.0)) / scale)

    def _denormalize_temperature(self, t_norm: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.temperature_log_scale), 1e-6)
        return torch.expm1(_to_zero_one(t_norm) * scale)

    def _normalize_env(self, r_env: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.env_log_scale), 1e-6)
        return _to_zero_one(torch.log1p(torch.clamp(r_env, min=0.0)) / scale)

    def _denormalize_env(self, r_norm: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.env_log_scale), 1e-6)
        return torch.expm1(_to_zero_one(r_norm) * scale)

    def _normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        clip_value = max(float(self.config.delta_clip), 1e-6)
        delta = torch.clamp(delta, -clip_value, clip_value)
        return torch.clamp(0.5 * (delta / clip_value + 1.0), 0.0, 1.0)

    def _denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        clip_value = max(float(self.config.delta_clip), 1e-6)
        return clip_value * (2.0 * _to_zero_one(delta_norm) - 1.0)

    @staticmethod
    def compute_s_phys(e_map: torch.Tensor, t_rad: torch.Tensor, r_env: torch.Tensor) -> torch.Tensor:
        return torch.clamp(e_map * t_rad + (1.0 - e_map) * r_env, 0.0, 1.0)

    def compute_delta_target(self, x_01: torch.Tensor, s_phys: torch.Tensor) -> torch.Tensor:
        residual = x_01 - s_phys
        if bool(self.config.residual_use_highpass):
            residual = residual - self.blur(residual)
        clip_value = max(float(self.config.delta_clip), 1e-6)
        return torch.clamp(residual, -clip_value, clip_value)

    def compose_image(
        self,
        s_phys: torch.Tensor,
        b_map: torch.Tensor,
        delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gate = _to_zero_one(b_map).pow(float(self.config.residual_gate_power))
        y_hat = torch.clamp(s_phys + gate * delta, 0.0, 1.0)
        return {"gate": gate, "y_hat": y_hat}

    def stack_from_teacher(self, teacher_out: Dict[str, torch.Tensor], x_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_rad = torch.clamp(teacher_out["T_rad"], min=0.0)
        e_map = _to_zero_one(teacher_out["e"])
        r_env = torch.clamp(teacher_out["R_env"], min=0.0)
        b_map = _to_zero_one(teacher_out["B_edge"])
        s_phys = self.compute_s_phys(e_map, t_rad, r_env)
        delta = self.compute_delta_target(x_01, s_phys)

        stack_01 = torch.cat(
            [
                self._normalize_temperature(t_rad),
                e_map,
                self._normalize_env(r_env),
                b_map,
                self._normalize_delta(delta),
            ],
            dim=1,
        )
        return {
            "factor_stack_01": stack_01,
            "factor_stack_tanh": stack_01 * 2.0 - 1.0,
            "T_rad": t_rad,
            "e": e_map,
            "R_env": r_env,
            "B_edge": b_map,
            "S_phys": s_phys,
            "delta_res": delta,
        }

    def decode_stack(self, stack_tanh: torch.Tensor) -> Dict[str, torch.Tensor]:
        stack_01 = _to_zero_one((stack_tanh + 1.0) * 0.5)
        if int(stack_01.shape[1]) != self.stack_channels:
            raise ValueError(f"Expected {self.stack_channels} PSL channels, got {stack_01.shape}.")
        t_rad = self._denormalize_temperature(stack_01[:, 0:1])
        e_map = stack_01[:, 1:2]
        r_env = self._denormalize_env(stack_01[:, 2:3])
        b_map = stack_01[:, 3:4]
        delta = self._denormalize_delta(stack_01[:, 4:5])
        if bool(self.config.residual_use_highpass):
            delta = delta - self.blur(delta)
        s_phys = self.compute_s_phys(e_map, t_rad, r_env)
        composed = self.compose_image(s_phys, b_map, delta)
        return {
            "factor_stack_01": stack_01,
            "factor_stack_tanh": stack_tanh,
            "T_rad": t_rad,
            "e": e_map,
            "R_env": r_env,
            "B_edge": b_map,
            "delta_res": delta,
            "S_phys": s_phys,
            "gate": composed["gate"],
            "y_hat": composed["y_hat"],
        }

