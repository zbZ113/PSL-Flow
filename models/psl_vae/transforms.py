from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def _to_zero_one(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


@dataclass
class PhysFactorTransformConfig:
    a_low: float = 0.8
    a_high: float = 1.2
    temperature_log_scale: float = 4.0
    env_log_scale: float = 4.0
    delta_clip: float = 0.25
    residual_blur_kernel: int = 9
    residual_blur_sigma: float = 2.0
    residual_gate_power: float = 1.0
    residual_use_highpass: bool = True

    @classmethod
    def from_dict(cls, payload: Dict | None) -> "PhysFactorTransformConfig":
        data = asdict(cls())
        if payload is not None:
            for key, value in dict(payload).items():
                if key in data:
                    data[key] = value
        return cls(**data)


class GaussianBlur2d(torch.nn.Module):
    def __init__(self, kernel_size: int = 9, sigma: float = 2.0):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        if float(sigma) <= 0:
            raise ValueError("sigma must be positive.")
        self.kernel_size = kernel_size
        self.sigma = float(sigma)

        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * self.sigma * self.sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = int(x.shape[1])
        kernel = self.kernel.to(dtype=x.dtype, device=x.device).repeat(channels, 1, 1, 1)
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel, groups=channels)


class PhysFactorTransform:
    def __init__(self, config: PhysFactorTransformConfig | Dict | None = None):
        if config is None:
            config = PhysFactorTransformConfig()
        elif isinstance(config, dict):
            config = PhysFactorTransformConfig.from_dict(config)
        self.config = config
        self.blur = GaussianBlur2d(
            kernel_size=int(config.residual_blur_kernel),
            sigma=float(config.residual_blur_sigma),
        )

    @property
    def stack_channels(self) -> int:
        return 6

    def _normalize_recompose_mode(self, mode: str | None) -> str:
        mode = str(mode or "full").lower()
        aliases = {
            "full": "full",
            "b_delta": "full",
            "both": "full",
            "delta_only": "delta_only",
            "only_delta": "delta_only",
            "delta": "delta_only",
            "phys_only": "phys_only",
            "no_b_no_delta": "phys_only",
            "none": "phys_only",
        }
        if mode not in aliases:
            raise ValueError(
                f"Unsupported PSL-VAE recompose_mode={mode}. "
                "Expected one of: full, delta_only, phys_only."
            )
        return aliases[mode]

    def _normalize_temperature(self, t_rad: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.temperature_log_scale), 1e-6)
        return _to_zero_one(torch.log1p(torch.clamp(t_rad, min=0.0)) / scale)

    def _denormalize_temperature(self, t_norm: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.temperature_log_scale), 1e-6)
        return torch.expm1(torch.clamp(t_norm, 0.0, 1.0) * scale)

    def _normalize_env(self, r_env: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.env_log_scale), 1e-6)
        return _to_zero_one(torch.log1p(torch.clamp(r_env, min=0.0)) / scale)

    def _denormalize_env(self, r_norm: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.env_log_scale), 1e-6)
        return torch.expm1(torch.clamp(r_norm, 0.0, 1.0) * scale)

    def _normalize_a(self, a_map: torch.Tensor) -> torch.Tensor:
        a_lo = float(self.config.a_low)
        a_hi = float(self.config.a_high)
        return _to_zero_one((a_map - a_lo) / max(a_hi - a_lo, 1e-6))

    def _denormalize_a(self, a_norm: torch.Tensor) -> torch.Tensor:
        a_lo = float(self.config.a_low)
        a_hi = float(self.config.a_high)
        return torch.clamp(a_lo + (a_hi - a_lo) * torch.clamp(a_norm, 0.0, 1.0), a_lo, a_hi)

    def _normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        clip_value = max(float(self.config.delta_clip), 1e-6)
        delta = torch.clamp(delta, -clip_value, clip_value)
        return torch.clamp(0.5 * (delta / clip_value + 1.0), 0.0, 1.0)

    def _denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        clip_value = max(float(self.config.delta_clip), 1e-6)
        return clip_value * (2.0 * torch.clamp(delta_norm, 0.0, 1.0) - 1.0)

    def compute_s_phys(
        self,
        e_map: torch.Tensor,
        t_rad: torch.Tensor,
        r_env: torch.Tensor,
        a_map: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(a_map * (e_map * t_rad + (1.0 - e_map) * r_env), 0.0, 1.0)

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
        delta_res: torch.Tensor,
        *,
        recompose_mode: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        mode = self._normalize_recompose_mode(recompose_mode)
        if mode == "full":
            gate = torch.clamp(b_map, 0.0, 1.0).pow(float(self.config.residual_gate_power))
            y_hat = torch.clamp(s_phys + gate * delta_res, 0.0, 1.0)
        elif mode == "delta_only":
            gate = torch.ones_like(b_map)
            y_hat = torch.clamp(s_phys + delta_res, 0.0, 1.0)
        else:
            gate = torch.zeros_like(b_map)
            y_hat = torch.clamp(s_phys, 0.0, 1.0)
        return {
            "recompose_mode": mode,
            "gate": gate,
            "y_hat": y_hat,
        }

    def stack_from_teacher(self, teacher_out: Dict[str, torch.Tensor], x_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        e_map = _to_zero_one(teacher_out["e"])
        t_rad = torch.clamp(teacher_out["T_rad"], min=0.0)
        r_env = torch.clamp(teacher_out["R_env"], min=0.0)
        a_map = torch.clamp(teacher_out["A"], min=float(self.config.a_low), max=float(self.config.a_high))
        b_map = _to_zero_one(teacher_out["B_edge"])
        s_phys = self.compute_s_phys(e_map, t_rad, r_env, a_map)
        delta_target = self.compute_delta_target(x_01, s_phys)

        stack_01 = torch.cat([
            e_map,
            self._normalize_temperature(t_rad),
            self._normalize_env(r_env),
            self._normalize_a(a_map),
            b_map,
            self._normalize_delta(delta_target),
        ], dim=1)
        stack_tanh = stack_01 * 2.0 - 1.0
        return {
            "factor_stack_01": stack_01,
            "factor_stack_tanh": stack_tanh,
            "e": e_map,
            "T_rad": t_rad,
            "R_env": r_env,
            "A": a_map,
            "B_edge": b_map,
            "S_phys": s_phys,
            "delta_res": delta_target,
        }

    def decode_stack(self, stack_tanh: torch.Tensor, *, recompose_mode: str | None = None) -> Dict[str, torch.Tensor]:
        stack_01 = torch.clamp((stack_tanh + 1.0) * 0.5, 0.0, 1.0)
        if int(stack_01.shape[1]) != self.stack_channels:
            raise ValueError(f"Expected {self.stack_channels} decoded channels, got {stack_01.shape}.")

        e_map = stack_01[:, 0:1]
        t_rad = self._denormalize_temperature(stack_01[:, 1:2])
        r_env = self._denormalize_env(stack_01[:, 2:3])
        a_map = self._denormalize_a(stack_01[:, 3:4])
        b_map = stack_01[:, 4:5]
        delta_res = self._denormalize_delta(stack_01[:, 5:6])
        if bool(self.config.residual_use_highpass):
            delta_res = delta_res - self.blur(delta_res)
        s_phys = self.compute_s_phys(e_map, t_rad, r_env, a_map)
        composed = self.compose_image(s_phys, b_map, delta_res, recompose_mode=recompose_mode)
        return {
            "factor_stack_01": stack_01,
            "factor_stack_tanh": stack_tanh,
            "e": e_map,
            "T_rad": t_rad,
            "R_env": r_env,
            "A": a_map,
            "B_edge": b_map,
            "delta_res": delta_res,
            "gate": composed["gate"],
            "recompose_mode": composed["recompose_mode"],
            "S_phys": s_phys,
            "y_hat": composed["y_hat"],
        }


