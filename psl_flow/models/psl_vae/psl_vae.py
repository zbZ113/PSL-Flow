from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from diffusers.models import AutoencoderKL

from psl_flow.models.terb import TeRB
from psl_flow.utils.checkpoint import load_state_dict_flexible
from .transforms import PSLFactorTransform, PSLFactorTransformConfig


def _freeze(module: nn.Module | None) -> None:
    if module is not None:
        module.requires_grad_(False)
        module.eval()


def _build_autoencoder_kl(cfg: dict):
    cfg = dict(cfg)
    try:
        sig = inspect.signature(AutoencoderKL.__init__)
        valid = {k for k in sig.parameters.keys() if k not in {"self", "args", "kwargs"}}
        cfg = {k: v for k, v in cfg.items() if k in valid}
    except Exception:
        pass
    return AutoencoderKL(**cfg)


@dataclass
class PSLVAEConfig:
    in_channels: int = 5
    out_channels: int = 5
    down_block_types: Tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    latent_channels: int = 8
    norm_num_groups: int = 32
    sample_size: int = 256
    force_upcast: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    divisible: int = 8
    temperature_log_scale: float = 4.0
    env_log_scale: float = 4.0
    delta_clip: float = 0.25
    residual_gate_power: float = 1.0
    residual_blur_kernel: int = 9
    residual_blur_sigma: float = 2.0
    residual_use_highpass: bool = False

    @classmethod
    def from_dict(cls, payload: Dict | None) -> "PSLVAEConfig":
        data = asdict(cls())
        if payload:
            for key, value in dict(payload).items():
                if key in data:
                    data[key] = value
        tuple_keys = {"down_block_types", "up_block_types", "block_out_channels"}
        for key in tuple_keys:
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        return cls(**data)

    def autoencoder_config(self) -> Dict[str, object]:
        return {
            "in_channels": int(self.in_channels),
            "out_channels": int(self.out_channels),
            "down_block_types": list(self.down_block_types),
            "up_block_types": list(self.up_block_types),
            "block_out_channels": list(self.block_out_channels),
            "layers_per_block": int(self.layers_per_block),
            "act_fn": str(self.act_fn),
            "latent_channels": int(self.latent_channels),
            "norm_num_groups": int(self.norm_num_groups),
            "sample_size": int(self.sample_size),
            "force_upcast": bool(self.force_upcast),
            "use_quant_conv": bool(self.use_quant_conv),
            "use_post_quant_conv": bool(self.use_post_quant_conv),
            "mid_block_add_attention": bool(self.mid_block_add_attention),
        }

    def transform_config(self) -> PSLFactorTransformConfig:
        return PSLFactorTransformConfig(
            temperature_log_scale=float(self.temperature_log_scale),
            env_log_scale=float(self.env_log_scale),
            delta_clip=float(self.delta_clip),
            residual_gate_power=float(self.residual_gate_power),
            residual_blur_kernel=int(self.residual_blur_kernel),
            residual_blur_sigma=float(self.residual_blur_sigma),
            residual_use_highpass=bool(self.residual_use_highpass),
        )


def build_terb_teacher(teacher_cfg: Dict, *, ckpt_path: str = "", strict: bool = False) -> tuple[TeRB, dict]:
    teacher_cfg = dict(teacher_cfg)
    teacher = TeRB(
        smp_model=str(teacher_cfg.get("smp_model", "Unet")),
        smp_encoder=str(teacher_cfg.get("smp_encoder", "resnet18")),
        smp_encoder_weights=teacher_cfg.get("smp_encoder_weights", None),
        vnums=int(teacher_cfg.get("vnums", 4)),
        erme_kernel=int(teacher_cfg.get("erme_kernel", 5)),
        lambda_env_init=float(teacher_cfg.get("lambda_env_init", 0.1)),
    )
    info = {}
    resolved = str(ckpt_path or teacher_cfg.get("ckpt", ""))
    if resolved:
        info = load_state_dict_flexible(teacher, resolved, strict=strict)
    _freeze(teacher)
    return teacher, info


class PSLVAE(nn.Module):
    def __init__(self, config: PSLVAEConfig | Dict | None = None):
        super().__init__()
        if config is None:
            config = PSLVAEConfig()
        elif isinstance(config, dict):
            config = PSLVAEConfig.from_dict(config)
        self.config = config
        self.transform = PSLFactorTransform(config.transform_config())
        expected_channels = int(self.transform.stack_channels)
        if int(config.in_channels) != expected_channels or int(config.out_channels) != expected_channels:
            raise ValueError(
                "PSL-VAE expects five factor channels [T,e,R_env,B,Delta]; "
                f"got in_channels={config.in_channels}, out_channels={config.out_channels}."
            )
        self.vae = _build_autoencoder_kl(config.autoencoder_config())
        self.downsample_factor = 2 ** max(0, len(config.block_out_channels) - 1)
        self.divisible = int(config.divisible)
        object.__setattr__(self, "_teacher_q", None)

    @property
    def latent_channels(self) -> int:
        return int(self.config.latent_channels)

    def attach_teacher(self, teacher: TeRB | None) -> None:
        _freeze(teacher)
        object.__setattr__(self, "_teacher_q", teacher)

    def get_teacher(self) -> TeRB:
        teacher = getattr(self, "_teacher_q", None)
        if teacher is None:
            raise RuntimeError("PSLVAE requires an attached frozen TeR-B teacher.")
        return teacher

    def build_teacher_targets(self, x_01: torch.Tensor, *, teacher_out: Dict[str, torch.Tensor] | None = None):
        if teacher_out is None:
            teacher = self.get_teacher()
            if next(teacher.parameters()).device != x_01.device:
                teacher = teacher.to(x_01.device)
                _freeze(teacher)
                object.__setattr__(self, "_teacher_q", teacher)
            with torch.no_grad():
                teacher_out = teacher(x_01)
        return self.transform.stack_from_teacher(teacher_out, x_01)

    def encode_factor_stack(self, factor_stack_tanh: torch.Tensor, *, sample: bool = True):
        posterior = self.vae.encode(factor_stack_tanh).latent_dist
        z_phys = posterior.sample() if sample else posterior.mode()
        return {"posterior": posterior, "z_phys": z_phys}

    def decode_latents(self, z_phys: torch.Tensor):
        stack_tanh = self.vae.decode(z_phys).sample
        decoded = self.transform.decode_stack(stack_tanh)
        decoded["z_phys"] = z_phys
        return decoded

    def encode_from_ir(
        self,
        x_01: torch.Tensor,
        *,
        sample: bool = True,
        teacher_out: Dict[str, torch.Tensor] | None = None,
    ):
        targets = self.build_teacher_targets(x_01, teacher_out=teacher_out)
        encoded = self.encode_factor_stack(targets["factor_stack_tanh"], sample=sample)
        encoded["targets"] = targets
        return encoded

    def forward_from_ir(
        self,
        x_01: torch.Tensor,
        *,
        sample: bool = True,
        teacher_out: Dict[str, torch.Tensor] | None = None,
    ):
        encoded = self.encode_from_ir(x_01, sample=sample, teacher_out=teacher_out)
        decoded = self.decode_latents(encoded["z_phys"])
        decoded["posterior"] = encoded["posterior"]
        decoded["targets"] = encoded["targets"]
        return decoded

    def forward(self, x_01: torch.Tensor, *, sample: bool = True):
        return self.forward_from_ir(x_01, sample=sample)
