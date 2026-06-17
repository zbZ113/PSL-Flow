from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.physics import TeR_B, load_module_checkpoint, sobel_mag

from .model import split_joint_latent


def _freeze(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


class GaussianBlur2d(nn.Module):
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


@dataclass
class PhysVAERLossConfig:
    w_e: float = 1.0
    w_t: float = 1.0
    w_r: float = 1.0
    w_a: float = 0.1
    w_b: float = 0.25
    lambda_phys: float = 1.0
    lambda_base: float = 0.5
    lambda_sphys: float = 1.0
    lambda_img: float = 1.0
    lambda_edge: float = 0.0
    delta_lowfreq_weight: float = 1.0
    delta_l1_weight: float = 0.1
    kl_phys_weight: float = 1e-4
    kl_res_weight: float = 5e-4
    blur_kernel_size: int = 9
    blur_sigma: float = 2.0

    @classmethod
    def from_dict(cls, payload: Dict) -> "PhysVAERLossConfig":
        base = cls()
        data = asdict(base)
        for key, value in dict(payload).items():
            if key in data:
                data[key] = value
        return cls(**data)


@dataclass
class JointFlowLossConfig:
    phys_weight: float = 1.0
    res_weight: float = 0.3


def build_teacher_q(
    teacher_cfg: Dict,
    *,
    ckpt_path: str = "",
    strict: bool = False,
) -> Tuple[TeR_B, Dict[str, object]]:
    teacher_cfg = dict(teacher_cfg)
    a_low_range = tuple(teacher_cfg.get("a_low_range", [0.8, 1.2]))
    teacher = TeR_B(
        smp_model=str(teacher_cfg.get("smp_model", "Unet")),
        smp_encoder=str(teacher_cfg.get("smp_encoder", "resnet18")),
        smp_encoder_weights=teacher_cfg.get("smp_encoder_weights", None),
        vnums=int(teacher_cfg.get("vnums", 4)),
        erme_kernel=int(teacher_cfg.get("erme_kernel", 5)),
        lambda_env_init=float(teacher_cfg.get("lambda_env_init", 0.1)),
        a_low_range=(float(a_low_range[0]), float(a_low_range[1])),
    )
    load_info: Dict[str, object] = {}
    resolved_ckpt = str(ckpt_path or teacher_cfg.get("ckpt", ""))
    if resolved_ckpt:
        load_info = load_module_checkpoint(teacher, resolved_ckpt, strict=strict)
    _freeze(teacher)
    return teacher, load_info


class PhysVAERLoss(nn.Module):
    def __init__(self, teacher: TeR_B, config: PhysVAERLossConfig | Dict | None = None):
        super().__init__()
        if config is None:
            config = PhysVAERLossConfig()
        elif isinstance(config, dict):
            config = PhysVAERLossConfig.from_dict(config)
        self.config = config
        self.teacher = teacher
        _freeze(self.teacher)
        self.blur = GaussianBlur2d(
            kernel_size=int(config.blur_kernel_size),
            sigma=float(config.blur_sigma),
        )

    def _zero(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = outputs["x_hat"]
        return ref.new_tensor(0.0)

    def _resolve_phys_component_weights(self, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        base = {
            "e": ref.new_tensor(float(self.config.w_e)),
            "t": ref.new_tensor(float(self.config.w_t)),
            "r": ref.new_tensor(float(self.config.w_r)),
            "a": ref.new_tensor(float(self.config.w_a)),
            "b": ref.new_tensor(float(self.config.w_b)),
        }
        return base

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        x_01: torch.Tensor,
        *,
        stage: str = "joint",
        teacher_targets: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        stage = str(stage).lower()
        with torch.no_grad():
            targets = teacher_targets if teacher_targets is not None else self.teacher(x_01)

        loss_phys_e = F.l1_loss(outputs["e"], targets["e"])
        loss_phys_t = F.l1_loss(outputs["T_rad"], targets["T_rad"])
        loss_phys_r = F.l1_loss(outputs["R_env"], targets["R_env"])
        loss_phys_a = F.l1_loss(outputs["A"], targets["A"])
        loss_phys_b = F.l1_loss(outputs["B_edge"], targets["B_edge"])
        phys_weights = self._resolve_phys_component_weights(targets["S_phys"])
        loss_phys = (
            phys_weights["e"] * loss_phys_e
            + phys_weights["t"] * loss_phys_t
            + phys_weights["r"] * loss_phys_r
            + phys_weights["a"] * loss_phys_a
            + phys_weights["b"] * loss_phys_b
        )

        loss_base = F.l1_loss(self.blur(outputs["S_phys"]), self.blur(targets["S_phys"]))
        loss_sphys = F.l1_loss(outputs["S_phys"], targets["S_phys"])
        loss_img = F.l1_loss(outputs["x_hat"], x_01)
        loss_edge = F.l1_loss(sobel_mag(outputs["x_hat"]), sobel_mag(x_01))
        loss_delta_lowfreq = self.blur(outputs["delta"]).abs().mean()
        loss_delta_l1 = outputs["delta"].abs().mean()

        zero = self._zero(outputs)
        posterior_phys = outputs.get("posterior_phys", None)
        posterior_res = outputs.get("posterior_res", None)
        loss_kl_phys = posterior_phys.kl().mean() if posterior_phys is not None else zero
        loss_kl_res = posterior_res.kl().mean() if posterior_res is not None else zero

        if stage in {"phys", "physics"}:
            loss_total = (
                float(self.config.lambda_phys) * loss_phys
                + float(self.config.lambda_base) * loss_base
                + float(self.config.lambda_sphys) * loss_sphys
                + float(self.config.kl_phys_weight) * loss_kl_phys
            )
        elif stage in {"res", "residual"}:
            loss_total = (
                float(self.config.lambda_img) * loss_img
                + float(self.config.lambda_edge) * loss_edge
                + float(self.config.delta_lowfreq_weight) * loss_delta_lowfreq
                + float(self.config.delta_l1_weight) * loss_delta_l1
                + float(self.config.kl_res_weight) * loss_kl_res
            )
        elif stage in {"joint", "all"}:
            loss_total = (
                float(self.config.lambda_phys) * loss_phys
                + float(self.config.lambda_base) * loss_base
                + float(self.config.lambda_sphys) * loss_sphys
                + float(self.config.lambda_img) * loss_img
                + float(self.config.lambda_edge) * loss_edge
                + float(self.config.delta_lowfreq_weight) * loss_delta_lowfreq
                + float(self.config.delta_l1_weight) * loss_delta_l1
                + float(self.config.kl_phys_weight) * loss_kl_phys
                + float(self.config.kl_res_weight) * loss_kl_res
            )
        else:
            raise ValueError(f"Unsupported stage={stage}. Expected phys, res, or joint.")

        return {
            "loss_total": loss_total,
            "loss_phys": loss_phys,
            "loss_phys_e": loss_phys_e,
            "loss_phys_t": loss_phys_t,
            "loss_phys_r": loss_phys_r,
            "loss_phys_a": loss_phys_a,
            "loss_phys_b": loss_phys_b,
            "loss_base": loss_base,
            "loss_sphys": loss_sphys,
            "loss_img": loss_img,
            "loss_edge": loss_edge,
            "loss_delta_lowfreq": loss_delta_lowfreq,
            "loss_delta_l1": loss_delta_l1,
            "loss_kl_phys": loss_kl_phys,
            "loss_kl_res": loss_kl_res,
            "weight_phys_e": phys_weights["e"].detach(),
            "weight_phys_t": phys_weights["t"].detach(),
            "weight_phys_r": phys_weights["r"].detach(),
            "weight_phys_a": phys_weights["a"].detach(),
            "weight_phys_b": phys_weights["b"].detach(),
        }


def weighted_joint_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    phys_channels: int,
    res_channels: int,
    config: JointFlowLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    cfg = config if config is not None else JointFlowLossConfig()
    pred_phys, pred_res = split_joint_latent(
        pred,
        phys_channels=int(phys_channels),
        res_channels=int(res_channels),
    )
    tgt_phys, tgt_res = split_joint_latent(
        target,
        phys_channels=int(phys_channels),
        res_channels=int(res_channels),
    )
    loss_phys = F.mse_loss(pred_phys, tgt_phys)
    loss_res = F.mse_loss(pred_res, tgt_res)
    loss_total = float(cfg.phys_weight) * loss_phys + float(cfg.res_weight) * loss_res
    return {
        "loss_total": loss_total,
        "loss_phys": loss_phys,
        "loss_res": loss_res,
    }
