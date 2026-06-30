from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
import yaml
from diffusers.models import AutoencoderKL

from psl_flow.data import build_data_module, thermal_to_01
from psl_flow.evaluation.metrics import ssim_per_sample
from psl_flow.models.flow.routes import validate_route
from psl_flow.models.lpips import LPIPS
from psl_flow.models.psl_vae import PSLVAE, build_terb_teacher
from psl_flow.models.sit import sit_networks
from psl_flow.models.sit.transport import Sampler, create_transport
from psl_flow.utils.checkpoint import load_state_dict_flexible


def _repeat_to_three(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


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


def _load_rgb_vae_kl(model_cfg: dict[str, Any]) -> nn.Module:
    rgb_cfg = dict(model_cfg.get("rgb_vae_config", {}))
    rgb_path = str(model_cfg.get("rgb_vae_path", "") or "")
    rgb_repo = str(model_cfg.get("rgb_vae_repo", f"stabilityai/sd-vae-ft-{model_cfg.get('vae', 'ema')}"))
    local_only = bool(model_cfg.get("rgb_vae_local_files_only", False))
    default_local = str(model_cfg.get("rgb_vae_default_local_path", "/root/autodl-fs/sd-vae-ft-ema"))

    for candidate in (rgb_path, default_local):
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            if candidate_path.is_dir():
                print(f"[PSL-Flow] Load frozen RGB VAE from local diffusers dir: {candidate}")
                return AutoencoderKL.from_pretrained(candidate, local_files_only=True)
            model = _build_autoencoder_kl(rgb_cfg)
            info = load_state_dict_flexible(
                model,
                candidate,
                strict=False,
                strip_prefixes=("model.", "module.", "RGB_vae.", "rgb_vae."),
            )
            print(f"[PSL-Flow] Loaded frozen RGB VAE state dict: {info}")
            return model

    if rgb_path and "/" in rgb_path and not Path(rgb_path).is_absolute():
        print(f"[PSL-Flow] Load frozen RGB VAE from repo/id: {rgb_path}")
        return AutoencoderKL.from_pretrained(rgb_path, local_files_only=local_only)

    print(f"[PSL-Flow] Load frozen RGB VAE from repo/id: {rgb_repo}")
    return AutoencoderKL.from_pretrained(rgb_repo, local_files_only=local_only)


def _load_state_or_pretrained_vae(
    *,
    cfg: dict[str, Any],
    path: str,
    repo: str = "",
    local_files_only: bool = False,
    label: str,
) -> nn.Module:
    if path:
        candidate = Path(path)
        if candidate.exists():
            if candidate.is_dir():
                print(f"[{label}] Load frozen VAE from local diffusers dir: {path}")
                return AutoencoderKL.from_pretrained(path, local_files_only=True)
            model = _build_autoencoder_kl(cfg)
            info = load_state_dict_flexible(
                model,
                path,
                strict=False,
                strip_prefixes=("model.", "module.", "thermal_vae.", "vae.", "klvae."),
            )
            print(f"[{label}] Loaded frozen VAE state dict: {info}")
            return model
        if "/" in path and not candidate.is_absolute():
            print(f"[{label}] Load frozen VAE from repo/id: {path}")
            return AutoencoderKL.from_pretrained(path, local_files_only=local_files_only)
    if repo:
        print(f"[{label}] Load frozen VAE from repo/id: {repo}")
        return AutoencoderKL.from_pretrained(repo, local_files_only=local_files_only)
    print(f"[{label}] Build VAE from config")
    return _build_autoencoder_kl(cfg)


def _psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = (pred - target).square().flatten(1).mean(dim=1).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def _require_float(model_cfg: dict[str, Any], key: str, *, hint: str) -> float:
    value = model_cfg.get(key, None)
    if value in (None, ""):
        raise ValueError(f"model.model_config.{key} is required. {hint}")
    return float(value)


def _decode_thermal_to_01(decoded: torch.Tensor) -> torch.Tensor:
    if decoded.shape[1] > 1:
        decoded = decoded.mean(dim=1, keepdim=True)
    return thermal_to_01(decoded)


class PSLFlowLightningModule(pl.LightningModule):
    """Paper-route SiT wrapper.

    Target latent: z_phys = PSLVAE([T,e,R_env,B,Delta]).
    Condition latent: z_vis = RGB KL-VAE(x_vis).
    Trainable module: SiT only.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = dict(config.get("model", {}).get("model_config", {}))
        self.model_cfg = model_cfg

        psl_cfg = dict(model_cfg.get("psl_vae_config", {}))
        self.psl_vae = PSLVAE(psl_cfg)
        teacher_cfg = dict(model_cfg.get("teacher", {}))
        if teacher_cfg.get("ckpt"):
            teacher, info = build_terb_teacher(
                teacher_cfg,
                ckpt_path=str(teacher_cfg.get("ckpt")),
                strict=bool(teacher_cfg.get("strict_load", False)),
            )
            print(f"[PSL-Flow] Loaded frozen TeR-B teacher: {info}")
            self.psl_vae.attach_teacher(teacher)
        psl_vae_ckpt = str(model_cfg.get("psl_vae_ckpt", "") or model_cfg.get("vae_path", ""))
        if psl_vae_ckpt:
            info = load_state_dict_flexible(
                self.psl_vae,
                psl_vae_ckpt,
                strict=False,
                strip_prefixes=("model.", "module.", "psl_vae."),
            )
            print(f"[PSL-Flow] Loaded frozen PSL-VAE: {info}")
        _freeze(self.psl_vae)

        self.rgb_vae = _load_rgb_vae_kl(model_cfg)
        _freeze(self.rgb_vae)

        sit_cfg = dict(model_cfg.get("sit_config", {}))
        injection_args = copy.deepcopy(sit_cfg.get("injection_args", {"injection_method": "concat", "replace_RGB": False}))
        injection_args.setdefault("rgb_in_chans", int(model_cfg.get("rgb_vae_latent_channels", 4)))
        arch = str(sit_cfg.get("arch", "L"))
        patch_size = int(sit_cfg.get("patch_size", 2))
        self.sit = sit_networks.SiT_models[f"SiT-{arch}/{patch_size}"](
            in_channels=int(self.psl_vae.latent_channels),
            num_classes=int(sit_cfg.get("num_classes", 1000)),
            injection_args=injection_args,
            repa=False,
        )
        self.ema = copy.deepcopy(self.sit).eval()
        self.ema.requires_grad_(False)
        self.transport = create_transport(**dict(model_cfg.get("transport_config", {"path_type": "Linear", "prediction": "velocity", "loss_weight": None})))
        self.sampler = Sampler(self.transport)
        self.thermal_normalizer = _require_float(
            model_cfg,
            "thermal_normalizer",
            hint=(
                "Run python -m psl_flow.models.psl_vae.prepare_psl_flow_config after PSL-VAE training "
                "so it can estimate thermal_normalizer = 1 / latent_std and patch the PSL-Flow config."
            ),
        )
        self.rgb_normalizer = float(model_cfg.get("rgb_normalizer", 0.18215) or 1.0)
        self.cfg_scale = float(model_cfg.get("cfg_scale", 1.0))
        self.sample_ode = dict(model_cfg.get("sample_ode", {"sampling_method": "dopri5", "num_steps": 50}))
        self.eval_lpips = LPIPS().eval() if bool(model_cfg.get("eval_lpips", True)) else None
        _freeze(self.eval_lpips)
        self.optimizer_cfg = dict(config.get("training", {}).get("optimizer", {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0}))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.sit.parameters(),
            lr=float(self.optimizer_cfg.get("lr", 1e-4)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
        )

    def _update_ema(self, decay: float = 0.9999) -> None:
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), self.sit.parameters()):
                ema_p.mul_(decay).add_(p, alpha=1.0 - decay)

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.rgb_vae.encode(rgb).latent_dist
            z_vis = posterior.sample()
            return z_vis * self.rgb_normalizer

    def _encode_phys(self, thermal: torch.Tensor) -> tuple[torch.Tensor, dict]:
        thermal_01 = thermal_to_01(thermal)
        with torch.no_grad():
            encoded = self.psl_vae.encode_from_ir(thermal_01, sample=True)
            z = encoded["z_phys"] * self.thermal_normalizer
        return z, encoded

    def training_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        z_phys, encoded = self._encode_phys(thermal)
        z_vis = self._encode_rgb(rgb)
        kwargs = {"y": dataset_idx, "x_RGB": z_vis}
        loss_dict = self.transport.training_losses(self.sit, z_phys, kwargs)
        loss = loss_dict["loss"].mean()
        self.log("train/loss_flow", loss, prog_bar=True, sync_dist=True)
        self.log("train/z_phys_std", z_phys.std(unbiased=False), sync_dist=True)
        self.log("train/z_phys_mean", z_phys.mean(), sync_dist=True)
        self.log("train/z_vis_std", z_vis.std(unbiased=False), sync_dist=True)
        if "xt" in loss_dict:
            self.log("train/z_t_std", loss_dict["xt"].std(unbiased=False), sync_dist=True)
        self._update_ema()
        return loss

    def _sample_latent(self, rgb: torch.Tensor, dataset_idx: torch.Tensor, *, use_ema: bool = True) -> torch.Tensor:
        z_vis = self._encode_rgb(rgb)
        factor = int(getattr(self.psl_vae, "downsample_factor", 8))
        h = rgb.shape[-2] // factor
        w = rgb.shape[-1] // factor
        z0 = torch.randn(rgb.shape[0], int(self.psl_vae.latent_channels), h, w, device=rgb.device, dtype=z_vis.dtype)
        sample_fn = self.sampler.sample_ode(**self.sample_ode)
        model = self.ema if use_ema else self.sit
        with torch.no_grad():
            samples = sample_fn(z0, model.forward, y=dataset_idx, x_RGB=z_vis)[-1]
        return samples / self.thermal_normalizer

    def generate(self, rgb: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        z_phys = self._sample_latent(rgb, dataset_idx, use_ema=True)
        with torch.no_grad():
            decoded = self.psl_vae.decode_latents(z_phys)
        return decoded["y_hat"]

    def validation_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self.generate(rgb, dataset_idx)
        self.log("val/PSNR", _psnr(pred, target).mean(), sync_dist=True)
        self.log("val/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True)
        if self.eval_lpips is not None:
            pred_lpips = _repeat_to_three(pred * 2.0 - 1.0)
            target_lpips = _repeat_to_three(target * 2.0 - 1.0)
            self.log("val/LPIPS", self.eval_lpips(pred_lpips, target_lpips).mean(), sync_dist=True)
        return pred

    def test_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self.generate(rgb, dataset_idx)
        self.log("test/PSNR", _psnr(pred, target).mean(), sync_dist=True)
        self.log("test/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True)
        if self.eval_lpips is not None:
            pred_lpips = _repeat_to_three(pred * 2.0 - 1.0)
            target_lpips = _repeat_to_three(target * 2.0 - 1.0)
            self.log("test/LPIPS", self.eval_lpips(pred_lpips, target_lpips).mean(), sync_dist=True)
        return pred


class KLVaeSiTLightningModule(pl.LightningModule):
    """SiT ablation using KL-VAE latents on both thermal target and RGB condition."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = dict(config.get("model", {}).get("model_config", {}))
        self.model_cfg = model_cfg

        thermal_cfg = dict(model_cfg.get("thermal_vae_config", model_cfg.get("vae_config", {})))
        thermal_path = str(model_cfg.get("thermal_vae_ckpt", "") or model_cfg.get("vae_path", ""))
        thermal_repo = str(model_cfg.get("thermal_vae_repo", ""))
        local_only = bool(model_cfg.get("thermal_vae_local_files_only", False))
        self.thermal_vae = _load_state_or_pretrained_vae(
            cfg=thermal_cfg,
            path=thermal_path,
            repo=thermal_repo,
            local_files_only=local_only,
            label="KLVAE-SiT thermal",
        )
        _freeze(self.thermal_vae)

        self.rgb_vae = _load_rgb_vae_kl(model_cfg)
        _freeze(self.rgb_vae)

        self.thermal_latent_channels = int(
            model_cfg.get(
                "thermal_vae_latent_channels",
                thermal_cfg.get("latent_channels", getattr(getattr(self.thermal_vae, "config", None), "latent_channels", 4)),
            )
        )
        self.rgb_latent_channels = int(model_cfg.get("rgb_vae_latent_channels", 4))
        self.downsample_factor = int(model_cfg.get("downsample_factor", 8))

        sit_cfg = dict(model_cfg.get("sit_config", {}))
        injection_args = copy.deepcopy(sit_cfg.get("injection_args", {"injection_method": "concat", "replace_RGB": False}))
        injection_args.setdefault("rgb_in_chans", self.rgb_latent_channels)
        arch = str(sit_cfg.get("arch", "L"))
        patch_size = int(sit_cfg.get("patch_size", 2))
        self.sit = sit_networks.SiT_models[f"SiT-{arch}/{patch_size}"](
            in_channels=self.thermal_latent_channels,
            num_classes=int(sit_cfg.get("num_classes", 1000)),
            injection_args=injection_args,
            repa=False,
        )
        self.ema = copy.deepcopy(self.sit).eval()
        self.ema.requires_grad_(False)
        self.transport = create_transport(**dict(model_cfg.get("transport_config", {"path_type": "Linear", "prediction": "velocity", "loss_weight": None})))
        self.sampler = Sampler(self.transport)
        self.thermal_normalizer = float(model_cfg.get("thermal_normalizer", 1.0) or 1.0)
        self.rgb_normalizer = float(model_cfg.get("rgb_normalizer", 0.18215) or 1.0)
        self.sample_ode = dict(model_cfg.get("sample_ode", {"sampling_method": "dopri5", "num_steps": 50}))
        self.eval_lpips = LPIPS().eval() if bool(model_cfg.get("eval_lpips", True)) else None
        _freeze(self.eval_lpips)
        self.optimizer_cfg = dict(config.get("training", {}).get("optimizer", {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0}))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.sit.parameters(),
            lr=float(self.optimizer_cfg.get("lr", 1e-4)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
        )

    def _update_ema(self, decay: float = 0.9999) -> None:
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), self.sit.parameters()):
                ema_p.mul_(decay).add_(p, alpha=1.0 - decay)

    def _thermal_for_vae(self, thermal: torch.Tensor) -> torch.Tensor:
        in_channels = int(getattr(self.thermal_vae, "in_channels", 1))
        if in_channels == 3 and thermal.shape[1] == 1:
            return thermal.repeat(1, 3, 1, 1)
        return thermal

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.rgb_vae.encode(rgb).latent_dist
            return posterior.sample() * self.rgb_normalizer

    def _encode_thermal(self, thermal: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.thermal_vae.encode(self._thermal_for_vae(thermal)).latent_dist
            return posterior.sample() * self.thermal_normalizer

    def training_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        z_thermal = self._encode_thermal(thermal)
        z_vis = self._encode_rgb(rgb)
        loss_dict = self.transport.training_losses(self.sit, z_thermal, {"y": dataset_idx, "x_RGB": z_vis})
        loss = loss_dict["loss"].mean()
        self.log("train/loss_flow", loss, prog_bar=True, sync_dist=True)
        self.log("train/z_thermal_std", z_thermal.std(unbiased=False), sync_dist=True)
        self.log("train/z_vis_std", z_vis.std(unbiased=False), sync_dist=True)
        self._update_ema()
        return loss

    def _sample_latent(self, rgb: torch.Tensor, dataset_idx: torch.Tensor, *, use_ema: bool = True) -> torch.Tensor:
        z_vis = self._encode_rgb(rgb)
        h = rgb.shape[-2] // self.downsample_factor
        w = rgb.shape[-1] // self.downsample_factor
        z0 = torch.randn(rgb.shape[0], self.thermal_latent_channels, h, w, device=rgb.device, dtype=z_vis.dtype)
        sample_fn = self.sampler.sample_ode(**self.sample_ode)
        model = self.ema if use_ema else self.sit
        with torch.no_grad():
            samples = sample_fn(z0, model.forward, y=dataset_idx, x_RGB=z_vis)[-1]
        return samples / self.thermal_normalizer

    def generate(self, rgb: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        z_thermal = self._sample_latent(rgb, dataset_idx, use_ema=True)
        with torch.no_grad():
            decoded = self.thermal_vae.decode(z_thermal).sample
        return _decode_thermal_to_01(decoded)

    def validation_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self.generate(rgb, dataset_idx)
        self.log("val/PSNR", _psnr(pred, target).mean(), sync_dist=True)
        self.log("val/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True)
        if self.eval_lpips is not None:
            self.log(
                "val/LPIPS",
                self.eval_lpips(_repeat_to_three(pred * 2.0 - 1.0), _repeat_to_three(target * 2.0 - 1.0)).mean(),
                sync_dist=True,
            )
        return pred

    def test_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self.generate(rgb, dataset_idx)
        self.log("test/PSNR", _psnr(pred, target).mean(), sync_dist=True)
        self.log("test/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True)
        if self.eval_lpips is not None:
            self.log(
                "test/LPIPS",
                self.eval_lpips(_repeat_to_three(pred * 2.0 - 1.0), _repeat_to_three(target * 2.0 - 1.0)).mean(),
                sync_dist=True,
            )
        return pred


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _plain_metrics(results: list[dict[str, Any]]) -> list[dict[str, float]]:
    plain: list[dict[str, float]] = []
    for row in results:
        plain.append({key: float(value) for key, value in row.items()})
    return plain


def _emit_metrics(label: str, results: list[dict[str, Any]], metrics_json: str | None) -> None:
    plain = _plain_metrics(results)
    print(f"[{label}] metrics: {json.dumps(plain, ensure_ascii=False, sort_keys=True)}")
    if metrics_json:
        out = Path(metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(plain, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        print(f"[{label}] wrote metrics: {out}")


def _trainer_from_args(config: dict[str, Any], args: argparse.Namespace, *, with_checkpoints: bool = False) -> pl.Trainer:
    training = dict(config.get("training", {}))
    callbacks = []
    if with_checkpoints:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                every_n_train_steps=args.checkpoint_every_n_train_steps,
                save_top_k=0 if args.limit_val_batches == 0 else int(training.get("checkpoint_save_top_k", 1)),
            )
        )
    return pl.Trainer(
        max_epochs=int(training.get("num_epochs", 1000)),
        max_steps=args.max_steps if args.max_steps is not None else int(training.get("max_steps", -1)),
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
        precision="16-mixed" if bool(training.get("mixed_precision", False)) else "32-true",
        callbacks=callbacks,
        logger=False if bool(training.get("disable_logger", False)) else True,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else training.get("limit_val_batches", None),
        check_val_every_n_epoch=args.check_val_every_n_epoch
        if args.check_val_every_n_epoch is not None
        else int(training.get("check_val_every_n_epoch", 1)),
        accumulate_grad_batches=int(training.get("gradient_accumulation", 1)),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
    )


def main() -> None:
    parser = argparse.ArgumentParser("Train/test paper-aligned PSL-Flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["fit", "validate", "test"], default="fit")
    parser.add_argument("--default-root-dir", default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-n-train-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    route = validate_route(config.get("route") or config.get("model", {}).get("route") or config.get("model", {}).get("model_arch"))
    data = build_data_module(config)
    module = PSLFlowLightningModule(config) if route == "psl_flow" else KLVaeSiTLightningModule(config)
    trainer = _trainer_from_args(config, args, with_checkpoints=args.mode == "fit")
    if args.mode == "fit":
        trainer.fit(module, datamodule=data, ckpt_path=args.resume_from)
        if trainer.is_global_zero:
            final_ckpt = Path(args.default_root_dir or trainer.default_root_dir) / "checkpoints" / "last.ckpt"
            final_ckpt.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(final_ckpt))
            print(f"[{route}] saved final checkpoint: {final_ckpt}")
    elif args.mode == "validate":
        results = trainer.validate(module, datamodule=data, ckpt_path=args.ckpt)
        if trainer.is_global_zero:
            _emit_metrics(f"{route} validation", results, args.metrics_json)
    else:
        results = trainer.test(module, datamodule=data, ckpt_path=args.ckpt)
        if trainer.is_global_zero:
            _emit_metrics(f"{route} test", results, args.metrics_json)


if __name__ == "__main__":
    main()
