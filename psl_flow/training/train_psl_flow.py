from __future__ import annotations

import argparse
import copy
import inspect
import json
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile
import yaml
from diffusers.models import AutoencoderKL

from psl_flow.data import build_data_module, thermal_to_01
from psl_flow.evaluation.metrics import ssim_per_sample
from psl_flow.models.flow.routes import validate_route
from psl_flow.models.lpips import LPIPS
from psl_flow.models.psl_vae import PSLVAE, build_terb_teacher
from psl_flow.models.sit import sit_networks
from psl_flow.models.sit.transport import Sampler, create_transport
from psl_flow.training.callbacks import (
    build_management_callbacks,
    configure_training_runtime,
    copy_best_checkpoint_alias,
)
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
    default_local = str(model_cfg.get("rgb_vae_default_local_path", "") or "")

    def accessible_path(candidate: str) -> Path | None:
        path = Path(candidate)
        try:
            if path.exists():
                return path
        except OSError as exc:
            print(f"[PSL-Flow][WARN] Skip inaccessible RGB VAE path: {candidate} ({type(exc).__name__}: {exc})")
        return None

    for candidate in (rgb_path, default_local):
        if not candidate:
            continue
        candidate_path = accessible_path(candidate)
        if candidate_path is not None:
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

    if local_only:
        raise FileNotFoundError(
            "RGB VAE local loading is required, but no accessible rgb_vae_path or "
            f"rgb_vae_default_local_path was found. rgb_vae_path={rgb_path!r}, "
            f"rgb_vae_default_local_path={default_local!r}"
        )

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


def _rgb_to_01(rgb: torch.Tensor) -> torch.Tensor:
    return torch.clamp(rgb * 0.5 + 0.5, 0.0, 1.0)


def _to_fid_image(x: torch.Tensor) -> torch.Tensor:
    return (_repeat_to_three(x.detach().clamp(0.0, 1.0)) * 255.0).to(torch.uint8)


def _make_fid_metric(enabled: bool, label: str) -> nn.Module | None:
    if not enabled:
        return None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as exc:
        print(f"[FID][WARN] eval_fid requested for {label}, but torchmetrics FID is unavailable: {exc}")
        return None
    try:
        return FrechetInceptionDistance(feature=2048, normalize=False)
    except Exception as exc:
        print(f"[FID][WARN] eval_fid requested for {label}, but FID metric could not be initialized: {exc}")
        return None


def _is_optional_eval_state_key(key: str) -> bool:
    return key.startswith(("val_fid.", "test_fid.", "eval_lpips."))


def _format_incompatible_keys(missing: list[str], unexpected: list[str]) -> str:
    chunks = []
    if missing:
        chunks.append("Missing key(s): " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    if unexpected:
        chunks.append("Unexpected key(s): " + ", ".join(unexpected[:20]) + (" ..." if len(unexpected) > 20 else ""))
    return "; ".join(chunks)


def _update_fid_metric(metric: nn.Module | None, pred: torch.Tensor, target: torch.Tensor) -> None:
    if metric is None:
        return
    metric.update(_to_fid_image(target), real=True)
    metric.update(_to_fid_image(pred), real=False)


def _log_and_reset_fid(pl_module: pl.LightningModule, metric: nn.Module | None, name: str) -> None:
    if metric is None:
        return
    try:
        fid = metric.compute()
        pl_module.log(name, fid, sync_dist=True, add_dataloader_idx=False)
    except Exception as exc:
        print(f"[FID][WARN] failed to compute {name}: {exc}")
    finally:
        metric.reset()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _module_params_m(modules: tuple[nn.Module | None, ...]) -> float:
    seen: set[int] = set()
    total = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            ident = id(parameter)
            if ident in seen:
                continue
            seen.add(ident)
            total += int(parameter.numel())
    return total / 1e6


def _profile_generate_flops(generate_fn, device: torch.device) -> int:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    _sync_if_cuda(device)
    with torch.no_grad():
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            generate_fn()
    _sync_if_cuda(device)
    total = 0
    for event in prof.key_averages():
        total += int(getattr(event, "flops", 0) or 0)
    return total


class PSLFlowLightningModule(pl.LightningModule):
    """PSL-Flow SiT wrapper.

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
        eval_fid = bool(config.get("training", {}).get("eval_fid", False) or model_cfg.get("eval_fid", False))
        self.val_fid = _make_fid_metric(eval_fid, "psl_flow/val")
        self.test_fid = _make_fid_metric(eval_fid, "psl_flow/test")
        self.eval_efficiency = bool(config.get("training", {}).get("eval_efficiency", False) or model_cfg.get("eval_efficiency", False))
        self._efficiency_stats: dict[str, dict[str, float | int | None]] = {}
        self.optimizer_cfg = dict(config.get("training", {}).get("optimizer", {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0}))

    def load_state_dict(self, state_dict, strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [key for key in incompatible.missing_keys if not _is_optional_eval_state_key(key)]
        unexpected = [key for key in incompatible.unexpected_keys if not _is_optional_eval_state_key(key)]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                f"{_format_incompatible_keys(missing, unexpected)}"
            )
        ignored_missing = len(incompatible.missing_keys) - len(missing)
        ignored_unexpected = len(incompatible.unexpected_keys) - len(unexpected)
        if ignored_missing or ignored_unexpected:
            print(
                "[checkpoint][WARN] ignored optional eval metric state mismatch: "
                f"missing={ignored_missing}, unexpected={ignored_unexpected}"
            )
        return incompatible

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
        self.log("train/loss_flow", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log("train/z_phys_std", z_phys.std(unbiased=False), sync_dist=True, add_dataloader_idx=False)
        self.log("train/z_phys_mean", z_phys.mean(), sync_dist=True, add_dataloader_idx=False)
        self.log("train/z_vis_std", z_vis.std(unbiased=False), sync_dist=True, add_dataloader_idx=False)
        if "xt" in loss_dict:
            self.log("train/z_t_std", loss_dict["xt"].std(unbiased=False), sync_dist=True, add_dataloader_idx=False)
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
        model_forward = model.forward
        active_efficiency_split = getattr(self, "_active_efficiency_split", None)
        if active_efficiency_split:
            def counted_forward(*args, **kwargs):
                stats = self._efficiency_stats.setdefault(active_efficiency_split, {"elapsed": 0.0, "samples": 0, "flops": None, "nfe": 0, "batches": 0})
                stats["nfe"] = int(stats.get("nfe") or 0) + 1
                return model.forward(*args, **kwargs)
            model_forward = counted_forward
        with torch.no_grad():
            samples = sample_fn(z0, model_forward, y=dataset_idx, x_RGB=z_vis)[-1]
        return samples / self.thermal_normalizer

    def generate(self, rgb: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        z_phys = self._sample_latent(rgb, dataset_idx, use_ema=True)
        with torch.no_grad():
            decoded = self.psl_vae.decode_latents(z_phys)
        return decoded["y_hat"]

    def _inference_params_m(self) -> float:
        return _module_params_m((self.rgb_vae, self.ema, self.psl_vae))

    def _reset_efficiency(self, split: str) -> None:
        if self.eval_efficiency and not getattr(self.trainer, "sanity_checking", False):
            self._efficiency_stats[split] = {"elapsed": 0.0, "samples": 0, "flops": None, "nfe": 0, "batches": 0}

    def _generate_with_efficiency(self, split: str, rgb: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        if not self.eval_efficiency or getattr(self.trainer, "sanity_checking", False):
            return self.generate(rgb, dataset_idx)
        device = rgb.device
        _sync_if_cuda(device)
        start = time.perf_counter()
        self._active_efficiency_split = split
        try:
            pred = self.generate(rgb, dataset_idx)
        finally:
            self._active_efficiency_split = None
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - start
        stats = self._efficiency_stats.setdefault(split, {"elapsed": 0.0, "samples": 0, "flops": None, "nfe": 0, "batches": 0})
        stats["elapsed"] = float(stats["elapsed"] or 0.0) + elapsed
        stats["samples"] = int(stats["samples"] or 0) + int(rgb.shape[0])
        stats["batches"] = int(stats.get("batches") or 0) + 1
        if stats.get("flops") is None:
            batch_size = max(int(rgb.shape[0]), 1)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            try:
                with torch.random.fork_rng(devices=devices, enabled=True):
                    flops = _profile_generate_flops(lambda: self.generate(rgb, dataset_idx), device)
                stats["flops"] = float(flops) / float(batch_size)
                if flops == 0:
                    print("[efficiency][WARN] torch.profiler returned 0 FLOPs. Some operators may be unsupported.")
            except Exception as exc:
                stats["flops"] = 0.0
                print(f"[efficiency][WARN] failed to profile FLOPs: {exc}")
        return pred

    def _log_efficiency(self, split: str) -> None:
        if not self.eval_efficiency or getattr(self.trainer, "sanity_checking", False):
            return
        stats = self._efficiency_stats.get(split, {})
        samples = int(stats.get("samples") or 0)
        if samples <= 0:
            return
        rt = float(stats.get("elapsed") or 0.0) / float(samples)
        params_m = self._inference_params_m()
        flops_g = float(stats.get("flops") or 0.0) / 1e9
        nfe = int(stats.get("nfe") or 0)
        batches = max(int(stats.get("batches") or 0), 1)
        self.log(f"{split}/RT(s)", torch.tensor(rt, device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/Params(M)", torch.tensor(params_m, device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/NFE", torch.tensor(float(nfe), device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/NFE_per_batch", torch.tensor(float(nfe) / float(batches), device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/NFE_per_sample", torch.tensor(float(nfe) / float(samples), device=self.device), sync_dist=True, add_dataloader_idx=False)
        if flops_g > 0.0:
            self.log(f"{split}/FLOPs(G)", torch.tensor(flops_g, device=self.device), sync_dist=True, add_dataloader_idx=False)
        print(f"[efficiency] {split}/FLOPs(G)={flops_g:.4f}, {split}/Params(M)={params_m:.4f}, {split}/RT(s)={rt:.6f}, samples={samples}, NFE={nfe}, NFE/batch={float(nfe)/float(batches):.4f}, NFE/sample={float(nfe)/float(samples):.4f}")

    def validation_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self._generate_with_efficiency("val", rgb, dataset_idx)
        self.log("val/PSNR", _psnr(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        self.log("val/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        if self.eval_lpips is not None:
            pred_lpips = _repeat_to_three(pred * 2.0 - 1.0)
            target_lpips = _repeat_to_three(target * 2.0 - 1.0)
            self.log("val/LPIPS", self.eval_lpips(pred_lpips, target_lpips).mean(), sync_dist=True, add_dataloader_idx=False)
        _update_fid_metric(self.val_fid, pred, target)
        error = (pred - target).abs()
        return {
            "RGB": _rgb_to_01(rgb),
            "GT": target,
            "pred": pred,
            "error": error,
            "compare": torch.cat([target, pred, error], dim=-1),
        }

    def test_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self._generate_with_efficiency("test", rgb, dataset_idx)
        self.log("test/PSNR", _psnr(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        self.log("test/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        if self.eval_lpips is not None:
            pred_lpips = _repeat_to_three(pred * 2.0 - 1.0)
            target_lpips = _repeat_to_three(target * 2.0 - 1.0)
            self.log("test/LPIPS", self.eval_lpips(pred_lpips, target_lpips).mean(), sync_dist=True, add_dataloader_idx=False)
        _update_fid_metric(self.test_fid, pred, target)
        error = (pred - target).abs()
        return {
            "RGB": _rgb_to_01(rgb),
            "GT": target,
            "pred": pred,
            "error": error,
            "compare": torch.cat([target, pred, error], dim=-1),
        }

    def on_validation_epoch_start(self) -> None:
        self._reset_efficiency("val")

    def on_validation_epoch_end(self) -> None:
        self._log_efficiency("val")
        _log_and_reset_fid(self, self.val_fid, "val/FID")

    def on_test_epoch_start(self) -> None:
        self._reset_efficiency("test")

    def on_test_epoch_end(self) -> None:
        self._log_efficiency("test")
        _log_and_reset_fid(self, self.test_fid, "test/FID")


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
        eval_fid = bool(config.get("training", {}).get("eval_fid", False) or model_cfg.get("eval_fid", False))
        self.val_fid = _make_fid_metric(eval_fid, "klvae_sit/val")
        self.test_fid = _make_fid_metric(eval_fid, "klvae_sit/test")
        self.eval_efficiency = bool(config.get("training", {}).get("eval_efficiency", False) or model_cfg.get("eval_efficiency", False))
        self._efficiency_stats: dict[str, dict[str, float | int | None]] = {}
        self.optimizer_cfg = dict(config.get("training", {}).get("optimizer", {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0}))

    def load_state_dict(self, state_dict, strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [key for key in incompatible.missing_keys if not _is_optional_eval_state_key(key)]
        unexpected = [key for key in incompatible.unexpected_keys if not _is_optional_eval_state_key(key)]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                f"{_format_incompatible_keys(missing, unexpected)}"
            )
        ignored_missing = len(incompatible.missing_keys) - len(missing)
        ignored_unexpected = len(incompatible.unexpected_keys) - len(unexpected)
        if ignored_missing or ignored_unexpected:
            print(
                "[checkpoint][WARN] ignored optional eval metric state mismatch: "
                f"missing={ignored_missing}, unexpected={ignored_unexpected}"
            )
        return incompatible

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
        self.log("train/loss_flow", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log("train/z_thermal_std", z_thermal.std(unbiased=False), sync_dist=True, add_dataloader_idx=False)
        self.log("train/z_vis_std", z_vis.std(unbiased=False), sync_dist=True, add_dataloader_idx=False)
        self._update_ema()
        return loss

    def _sample_latent(self, rgb: torch.Tensor, dataset_idx: torch.Tensor, *, use_ema: bool = True) -> torch.Tensor:
        z_vis = self._encode_rgb(rgb)
        h = rgb.shape[-2] // self.downsample_factor
        w = rgb.shape[-1] // self.downsample_factor
        z0 = torch.randn(rgb.shape[0], self.thermal_latent_channels, h, w, device=rgb.device, dtype=z_vis.dtype)
        sample_fn = self.sampler.sample_ode(**self.sample_ode)
        model = self.ema if use_ema else self.sit
        model_forward = model.forward
        active_efficiency_split = getattr(self, "_active_efficiency_split", None)
        if active_efficiency_split:
            def counted_forward(*args, **kwargs):
                stats = self._efficiency_stats.setdefault(active_efficiency_split, {"elapsed": 0.0, "samples": 0, "flops": None, "nfe": 0, "batches": 0})
                stats["nfe"] = int(stats.get("nfe") or 0) + 1
                return model.forward(*args, **kwargs)
            model_forward = counted_forward
        with torch.no_grad():
            samples = sample_fn(z0, model_forward, y=dataset_idx, x_RGB=z_vis)[-1]
        return samples / self.thermal_normalizer

    def generate(self, rgb: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        z_thermal = self._sample_latent(rgb, dataset_idx, use_ema=True)
        with torch.no_grad():
            decoded = self.thermal_vae.decode(z_thermal).sample
        return _decode_thermal_to_01(decoded)

    def _inference_params_m(self) -> float:
        return _module_params_m((self.rgb_vae, self.ema, self.thermal_vae))

    def _reset_efficiency(self, split: str) -> None:
        if self.eval_efficiency and not getattr(self.trainer, "sanity_checking", False):
            self._efficiency_stats[split] = {"elapsed": 0.0, "samples": 0, "flops": None, "nfe": 0, "batches": 0}

    def _generate_with_efficiency(self, split: str, rgb: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        if not self.eval_efficiency or getattr(self.trainer, "sanity_checking", False):
            return self.generate(rgb, dataset_idx)
        device = rgb.device
        _sync_if_cuda(device)
        start = time.perf_counter()
        self._active_efficiency_split = split
        try:
            pred = self.generate(rgb, dataset_idx)
        finally:
            self._active_efficiency_split = None
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - start
        stats = self._efficiency_stats.setdefault(split, {"elapsed": 0.0, "samples": 0, "flops": None, "nfe": 0, "batches": 0})
        stats["elapsed"] = float(stats["elapsed"] or 0.0) + elapsed
        stats["samples"] = int(stats["samples"] or 0) + int(rgb.shape[0])
        stats["batches"] = int(stats.get("batches") or 0) + 1
        if stats.get("flops") is None:
            batch_size = max(int(rgb.shape[0]), 1)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            try:
                with torch.random.fork_rng(devices=devices, enabled=True):
                    flops = _profile_generate_flops(lambda: self.generate(rgb, dataset_idx), device)
                stats["flops"] = float(flops) / float(batch_size)
                if flops == 0:
                    print("[efficiency][WARN] torch.profiler returned 0 FLOPs. Some operators may be unsupported.")
            except Exception as exc:
                stats["flops"] = 0.0
                print(f"[efficiency][WARN] failed to profile FLOPs: {exc}")
        return pred

    def _log_efficiency(self, split: str) -> None:
        if not self.eval_efficiency or getattr(self.trainer, "sanity_checking", False):
            return
        stats = self._efficiency_stats.get(split, {})
        samples = int(stats.get("samples") or 0)
        if samples <= 0:
            return
        rt = float(stats.get("elapsed") or 0.0) / float(samples)
        params_m = self._inference_params_m()
        flops_g = float(stats.get("flops") or 0.0) / 1e9
        nfe = int(stats.get("nfe") or 0)
        batches = max(int(stats.get("batches") or 0), 1)
        self.log(f"{split}/RT(s)", torch.tensor(rt, device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/Params(M)", torch.tensor(params_m, device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/NFE", torch.tensor(float(nfe), device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/NFE_per_batch", torch.tensor(float(nfe) / float(batches), device=self.device), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{split}/NFE_per_sample", torch.tensor(float(nfe) / float(samples), device=self.device), sync_dist=True, add_dataloader_idx=False)
        if flops_g > 0.0:
            self.log(f"{split}/FLOPs(G)", torch.tensor(flops_g, device=self.device), sync_dist=True, add_dataloader_idx=False)
        print(f"[efficiency] {split}/FLOPs(G)={flops_g:.4f}, {split}/Params(M)={params_m:.4f}, {split}/RT(s)={rt:.6f}, samples={samples}, NFE={nfe}, NFE/batch={float(nfe)/float(batches):.4f}, NFE/sample={float(nfe)/float(samples):.4f}")

    def validation_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self._generate_with_efficiency("val", rgb, dataset_idx)
        self.log("val/PSNR", _psnr(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        self.log("val/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        if self.eval_lpips is not None:
            self.log(
                "val/LPIPS",
                self.eval_lpips(_repeat_to_three(pred * 2.0 - 1.0), _repeat_to_three(target * 2.0 - 1.0)).mean(),
                sync_dist=True,
                add_dataloader_idx=False,
            )
        _update_fid_metric(self.val_fid, pred, target)
        error = (pred - target).abs()
        return {
            "RGB": _rgb_to_01(rgb),
            "GT": target,
            "pred": pred,
            "error": error,
            "compare": torch.cat([target, pred, error], dim=-1),
        }

    def test_step(self, batch, batch_idx):
        rgb, thermal, dataset_idx = batch[:3]
        target = thermal_to_01(thermal)
        pred = self._generate_with_efficiency("test", rgb, dataset_idx)
        self.log("test/PSNR", _psnr(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        self.log("test/SSIM", ssim_per_sample(pred, target).mean(), sync_dist=True, add_dataloader_idx=False)
        if self.eval_lpips is not None:
            self.log(
                "test/LPIPS",
                self.eval_lpips(_repeat_to_three(pred * 2.0 - 1.0), _repeat_to_three(target * 2.0 - 1.0)).mean(),
                sync_dist=True,
                add_dataloader_idx=False,
            )
        _update_fid_metric(self.test_fid, pred, target)
        error = (pred - target).abs()
        return {
            "RGB": _rgb_to_01(rgb),
            "GT": target,
            "pred": pred,
            "error": error,
            "compare": torch.cat([target, pred, error], dim=-1),
        }

    def on_validation_epoch_start(self) -> None:
        self._reset_efficiency("val")

    def on_validation_epoch_end(self) -> None:
        self._log_efficiency("val")
        _log_and_reset_fid(self, self.val_fid, "val/FID")

    def on_test_epoch_start(self) -> None:
        self._reset_efficiency("test")

    def on_test_epoch_end(self) -> None:
        self._log_efficiency("test")
        _log_and_reset_fid(self, self.test_fid, "test/FID")


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


def _parse_interval(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    return float(text) if "." in text else int(text)


def _parse_check_val_every(value: Any) -> int | None:
    if value in (None, ""):
        return 1
    number = int(value)
    return None if number <= 0 else number


def _trainer_from_args(config: dict[str, Any], args: argparse.Namespace, *, with_checkpoints: bool = False) -> tuple[pl.Trainer, list[pl.Callback]]:
    training = dict(config.get("training", {}))
    monitor = str(training.get("checkpoint_monitor", "val/LPIPS"))
    mode = str(training.get("checkpoint_mode", "min"))
    callbacks = build_management_callbacks(training, root_dir=args.default_root_dir, monitor=monitor, mode=mode)
    limit_val_batches = args.limit_val_batches if args.limit_val_batches is not None else training.get("limit_val_batches", None)
    if with_checkpoints:
        if args.checkpoint_every_n_train_steps:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    save_last=False,
                    save_top_k=-1,
                    every_n_train_steps=args.checkpoint_every_n_train_steps,
                    filename="step_{step:06d}",
                    auto_insert_metric_name=False,
                )
            )
        save_top_k = 0 if limit_val_batches == 0 else int(training.get("checkpoint_save_top_k", 1))
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                monitor=monitor if save_top_k != 0 else None,
                mode=mode,
                save_top_k=save_top_k,
                filename="best",
                auto_insert_metric_name=False,
            )
        )
    trainer = pl.Trainer(
        max_epochs=int(training.get("num_epochs", 1000)),
        max_steps=args.max_steps if args.max_steps is not None else int(training.get("max_steps", -1)),
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
        precision="16-mixed" if bool(training.get("mixed_precision", False)) else "32-true",
        callbacks=callbacks,
        logger=False if bool(training.get("disable_logger", False)) else True,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=_parse_check_val_every(args.check_val_every_n_epoch)
        if args.check_val_every_n_epoch is not None
        else _parse_check_val_every(training.get("check_val_every_n_epoch", 1)),
        val_check_interval=_parse_interval(args.val_check_interval)
        if args.val_check_interval is not None
        else _parse_interval(training.get("val_check_interval", None)),
        accumulate_grad_batches=int(training.get("gradient_accumulation", 1)),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
        gradient_clip_val=float(training.get("gradient_clip_val", 0.0) or 0.0),
    )
    return trainer, callbacks


def main() -> None:
    parser = argparse.ArgumentParser("Train/test PSL-Flow")
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
    parser.add_argument("--val-check-interval", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    configure_training_runtime(dict(config.get("training", {})))
    route = validate_route(config.get("route") or config.get("model", {}).get("route") or config.get("model", {}).get("model_arch"))
    data = build_data_module(config)
    module = PSLFlowLightningModule(config) if route == "psl_flow" else KLVaeSiTLightningModule(config)
    trainer, callbacks = _trainer_from_args(config, args, with_checkpoints=args.mode == "fit")
    if args.mode == "fit":
        trainer.fit(module, datamodule=data, ckpt_path=args.resume_from)
        if trainer.is_global_zero:
            final_ckpt = Path(args.default_root_dir or trainer.default_root_dir) / "checkpoints" / "last.ckpt"
            final_ckpt.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(final_ckpt))
            print(f"[{route}] saved final checkpoint: {final_ckpt}")
            copy_best_checkpoint_alias(callbacks, Path(args.default_root_dir or trainer.default_root_dir))
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
