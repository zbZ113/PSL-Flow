from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from psl_flow.data import build_data_module, thermal_to_01
from psl_flow.models.psl_vae.psl_vae import PSLVAE, build_terb_teacher
from psl_flow.utils.checkpoint import load_state_dict_flexible


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_checkpoint_config(path: str | Path) -> dict[str, Any] | None:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        return None
    hparams = payload.get("hyper_parameters") or payload.get("hparams")
    if not isinstance(hparams, dict):
        return None
    if isinstance(hparams.get("config"), dict):
        return hparams["config"]
    if isinstance(hparams.get("model"), dict) and isinstance(hparams.get("training"), dict):
        return hparams
    return None


def _get_teacher_cfg(config: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = dict(config.get("training", {}).get("loss", {}).get("config", {}))
    teacher_cfg = dict(loss_cfg.get("teacher", config.get("teacher", {})))
    return teacher_cfg


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _file_fingerprint(path: str | Path) -> dict[str, int | str]:
    resolved = Path(path)
    stat = resolved.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _make_psl_vae(config: dict[str, Any], psl_vae_ckpt: str, teacher_ckpt: str, device: torch.device) -> PSLVAE:
    model_cfg = dict(config.get("model", {}).get("model_config", {}))
    model = PSLVAE(model_cfg)
    teacher_cfg = _get_teacher_cfg(config)
    if teacher_ckpt:
        teacher_cfg["ckpt"] = teacher_ckpt
    teacher_path = str(teacher_cfg.get("ckpt", ""))
    if not teacher_path:
        raise RuntimeError("A frozen TeR-B checkpoint is required to build PSL-VAE latent targets.")
    teacher, info = build_terb_teacher(
        teacher_cfg,
        ckpt_path=teacher_path,
        strict=bool(teacher_cfg.get("strict_load", False)),
    )
    print(f"[prepare-psl-flow] loaded TeR-B teacher: {info}")
    model.attach_teacher(teacher)
    info = load_state_dict_flexible(
        model,
        psl_vae_ckpt,
        strict=False,
        strip_prefixes=("model.", "module.", "psl_vae."),
    )
    print(f"[prepare-psl-flow] loaded PSL-VAE: {info}")
    model.to(device)
    model.eval()
    return model


def _accumulate_latents(
    model: PSLVAE,
    config: dict[str, Any],
    *,
    device: torch.device,
    max_samples: int,
    sample_latents: bool,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    data = build_data_module(config)
    data.setup("fit")
    loader = data.train_dataloader()

    total_values = 0
    total_samples = 0
    sum_x = 0.0
    sum_x2 = 0.0
    channel_sum = None
    channel_sum2 = None
    channel_count = 0

    with torch.inference_mode():
        for batch in loader:
            thermal = batch[1].to(device, non_blocking=True)
            thermal_01 = thermal_to_01(thermal)
            encoded = model.encode_from_ir(thermal_01, sample=sample_latents)
            z = encoded["z_phys"].detach().float()
            flat = z.reshape(-1).double()
            total_values += int(flat.numel())
            total_samples += int(z.shape[0])
            sum_x += float(flat.sum().item())
            sum_x2 += float(flat.square().sum().item())

            c_sum = z.double().sum(dim=(0, 2, 3)).cpu()
            c_sum2 = z.double().square().sum(dim=(0, 2, 3)).cpu()
            c_count = int(z.shape[0] * z.shape[2] * z.shape[3])
            if channel_sum is None:
                channel_sum = c_sum
                channel_sum2 = c_sum2
            else:
                channel_sum += c_sum
                channel_sum2 += c_sum2
            channel_count += c_count

            if max_samples > 0 and total_samples >= max_samples:
                break

    if total_values <= 1:
        raise RuntimeError("Not enough latent values were collected to estimate a normalizer.")
    mean = sum_x / total_values
    variance = max((sum_x2 - (sum_x * sum_x) / total_values) / max(total_values - 1, 1), 1e-12)
    std = math.sqrt(variance)
    normalizer = 1.0 / max(std, 1e-12)

    channel_mean = []
    channel_std = []
    if channel_sum is not None and channel_sum2 is not None and channel_count > 1:
        channel_mean_tensor = channel_sum / channel_count
        channel_var_tensor = (channel_sum2 - channel_sum.square() / channel_count) / max(channel_count - 1, 1)
        channel_var_tensor = channel_var_tensor.clamp_min(1e-12)
        channel_mean = [float(v) for v in channel_mean_tensor.tolist()]
        channel_std = [float(v) for v in channel_var_tensor.sqrt().tolist()]

    return {
        "num_samples": total_samples,
        "num_values": total_values,
        "latent_mean": float(mean),
        "latent_std": float(std),
        "latent_normalizer": float(normalizer),
        "thermal_normalizer": float(normalizer),
        "latent_channel_mean": channel_mean,
        "latent_channel_std": channel_std,
        "latent_sample_mode": "sample" if sample_latents else "mode",
        "seed": int(seed),
    }


def patch_flow_config(
    flow_cfg: dict[str, Any],
    psl_vae_cfg: dict[str, Any],
    stats: dict[str, Any],
    *,
    psl_vae_ckpt: str,
    teacher_ckpt: str,
    rgb_vae_path: str,
    rgb_vae_repo: str,
    rgb_vae_local_files_only: bool,
    stats_json: str,
) -> dict[str, Any]:
    flow_model_cfg = flow_cfg.setdefault("model", {}).setdefault("model_config", {})
    psl_model_cfg = dict(psl_vae_cfg.get("model", {}).get("model_config", {}))
    if not psl_model_cfg:
        raise RuntimeError("Unable to find PSL-VAE model.model_config for PSL-Flow config patching.")

    teacher_cfg = _get_teacher_cfg(psl_vae_cfg)
    if teacher_ckpt:
        teacher_cfg["ckpt"] = teacher_ckpt

    flow_model_cfg["psl_vae_config"] = psl_model_cfg
    flow_model_cfg["psl_vae_ckpt"] = psl_vae_ckpt
    flow_model_cfg["thermal_normalizer"] = float(stats["thermal_normalizer"])
    flow_model_cfg["thermal_latent_mean"] = float(stats["latent_mean"])
    flow_model_cfg["thermal_latent_std"] = float(stats["latent_std"])
    flow_model_cfg["thermal_latent_stats"] = stats
    flow_model_cfg["normalizer_stats_json"] = stats_json
    flow_model_cfg["teacher"] = teacher_cfg

    if rgb_vae_path:
        flow_model_cfg["rgb_vae_path"] = rgb_vae_path
    if rgb_vae_repo:
        flow_model_cfg["rgb_vae_repo"] = rgb_vae_repo
    flow_model_cfg["rgb_vae_local_files_only"] = bool(rgb_vae_local_files_only)
    if "latent_channels" in psl_model_cfg:
        flow_model_cfg["psl_vae_latent_channels"] = int(psl_model_cfg["latent_channels"])
    rgb_cfg = dict(flow_model_cfg.get("rgb_vae_config", {}))
    if "latent_channels" in rgb_cfg:
        flow_model_cfg["rgb_vae_latent_channels"] = int(rgb_cfg["latent_channels"])
        injection_args = flow_model_cfg.setdefault("sit_config", {}).setdefault("injection_args", {})
        injection_args["rgb_in_chans"] = int(rgb_cfg["latent_channels"])
    return flow_cfg


def main() -> None:
    parser = argparse.ArgumentParser("Estimate PSL-VAE latent normalizer and patch PSL-Flow config")
    parser.add_argument("--flow-config", required=True)
    parser.add_argument("--output-flow-config", default=None)
    parser.add_argument("--psl-vae-config", required=True)
    parser.add_argument("--psl-vae-ckpt", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--rgb-vae-path", default="")
    parser.add_argument("--rgb-vae-repo", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--rgb-vae-local-files-only", nargs="?", const="true", default="false")
    parser.add_argument("--rgb-vae-ckpt", default="", help=argparse.SUPPRESS)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--latent-sample-mode", choices=["sample", "mode"], default="sample")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    flow_cfg_path = Path(args.flow_config)
    out_cfg_path = Path(args.output_flow_config or args.flow_config)
    stats_path = Path(args.stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    flow_cfg = load_yaml(flow_cfg_path)
    file_psl_cfg = load_yaml(args.psl_vae_config)
    ckpt_psl_cfg = _load_checkpoint_config(args.psl_vae_ckpt)
    psl_cfg = ckpt_psl_cfg or file_psl_cfg
    cfg_source = "checkpoint_hyper_parameters" if ckpt_psl_cfg else "yaml"

    device = _resolve_device(args.device)
    model = _make_psl_vae(psl_cfg, args.psl_vae_ckpt, args.teacher_ckpt, device)
    stats = _accumulate_latents(
        model,
        file_psl_cfg,
        device=device,
        max_samples=int(args.max_samples),
        sample_latents=args.latent_sample_mode == "sample",
        seed=int(args.seed),
    )
    stats.update(
        {
            "psl_vae_ckpt": str(args.psl_vae_ckpt),
            "teacher_ckpt": str(args.teacher_ckpt),
            "psl_vae_ckpt_fingerprint": _file_fingerprint(args.psl_vae_ckpt),
            "teacher_ckpt_fingerprint": _file_fingerprint(args.teacher_ckpt),
            "psl_vae_config_source": cfg_source,
            "device": str(device),
            "requested_max_samples": int(args.max_samples),
            "normalizer_latent_sample_mode": str(args.latent_sample_mode),
            "normalizer_seed": int(args.seed),
            "train_num_samples_per_epoch": int(file_psl_cfg.get("training", {}).get("num_samples_per_epoch", 0) or 0),
        }
    )
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    patched = patch_flow_config(
        flow_cfg,
        psl_cfg,
        stats,
        psl_vae_ckpt=str(args.psl_vae_ckpt),
        teacher_ckpt=str(args.teacher_ckpt),
        rgb_vae_path=str(args.rgb_vae_path or args.rgb_vae_ckpt),
        rgb_vae_repo=str(args.rgb_vae_repo),
        rgb_vae_local_files_only=_str_to_bool(args.rgb_vae_local_files_only),
        stats_json=str(stats_path),
    )
    save_yaml(patched, out_cfg_path)
    print(
        "[prepare-psl-flow] thermal_normalizer="
        f"{stats['thermal_normalizer']:.10f}, latent_std={stats['latent_std']:.10f}, "
        f"num_samples={stats['num_samples']}, config_source={cfg_source}"
    )
    print(f"[prepare-psl-flow] wrote stats: {stats_path}")
    print(f"[prepare-psl-flow] wrote flow config: {out_cfg_path}")


if __name__ == "__main__":
    main()
