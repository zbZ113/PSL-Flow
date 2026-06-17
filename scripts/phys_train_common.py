from __future__ import annotations

import csv
import gc
import inspect
import json
import os
import random
import signal
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from diffusers.models import AutoencoderKL
except Exception:
    AutoencoderKL = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def bool_flag(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_metrics_csv(path: Path, payload: Dict, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: payload.get(name, "") for name in fieldnames})


def resolve_resume_path(explicit_resume: str, candidates: Sequence[Path | str]) -> str:
    resume_path = str(explicit_resume or "").strip()
    if resume_path and resume_path.lower() != "auto":
        return resume_path
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return str(path)
    return ""


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_datamodule(cfg, batch_size: int = 0, num_workers: int = 0, datasets_folder: str = ""):
    from dataloaders.GenericDataloader import GenericDataModule

    resolved_batch_size = int(batch_size) if int(batch_size) > 0 else int(cfg.training.train_batch_size)
    resolved_num_workers = int(num_workers) if int(num_workers) > 0 else int(cfg.training.num_workers)
    resolved_datasets_folder = datasets_folder if datasets_folder else cfg.datasets.datasets_folder
    dm = GenericDataModule(
        datasets_folder=resolved_datasets_folder,
        train_batch_size=resolved_batch_size,
        test_batch_size=int(cfg.training.test_batch_size),
        train_image_size=cfg.training.train_image_size,
        num_workers=resolved_num_workers,
        dataset_names=cfg.datasets,
        train_cfg_training=cfg.training,
        mixed_precision=False,
    )
    dm.setup("fit")
    return dm, resolved_batch_size, resolved_num_workers, resolved_datasets_folder


def build_eval_loaders(dm: GenericDataModule):
    loaders = []
    device_count = max(1, torch.cuda.device_count())
    for index, dataset_name in enumerate(dm.val_dataset_names):
        dataset = dm.val_datasets[index]
        if dataset_name.startswith("Boson") or dataset_name.startswith("DJI"):
            loaders.append(DataLoader(dataset=dataset, **dm.eval_loader_config))
            continue
        shard_num = int(getattr(dataset, "my_shard_num", dm.num_workers))
        if int(dm.num_workers) > 0:
            worker_count = max(min(shard_num, int(dm.num_workers)) // device_count, 1)
        else:
            worker_count = 0
        kwargs = dict(dm.eval_loader_config_generic)
        kwargs["num_workers"] = worker_count
        loaders.append(DataLoader(dataset=dataset, **kwargs))
    return loaders


def build_named_eval_loaders(dm: GenericDataModule, split: str = "val"):
    split = str(split).lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split={split}. Expected one of: val, test.")

    if split == "val":
        dataset_names = list(dm.val_dataset_names)
        datasets = list(dm.val_datasets)
        diverse_allowed = False
    else:
        dataset_names = list(dm.test_dataset_names)
        datasets = list(dm.test_datasets)
        diverse_allowed = True

    loaders = []
    device_count = max(1, torch.cuda.device_count())
    for dataset_name, dataset in zip(dataset_names, datasets):
        if dataset_name.startswith("Boson") or dataset_name.startswith("DJI"):
            loaders.append(DataLoader(dataset=dataset, **dm.eval_loader_config))
            continue

        shard_num = int(getattr(dataset, "my_shard_num", dm.num_workers))
        if int(dm.num_workers) > 0:
            worker_count = max(min(shard_num, int(dm.num_workers)) // device_count, 1)
        else:
            worker_count = 0

        if diverse_allowed and hasattr(dataset, "diverse_size") and bool(dataset.diverse_size):
            kwargs = dict(dm.eval_loader_config_generic_diverse)
        else:
            kwargs = dict(dm.eval_loader_config_generic)
        kwargs["num_workers"] = worker_count
        loaders.append(DataLoader(dataset=dataset, **kwargs))
    return loaders


def extract_thermal(batch) -> torch.Tensor:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise RuntimeError(f"Unexpected batch structure: {type(batch)}")
    return batch[1]


def thermal_to_01(thermal: torch.Tensor) -> torch.Tensor:
    return torch.clamp((thermal + 1.0) * 0.5, 0.0, 1.0)


def effective_max_steps(max_steps_per_epoch: int, num_samples_per_epoch: int, batch_size: int) -> int:
    if int(max_steps_per_epoch) > 0:
        return int(max_steps_per_epoch)
    num_samples_per_epoch = int(num_samples_per_epoch or 0)
    if num_samples_per_epoch > 0:
        return (num_samples_per_epoch + max(1, int(batch_size)) - 1) // max(1, int(batch_size))
    return 0


def build_autoencoder_kl(cfg: Dict, tag: str = "AutoencoderKL") -> AutoencoderKL:
    if AutoencoderKL is None:
        raise ImportError(
            f"{tag} requires diffusers.models.AutoencoderKL, but diffusers is unavailable in the current environment."
        )
    cfg = dict(cfg)
    try:
        sig = inspect.signature(AutoencoderKL.__init__)
        valid_keys = {k for k in sig.parameters.keys() if k not in {"self", "args", "kwargs"}}
        filtered_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        dropped = [k for k in cfg.keys() if k not in filtered_cfg]
    except Exception:
        filtered_cfg = cfg
        dropped = []

    if dropped:
        print(f"[WARN] {tag}: ignore unsupported AutoencoderKL args: {dropped}")
    return AutoencoderKL(**filtered_cfg)


def encode_thermal_latent(
    vae: torch.nn.Module,
    thermal: torch.Tensor,
    vae_model: str,
    thermal_normalizer: Optional[float],
) -> torch.Tensor:
    if vae_model == "klvae":
        in_channels = int(getattr(getattr(vae, "config", None), "in_channels", thermal.shape[1]))
        thermal_in = thermal
        if in_channels == 3 and thermal.shape[1] == 1:
            thermal_in = thermal.repeat(1, 3, 1, 1)
        z = vae.encode(thermal_in).latent_dist.sample()
    elif vae_model == "dcae":
        z = vae.encode(thermal).latent
    else:
        raise NotImplementedError(f"Unsupported vae_model={vae_model}")

    if thermal_normalizer is not None:
        z = z * float(thermal_normalizer)
    return z


def reset_device_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def device_memory_stats(device: torch.device) -> Dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_name": "cpu",
            "gpu_total_mem_gb": 0.0,
            "gpu_mem_alloc_gb": 0.0,
            "gpu_mem_reserved_gb": 0.0,
            "gpu_mem_peak_gb": 0.0,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": props.name,
        "gpu_total_mem_gb": round(float(props.total_memory) / (1024 ** 3), 3),
        "gpu_mem_alloc_gb": round(float(torch.cuda.memory_allocated(device)) / (1024 ** 3), 3),
        "gpu_mem_reserved_gb": round(float(torch.cuda.memory_reserved(device)) / (1024 ** 3), 3),
        "gpu_mem_peak_gb": round(float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3), 3),
    }


def device_summary(device: torch.device) -> str:
    stats = device_memory_stats(device)
    if device.type != "cuda":
        return "cpu"
    return f"cuda:{device.index if device.index is not None else torch.cuda.current_device()} {stats['gpu_name']} total={stats['gpu_total_mem_gb']:.2f}GB"

class InterruptHandler:
    def __init__(self, name: str = "Train"):
        self.name = str(name)
        self._count = 0
        self._installed = {}

    def _handle(self, signum, frame):
        self._count += 1
        if self._count >= 2:
            print(f"[{self.name}] Force exit on second interrupt.")
            os._exit(130)
        raise KeyboardInterrupt

    def install(self) -> None:
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            self._installed[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, handler in self._installed.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        self._installed.clear()


def close_progress(progress) -> None:
    if progress is None:
        return
    try:
        progress.close()
    except Exception:
        pass


def shutdown_dataloader(loader) -> None:
    if loader is None:
        return
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    except Exception:
        pass


def shutdown_dataloaders(loaders) -> None:
    if loaders is None:
        return
    if isinstance(loaders, (list, tuple)):
        for loader in loaders:
            shutdown_dataloader(loader)
        return
    shutdown_dataloader(loaders)


def release_runtime_resources(device: torch.device | None = None) -> None:
    gc.collect()
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _remove_path(path: Path) -> None:
    path = Path(path)
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def prepare_stage_output_dirs(
    run_dir: Path,
    ckpt_dir: Path,
    stage_name: str,
    fresh_start: bool = False,
) -> Tuple[Path, Path]:
    run_dir = Path(run_dir)
    ckpt_dir = Path(ckpt_dir)
    if fresh_start:
        print(f"[{stage_name}] Fresh start requested. Clearing previous artifacts in {run_dir} and {ckpt_dir}.")
        _remove_path(run_dir)
        _remove_path(ckpt_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir = run_dir / "states"
    visuals_dir = run_dir / "visuals"
    state_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    return state_dir, visuals_dir


def _tensor_to_map(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def _normalize_unit(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _normalize_auto(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_range(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = max(float(hi) - float(lo), 1e-6)
    return np.clip((arr - float(lo)) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _pair_normalize(a: torch.Tensor, b: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    arr_a = _tensor_to_map(a)
    arr_b = _tensor_to_map(b)
    lo = min(float(arr_a.min()), float(arr_b.min()))
    hi = max(float(arr_a.max()), float(arr_b.max()))
    return _normalize_range(arr_a, lo, hi), _normalize_range(arr_b, lo, hi)


def _upscale(arr: np.ndarray, factor: int = 1) -> np.ndarray:
    if int(factor) <= 1:
        return arr
    return np.repeat(np.repeat(arr, int(factor), axis=0), int(factor), axis=1)


def _save_panel(rows: Sequence[Sequence[Tuple[str, np.ndarray]]], path: Path, suptitle: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    fig, axes = plt.subplots(row_count, col_count, figsize=(2.2 * col_count, 2.2 * row_count), squeeze=False)
    for row_idx, row in enumerate(rows):
        for col_idx in range(col_count):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx >= len(row):
                continue
            title, image = row[col_idx]
            ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_teacher_visualization(teacher_out: Dict[str, torch.Tensor], path: Path, max_samples: int = 4) -> None:
    total = int(teacher_out["S_01"].shape[0])
    count = max(1, min(int(max_samples), total))
    rows = []
    for idx in range(count):
        rows.append([
            ("S_01", _normalize_unit(_tensor_to_map(teacher_out["S_01"][idx]))),
            ("e", _normalize_unit(_tensor_to_map(teacher_out["e"][idx]))),
            ("T_rad", _normalize_auto(_tensor_to_map(teacher_out["T_rad"][idx]))),
            ("R_basis", _normalize_unit(_tensor_to_map(teacher_out["R_env_basis"][idx]))),
            ("R_local", _normalize_unit(_tensor_to_map(teacher_out["R_env_local"][idx]))),
            ("R_env", _normalize_unit(_tensor_to_map(teacher_out["R_env"][idx]))),
            ("A", _normalize_auto(_tensor_to_map(teacher_out["A"][idx]))),
            ("B_edge", _normalize_unit(_tensor_to_map(teacher_out["B_edge"][idx]))),
            ("S_phys", _normalize_unit(_tensor_to_map(teacher_out["S_phys"][idx]))),
        ])
    lambda_env = float(teacher_out["lambda_env"].detach().mean().cpu().item()) if "lambda_env" in teacher_out else float("nan")
    _save_panel(rows, Path(path), suptitle=f"TeR-B Net samples | lambda_env={lambda_env:.4f}")


def save_proxy_visualization(
    lowres: Dict[str, torch.Tensor],
    pred32: Dict[str, torch.Tensor],
    path: Path,
    max_samples: int = 4,
    stage_name: str = "Proxy",
) -> None:
    total = int(lowres["S_01"].shape[0])
    count = max(1, min(int(max_samples), total))
    rows = []
    for idx in range(count):
        t_teacher, t_pred = _pair_normalize(lowres["T_rad"][idx], pred32["T_rad"][idx])
        r_teacher, r_pred = _pair_normalize(lowres["R_env"][idx], pred32["R_env"][idx])
        a_teacher, a_pred = _pair_normalize(lowres["A"][idx], pred32["A"][idx])
        rows.append([
            ("S_gt", _upscale(_normalize_unit(_tensor_to_map(lowres["S_01"][idx])), 8)),
            ("S_teacher", _upscale(_normalize_unit(_tensor_to_map(lowres["S_phys"][idx])), 8)),
            ("S_pred", _upscale(_normalize_unit(_tensor_to_map(pred32["S_phys"][idx])), 8)),
            ("e_teacher", _upscale(_normalize_unit(_tensor_to_map(lowres["e"][idx])), 8)),
            ("e_pred", _upscale(_normalize_unit(_tensor_to_map(pred32["e"][idx])), 8)),
            ("T_teacher", _upscale(t_teacher, 8)),
            ("T_pred", _upscale(t_pred, 8)),
            ("R_teacher", _upscale(r_teacher, 8)),
            ("R_pred", _upscale(r_pred, 8)),
            ("A_teacher", _upscale(a_teacher, 8)),
            ("A_pred", _upscale(a_pred, 8)),
            ("B_teacher", _upscale(_normalize_unit(_tensor_to_map(lowres["B_edge"][idx])), 8)),
            ("B_pred", _upscale(_normalize_unit(_tensor_to_map(pred32["B_edge"][idx])), 8)),
        ])
    _save_panel(rows, Path(path), suptitle=f"{stage_name} teacher-target vs prediction")


def save_phys_chain_visualization(
    teacher_out: Dict[str, torch.Tensor],
    lowres: Dict[str, torch.Tensor],
    proxy_out: Dict[str, torch.Tensor],
    sur_out: Dict[str, torch.Tensor],
    path: Path,
    max_samples: int = 4,
) -> None:
    total = int(teacher_out["S_01"].shape[0])
    count = max(1, min(int(max_samples), total))
    rows = []
    for idx in range(count):
        t_teacher_proxy, t_proxy = _pair_normalize(lowres["T_rad"][idx], proxy_out["T_rad"][idx])
        r_teacher_proxy, r_proxy = _pair_normalize(lowres["R_env"][idx], proxy_out["R_env"][idx])
        a_teacher_proxy, a_proxy = _pair_normalize(lowres["A"][idx], proxy_out["A"][idx])

        t_teacher_sur, t_sur = _pair_normalize(lowres["T_rad"][idx], sur_out["T_rad"][idx])
        r_teacher_sur, r_sur = _pair_normalize(lowres["R_env"][idx], sur_out["R_env"][idx])
        a_teacher_sur, a_sur = _pair_normalize(lowres["A"][idx], sur_out["A"][idx])

        proxy_delta = _normalize_auto(np.abs(_tensor_to_map(proxy_out["S_phys"][idx]) - _tensor_to_map(lowres["S_phys"][idx])))
        sur_delta = _normalize_auto(np.abs(_tensor_to_map(sur_out["S_phys"][idx]) - _tensor_to_map(lowres["S_phys"][idx])))

        rows.append([
            ("S_01", _normalize_unit(_tensor_to_map(teacher_out["S_01"][idx]))),
            ("S_teacher", _normalize_unit(_tensor_to_map(teacher_out["S_phys"][idx]))),
            ("e_teacher", _normalize_unit(_tensor_to_map(teacher_out["e"][idx]))),
            ("T_teacher", _normalize_auto(_tensor_to_map(teacher_out["T_rad"][idx]))),
            ("R_teacher", _normalize_unit(_tensor_to_map(teacher_out["R_env"][idx]))),
            ("A_teacher", _normalize_auto(_tensor_to_map(teacher_out["A"][idx]))),
            ("B_teacher", _normalize_unit(_tensor_to_map(teacher_out["B_edge"][idx]))),
        ])
        rows.append([
            ("S_proxy", _upscale(_normalize_unit(_tensor_to_map(proxy_out["S_phys"][idx])), 8)),
            ("S_teacher32", _upscale(_normalize_unit(_tensor_to_map(lowres["S_phys"][idx])), 8)),
            ("e_proxy", _upscale(_normalize_unit(_tensor_to_map(proxy_out["e"][idx])), 8)),
            ("T_proxy", _upscale(t_proxy, 8)),
            ("R_proxy", _upscale(r_proxy, 8)),
            ("A_proxy", _upscale(a_proxy, 8)),
            ("|S_p-S_t|", _upscale(proxy_delta, 8)),
        ])
        rows.append([
            ("S_sur", _upscale(_normalize_unit(_tensor_to_map(sur_out["S_phys"][idx])), 8)),
            ("S_teacher32", _upscale(_normalize_unit(_tensor_to_map(lowres["S_phys"][idx])), 8)),
            ("e_sur", _upscale(_normalize_unit(_tensor_to_map(sur_out["e"][idx])), 8)),
            ("T_sur", _upscale(t_sur, 8)),
            ("R_sur", _upscale(r_sur, 8)),
            ("A_sur", _upscale(a_sur, 8)),
            ("|S_s-S_t|", _upscale(sur_delta, 8)),
        ])

    lambda_env = float(teacher_out["lambda_env"].detach().mean().cpu().item()) if "lambda_env" in teacher_out else float("nan")
    _save_panel(rows, Path(path), suptitle=f"Phys chain consistency | lambda_env={lambda_env:.4f}")
