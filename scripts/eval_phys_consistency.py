from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.physics.latent_surrogate import load_module_checkpoint
from models.physics.phys_losses import l1_per_sample, normalize_01, sobel_mag, ssim_per_sample
from models.physics.terb_net import TeR_B


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class PairItem:
    key: str
    gt_path: Path
    pred_path: Path
    input_path: Optional[Path] = None


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate physical consistency on GT vs generated thermal image pairs by "
            "running a dataset-specific TeR-B Net on both images."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config for multi-dataset physics analysis.")
    return parser


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(path_str: str | None, base_dir: Path) -> Optional[Path]:
    if path_str is None:
        return None
    text = str(path_str).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _psnr_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _edge_scores(
    pred: torch.Tensor,
    gt: torch.Tensor,
    pred_thresh: float = 0.5,
    gt_thresh: float = 0.3,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_bin = (pred >= float(pred_thresh)).float()
    gt_bin = (gt >= float(gt_thresh)).float()
    tp = (pred_bin * gt_bin).flatten(1).sum(dim=1)
    fp = (pred_bin * (1.0 - gt_bin)).flatten(1).sum(dim=1)
    fn = ((1.0 - pred_bin) * gt_bin).flatten(1).sum(dim=1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def _rank_correlation_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    downsample: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    if int(downsample) > 0 and (
        pred.shape[-2] != int(downsample) or pred.shape[-1] != int(downsample)
    ):
        pred = F.adaptive_avg_pool2d(pred, (int(downsample), int(downsample)))
        target = F.adaptive_avg_pool2d(target, (int(downsample), int(downsample)))

    p = pred.flatten(1)
    t = target.flatten(1)
    p_rank = p.argsort(dim=1).argsort(dim=1).float()
    t_rank = t.argsort(dim=1).argsort(dim=1).float()

    p_rank = p_rank - p_rank.mean(dim=1, keepdim=True)
    t_rank = t_rank - t_rank.mean(dim=1, keepdim=True)
    numerator = (p_rank * t_rank).sum(dim=1)
    denominator = torch.sqrt(p_rank.square().sum(dim=1) * t_rank.square().sum(dim=1) + eps)
    return numerator / (denominator + eps)


def _hotspot_masks(temp: torch.Tensor, hot_percent: float = 1.0) -> torch.Tensor:
    hot_percent = float(hot_percent)
    if not (0.0 < hot_percent < 100.0):
        raise ValueError(f"hot_percent must be in (0, 100), got {hot_percent}.")
    flat = temp.flatten(1)
    q = 1.0 - hot_percent / 100.0
    threshold = torch.quantile(flat, q, dim=1).view(-1, 1, 1, 1)
    return (temp >= threshold).float()


def _hotspot_metrics(
    pred_temp: torch.Tensor,
    gt_temp: torch.Tensor,
    hot_percent: float,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    pred_hot = _hotspot_masks(pred_temp, hot_percent=hot_percent)
    gt_hot = _hotspot_masks(gt_temp, hot_percent=hot_percent)
    inter = (pred_hot * gt_hot).flatten(1).sum(dim=1)
    pred_sum = pred_hot.flatten(1).sum(dim=1)
    gt_sum = gt_hot.flatten(1).sum(dim=1)
    union = pred_sum + gt_sum - inter
    pred_only = (pred_hot * (1.0 - gt_hot)).flatten(1).sum(dim=1)
    return {
        "hotspot_iou": inter / (union + eps),
        "hotspot_precision": inter / (pred_sum + eps),
        "hotspot_recall": inter / (gt_sum + eps),
        "pseudo_hot_ratio": pred_only / (pred_sum + eps),
        "pred_hot_mask": pred_hot,
        "gt_hot_mask": gt_hot,
        "pred_only_hot_mask": pred_hot * (1.0 - gt_hot),
    }


def _to_numpy01(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().squeeze().numpy()
    return np.clip(arr, 0.0, 1.0)


def _normalize_auto(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    return np.clip((arr - lo) / max(hi - lo, eps), 0.0, 1.0)


def _pair_normalize(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = gt_tensor.detach().float().cpu().squeeze().numpy()
    pred = pred_tensor.detach().float().cpu().squeeze().numpy()
    lo = min(float(gt.min()), float(pred.min()))
    hi = max(float(gt.max()), float(pred.max()))
    scale = max(hi - lo, 1e-6)
    gt_vis = np.clip((gt - lo) / scale, 0.0, 1.0)
    pred_vis = np.clip((pred - lo) / scale, 0.0, 1.0)
    delta_vis = _normalize_auto(np.abs(gt - pred))
    return gt_vis, pred_vis, delta_vis


def _save_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    numeric_keys: List[str] = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_keys.append(key)
    summary: Dict[str, Dict[str, float]] = {}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _load_gray_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (256, 256):
            image = image.resize((256, 256), resample=Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (256, 256):
            image = image.resize((256, 256), resample=Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _index_dir_by_key(
    directory: Path,
    match_by: str,
) -> Dict[str, Path]:
    if match_by not in {"stem", "name"}:
        raise ValueError(f"match_by must be `stem` or `name`, got {match_by}.")

    index: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        key = path.stem if match_by == "stem" else path.name
        if key in index:
            raise RuntimeError(
                f"Duplicate key `{key}` found in {directory} with match_by={match_by}. "
                "Please rename files or switch match_by."
            )
        index[key] = path.resolve()
    return index


def _find_teacher_ckpt(repo_root: Path, dataset_name: str) -> Optional[Path]:
    candidates = [
        repo_root / "logs" / "physics" / dataset_name / "teacher" / "states" / "best.pth",
        repo_root / "logs" / "physics" / dataset_name / "teacher" / "states" / "last.pth",
        repo_root / "checkpoints" / "physics" / dataset_name / "teacher" / "teacher_best.pth",
        repo_root / "checkpoints" / "tevnet_thernet" / dataset_name / "best.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _dir_has_pairs(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_gt = any(item.is_file() and item.name.startswith("gt_") for item in path.iterdir())
    has_pred = any(item.is_file() and item.name.startswith("pred_") for item in path.iterdir())
    return has_gt and has_pred


def _find_pair_dir(repo_root: Path, dataset_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    for base_name in ("logs", "outputs"):
        base_dir = repo_root / base_name
        if not base_dir.is_dir():
            continue
        for path in base_dir.rglob(dataset_name):
            if _dir_has_pairs(path):
                candidates.append(path.resolve())
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _discover_pairs(
    pair_dir: Path,
    gt_prefix: str = "gt_",
    pred_prefix: str = "pred_",
    input_prefix: str = "input_",
) -> List[PairItem]:
    items: Dict[str, Dict[str, Path]] = {}
    for path in sorted(pair_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = path.stem
        role: Optional[str] = None
        key: Optional[str] = None
        if stem.startswith(gt_prefix):
            role = "gt"
            key = stem[len(gt_prefix) :]
        elif stem.startswith(pred_prefix):
            role = "pred"
            key = stem[len(pred_prefix) :]
        elif stem.startswith(input_prefix):
            role = "input"
            key = stem[len(input_prefix) :]
        if role is None or key is None:
            continue
        items.setdefault(key, {})[role] = path.resolve()

    pairs: List[PairItem] = []
    for key in sorted(items.keys()):
        entry = items[key]
        if "gt" not in entry or "pred" not in entry:
            continue
        pairs.append(
            PairItem(
                key=key,
                gt_path=entry["gt"],
                pred_path=entry["pred"],
                input_path=entry.get("input"),
            )
        )
    if not pairs:
        raise RuntimeError(
            f"No valid gt/pred pairs were found under {pair_dir}. "
            "Expected files named like gt_xxx.png and pred_xxx.png."
    )
    return pairs


def _discover_pairs_from_split_dirs(
    gt_dir: Path,
    pred_dir: Path,
    input_dir: Optional[Path] = None,
    match_by: str = "stem",
) -> List[PairItem]:
    gt_index = _index_dir_by_key(gt_dir, match_by=match_by)
    pred_index = _index_dir_by_key(pred_dir, match_by=match_by)
    input_index = _index_dir_by_key(input_dir, match_by=match_by) if input_dir is not None and input_dir.is_dir() else {}

    common_keys = sorted(set(gt_index.keys()) & set(pred_index.keys()))
    if not common_keys:
        raise RuntimeError(
            f"No paired files were found between gt_dir={gt_dir} and pred_dir={pred_dir} "
            f"with match_by={match_by}."
        )

    pairs: List[PairItem] = []
    for key in common_keys:
        pairs.append(
            PairItem(
                key=key,
                gt_path=gt_index[key],
                pred_path=pred_index[key],
                input_path=input_index.get(key),
            )
        )
    return pairs


def _load_teacher(teacher_cfg: Dict, ckpt_path: Path, device: torch.device) -> TeR_B:
    teacher = TeR_B(
        smp_model=str(teacher_cfg.get("smp_model", "Unet")),
        smp_encoder=str(teacher_cfg.get("smp_encoder", "resnet18")),
        smp_encoder_weights=teacher_cfg.get("smp_encoder_weights", None),
        vnums=int(teacher_cfg.get("vnums", 4)),
        erme_kernel=int(teacher_cfg.get("erme_kernel", 5)),
        lambda_env_init=float(teacher_cfg.get("lambda_env_init", 0.1)),
        a_low_range=tuple(teacher_cfg.get("a_low_range", [0.8, 1.2])),
    ).to(device)
    load_module_checkpoint(teacher, str(ckpt_path), strict=False)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def _batch_teacher_forward(
    teacher: TeR_B,
    pair_items: Sequence[PairItem],
    start: int,
    end: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    gt_batch = torch.stack([_load_gray_image(item.gt_path) for item in pair_items[start:end]], dim=0).to(device=device, dtype=torch.float32)
    pred_batch = torch.stack([_load_gray_image(item.pred_path) for item in pair_items[start:end]], dim=0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        gt_teacher = teacher(gt_batch)
        pred_teacher = teacher(pred_batch)
    return gt_batch, pred_batch, gt_teacher, pred_teacher


def _save_single_map(path: Path, image: np.ndarray) -> None:
    _save_single_map_with_cmap(path=path, image=image, cmap_name="inferno")


def _save_single_map_with_cmap(path: Path, image: np.ndarray, cmap_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0), cmap=cmap_name, vmin=0.0, vmax=1.0)


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _normalize_u8(arr: np.ndarray, upper: int) -> np.ndarray:
    return cv2.normalize(
        arr.astype(np.float32),
        None,
        alpha=0,
        beta=int(upper),
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def _make_tev_hsv_rgb(
    e_tensor: torch.Tensor,
    t_tensor: torch.Tensor,
    r_tensor: torch.Tensor,
) -> np.ndarray:
    e = e_tensor.detach().float().cpu().squeeze().numpy()
    t = t_tensor.detach().float().cpu().squeeze().numpy()
    r = r_tensor.detach().float().cpu().squeeze().numpy()

    t_u8 = _normalize_u8(t, upper=255)
    r_u8 = _normalize_u8(r, upper=255)

    e_vis = (e - float(e.mean())) / (float(e.std()) + 1e-6)
    h_u8 = _normalize_u8(e_vis, upper=179)

    hsv = np.stack([h_u8, t_u8, r_u8], axis=2).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


def _save_factor_maps_for_sample(
    sample_key: str,
    gt_img: torch.Tensor,
    pred_img: torch.Tensor,
    gt_teacher: Dict[str, torch.Tensor],
    pred_teacher: Dict[str, torch.Tensor],
    out_root: Path,
    cmap_name: str,
    save_single_maps: bool,
    save_tev_hsv: bool,
) -> None:
    sample_dir = _ensure_dir(out_root / sample_key)

    pair_defs = [
        ("img", gt_img[0], pred_img[0]),
        ("T", gt_teacher["T_rad"][0], pred_teacher["T_rad"][0]),
        ("e", gt_teacher["e"][0], pred_teacher["e"][0]),
        ("R", gt_teacher["R_env"][0], pred_teacher["R_env"][0]),
        ("B", gt_teacher["B_edge"][0], pred_teacher["B_edge"][0]),
        ("A", gt_teacher["A"][0], pred_teacher["A"][0]),
        ("S_phys", gt_teacher["S_phys"][0], pred_teacher["S_phys"][0]),
    ]

    raw_payload: Dict[str, np.ndarray] = {}
    for name, gt_tensor, pred_tensor in pair_defs:
        if save_single_maps:
            gt_vis, pred_vis, diff_vis = _pair_normalize(gt_tensor.unsqueeze(0), pred_tensor.unsqueeze(0))
            _save_single_map_with_cmap(sample_dir / f"gt_{name}.png", gt_vis, cmap_name=cmap_name)
            _save_single_map_with_cmap(sample_dir / f"pred_{name}.png", pred_vis, cmap_name=cmap_name)
            _save_single_map_with_cmap(sample_dir / f"diff_{name}.png", diff_vis, cmap_name=cmap_name)
        raw_payload[f"gt_{name}"] = gt_tensor.detach().float().cpu().numpy()
        raw_payload[f"pred_{name}"] = pred_tensor.detach().float().cpu().numpy()

    if save_tev_hsv:
        gt_hsv = _make_tev_hsv_rgb(
            e_tensor=gt_teacher["e"][0],
            t_tensor=gt_teacher["T_rad"][0],
            r_tensor=gt_teacher["R_env"][0],
        )
        pred_hsv = _make_tev_hsv_rgb(
            e_tensor=pred_teacher["e"][0],
            t_tensor=pred_teacher["T_rad"][0],
            r_tensor=pred_teacher["R_env"][0],
        )
        diff_hsv = np.clip(np.abs(gt_hsv - pred_hsv), 0.0, 1.0)
        _save_rgb_image(sample_dir / "gt_hsv.png", gt_hsv)
        _save_rgb_image(sample_dir / "pred_hsv.png", pred_hsv)
        _save_rgb_image(sample_dir / "diff_hsv.png", diff_hsv)

    np.savez_compressed(sample_dir / "raw_factors.npz", **raw_payload)


def _save_factor_maps_batch(
    pair_items: Sequence[PairItem],
    start: int,
    end: int,
    gt_img: torch.Tensor,
    pred_img: torch.Tensor,
    gt_teacher: Dict[str, torch.Tensor],
    pred_teacher: Dict[str, torch.Tensor],
    out_root: Path,
    cmap_name: str,
    save_single_maps: bool,
    save_tev_hsv: bool,
) -> None:
    for offset, item in enumerate(pair_items[start:end]):
        _save_factor_maps_for_sample(
            sample_key=item.key,
            gt_img=gt_img[offset : offset + 1],
            pred_img=pred_img[offset : offset + 1],
            gt_teacher={key: value[offset : offset + 1] for key, value in gt_teacher.items()},
            pred_teacher={key: value[offset : offset + 1] for key, value in pred_teacher.items()},
            out_root=out_root,
            cmap_name=cmap_name,
            save_single_maps=save_single_maps,
            save_tev_hsv=save_tev_hsv,
        )


def _collect_metrics(
    gt_img: torch.Tensor,
    pred_img: torch.Tensor,
    gt_teacher: Dict[str, torch.Tensor],
    pred_teacher: Dict[str, torch.Tensor],
    hot_percent: float,
    rank_size: int,
) -> Dict[str, torch.Tensor]:
    edge_gt = normalize_01(sobel_mag(gt_img))
    edge_pred = normalize_01(sobel_mag(pred_img))
    pred_edge_gt_p, pred_edge_gt_r, pred_edge_gt_f1 = _edge_scores(pred_teacher["B_edge"], edge_gt)
    pred_edge_self_p, pred_edge_self_r, pred_edge_self_f1 = _edge_scores(pred_teacher["B_edge"], edge_pred)
    gt_edge_self_p, gt_edge_self_r, gt_edge_self_f1 = _edge_scores(gt_teacher["B_edge"], edge_gt)

    hotspot = _hotspot_metrics(pred_teacher["T_rad"], gt_teacher["T_rad"], hot_percent=hot_percent)

    metrics: Dict[str, torch.Tensor] = {
        "img_l1": l1_per_sample(pred_img, gt_img),
        "img_psnr": _psnr_per_sample(pred_img, gt_img),
        "img_ssim": ssim_per_sample(pred_img, gt_img),
        "phys_mae_e": l1_per_sample(pred_teacher["e"], gt_teacher["e"]),
        "phys_mae_t": l1_per_sample(pred_teacher["T_rad"], gt_teacher["T_rad"]),
        "phys_mae_renv": l1_per_sample(pred_teacher["R_env"], gt_teacher["R_env"]),
        "phys_mae_a": l1_per_sample(pred_teacher["A"], gt_teacher["A"]),
        "phys_mae_b": l1_per_sample(pred_teacher["B_edge"], gt_teacher["B_edge"]),
        "phys_l1_sphys": l1_per_sample(pred_teacher["S_phys"], gt_teacher["S_phys"]),
        "phys_psnr_sphys": _psnr_per_sample(pred_teacher["S_phys"], gt_teacher["S_phys"]),
        "phys_ssim_sphys": ssim_per_sample(pred_teacher["S_phys"], gt_teacher["S_phys"]),
        "pred_recomp_l1": l1_per_sample(pred_teacher["S_phys"], pred_img),
        "pred_recomp_psnr": _psnr_per_sample(pred_teacher["S_phys"], pred_img),
        "gt_recomp_l1": l1_per_sample(gt_teacher["S_phys"], gt_img),
        "gt_recomp_psnr": _psnr_per_sample(gt_teacher["S_phys"], gt_img),
        "thermal_order_spearman": _rank_correlation_per_sample(pred_teacher["T_rad"], gt_teacher["T_rad"], downsample=rank_size),
        "radiance_order_spearman": _rank_correlation_per_sample(pred_teacher["S_phys"], gt_teacher["S_phys"], downsample=rank_size),
        "pred_edge_gt_f1": pred_edge_gt_f1,
        "pred_edge_self_f1": pred_edge_self_f1,
        "gt_edge_self_f1": gt_edge_self_f1,
        "gt_mean_e": gt_teacher["e"].flatten(1).mean(dim=1),
        "pred_mean_e": pred_teacher["e"].flatten(1).mean(dim=1),
        "gt_mean_t": gt_teacher["T_rad"].flatten(1).mean(dim=1),
        "pred_mean_t": pred_teacher["T_rad"].flatten(1).mean(dim=1),
        "gt_mean_renv": gt_teacher["R_env"].flatten(1).mean(dim=1),
        "pred_mean_renv": pred_teacher["R_env"].flatten(1).mean(dim=1),
        "gt_mean_a": gt_teacher["A"].flatten(1).mean(dim=1),
        "pred_mean_a": pred_teacher["A"].flatten(1).mean(dim=1),
    }
    metrics["phys_l1_weighted"] = (
        metrics["phys_mae_e"]
        + metrics["phys_mae_t"]
        + metrics["phys_mae_renv"]
        + 0.5 * metrics["phys_mae_a"]
        + 0.25 * metrics["phys_mae_b"]
    )
    metrics.update({
        "hotspot_iou": hotspot["hotspot_iou"],
        "hotspot_precision": hotspot["hotspot_precision"],
        "hotspot_recall": hotspot["hotspot_recall"],
        "pseudo_hot_ratio": hotspot["pseudo_hot_ratio"],
    })
    return metrics


def _tensor_to_rows(
    pair_items: Sequence[PairItem],
    start: int,
    end: int,
    metrics: Dict[str, torch.Tensor],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for offset, item in enumerate(pair_items[start:end]):
        row: Dict[str, object] = {
            "sample_key": item.key,
            "gt_path": str(item.gt_path),
            "pred_path": str(item.pred_path),
            "input_path": str(item.input_path) if item.input_path is not None else "",
        }
        for key, value in metrics.items():
            row[key] = float(value[offset].detach().cpu().item())
        rows.append(row)
    return rows


def _save_qualitative_panel(
    pair_item: PairItem,
    gt_img: torch.Tensor,
    pred_img: torch.Tensor,
    gt_teacher: Dict[str, torch.Tensor],
    pred_teacher: Dict[str, torch.Tensor],
    hot_percent: float,
    out_path: Path,
    title_suffix: str,
) -> None:
    gt_hot = _hotspot_masks(gt_teacher["T_rad"], hot_percent=hot_percent)[0]
    pred_hot = _hotspot_masks(pred_teacher["T_rad"], hot_percent=hot_percent)[0]
    pred_only = (pred_hot * (1.0 - gt_hot)).clamp(0.0, 1.0)

    gt_t, pred_t, diff_t = _pair_normalize(gt_teacher["T_rad"][0:1], pred_teacher["T_rad"][0:1])
    gt_r, pred_r, diff_r = _pair_normalize(gt_teacher["R_env"][0:1], pred_teacher["R_env"][0:1])
    gt_a, pred_a, diff_a = _pair_normalize(gt_teacher["A"][0:1], pred_teacher["A"][0:1])

    rows = [
        [
            ("GT", _to_numpy01(gt_img[0])),
            ("Pred", _to_numpy01(pred_img[0])),
            ("|GT-Pred|", _normalize_auto(np.abs(_to_numpy01(gt_img[0]) - _to_numpy01(pred_img[0])))),
            ("GT hot", _to_numpy01(gt_hot)),
            ("Pred hot", _to_numpy01(pred_hot)),
            ("Pseudo hot", _to_numpy01(pred_only)),
        ],
        [
            ("GT S_phys", _to_numpy01(gt_teacher["S_phys"][0])),
            ("Pred S_phys", _to_numpy01(pred_teacher["S_phys"][0])),
            ("|dS_phys|", _normalize_auto(np.abs(_to_numpy01(gt_teacher["S_phys"][0]) - _to_numpy01(pred_teacher["S_phys"][0])))),
            ("GT T_rad", gt_t),
            ("Pred T_rad", pred_t),
            ("|dT|", diff_t),
        ],
        [
            ("GT e", _to_numpy01(gt_teacher["e"][0])),
            ("Pred e", _to_numpy01(pred_teacher["e"][0])),
            ("|de|", _normalize_auto(np.abs(_to_numpy01(gt_teacher["e"][0]) - _to_numpy01(pred_teacher["e"][0])))),
            ("GT R_env", gt_r),
            ("Pred R_env", pred_r),
            ("|dR|", diff_r),
        ],
        [
            ("GT A", gt_a),
            ("Pred A", pred_a),
            ("|dA|", diff_a),
            ("GT B_edge", _to_numpy01(gt_teacher["B_edge"][0])),
            ("Pred B_edge", _to_numpy01(pred_teacher["B_edge"][0])),
            ("|dB|", _normalize_auto(np.abs(_to_numpy01(gt_teacher["B_edge"][0]) - _to_numpy01(pred_teacher["B_edge"][0])))),
        ],
    ]

    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(13.5, 9.0), squeeze=False)
    for row_idx, row in enumerate(rows):
        for col_idx, (name, image) in enumerate(row):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                ax.set_title(name, fontsize=9)
    fig.suptitle(f"{pair_item.key} | {title_suffix}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_selected_panels(
    pair_items: Sequence[PairItem],
    rows: Sequence[Dict[str, object]],
    teacher: TeR_B,
    device: torch.device,
    hot_percent: float,
    out_dir: Path,
    key_name: str,
    reverse: bool,
    num_vis: int,
) -> None:
    if num_vis <= 0 or not rows:
        return
    order = sorted(rows, key=lambda item: float(item[key_name]), reverse=reverse)[:num_vis]
    pair_lookup = {item.key: item for item in pair_items}
    for rank, row in enumerate(order, start=1):
        pair_item = pair_lookup[row["sample_key"]]
        gt_img = _load_gray_image(pair_item.gt_path).unsqueeze(0).to(device=device, dtype=torch.float32)
        pred_img = _load_gray_image(pair_item.pred_path).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            gt_teacher = teacher(gt_img)
            pred_teacher = teacher(pred_img)
        title = (
            f"{key_name}={float(row[key_name]):.4f} | "
            f"order_rho={float(row['thermal_order_spearman']):.4f} | "
            f"hot_iou={float(row['hotspot_iou']):.4f}"
        )
        filename = f"{rank:02d}_{pair_item.key}.png"
        _save_qualitative_panel(
            pair_item=pair_item,
            gt_img=gt_img,
            pred_img=pred_img,
            gt_teacher=gt_teacher,
            pred_teacher=pred_teacher,
            hot_percent=hot_percent,
            out_path=out_dir / filename,
            title_suffix=title,
        )


def _analyze_dataset(
    dataset_cfg: Dict,
    teacher_cfg: Dict,
    run_cfg: Dict,
    repo_root: Path,
) -> Dict[str, object]:
    dataset_name = str(dataset_cfg["name"])
    device_cfg = str(run_cfg.get("device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    pair_dir = _resolve_path(dataset_cfg.get("pair_dir", ""), repo_root)
    gt_dir = _resolve_path(dataset_cfg.get("gt_dir", ""), repo_root)
    pred_dir = _resolve_path(dataset_cfg.get("pred_dir", ""), repo_root)
    input_dir = _resolve_path(dataset_cfg.get("input_dir", ""), repo_root)
    match_by = str(dataset_cfg.get("match_by", "stem")).lower()

    teacher_ckpt = _resolve_path(dataset_cfg.get("teacher_ckpt", ""), repo_root)
    if teacher_ckpt is None:
        teacher_ckpt = _find_teacher_ckpt(repo_root, dataset_name)
    if teacher_ckpt is None or not teacher_ckpt.is_file():
        raise FileNotFoundError(
            f"Could not find teacher checkpoint for dataset={dataset_name}. "
            "Please set datasets[].teacher_ckpt in the config."
        )

    teacher = _load_teacher(teacher_cfg, teacher_ckpt, device=device)
    pair_source: Dict[str, str] = {}
    if pair_dir is not None and pair_dir.is_dir():
        pair_items = _discover_pairs(
            pair_dir=pair_dir,
            gt_prefix=str(dataset_cfg.get("gt_prefix", "gt_")),
            pred_prefix=str(dataset_cfg.get("pred_prefix", "pred_")),
            input_prefix=str(dataset_cfg.get("input_prefix", "input_")),
        )
        pair_source = {"mode": "shared_dir", "pair_dir": str(pair_dir)}
    elif gt_dir is not None and pred_dir is not None and gt_dir.is_dir() and pred_dir.is_dir():
        if gt_dir.resolve() == pred_dir.resolve():
            pair_items = _discover_pairs(
                pair_dir=gt_dir,
                gt_prefix=str(dataset_cfg.get("gt_prefix", "gt_")),
                pred_prefix=str(dataset_cfg.get("pred_prefix", "pred_")),
                input_prefix=str(dataset_cfg.get("input_prefix", "input_")),
            )
            pair_source = {"mode": "shared_dir", "pair_dir": str(gt_dir)}
        else:
            pair_items = _discover_pairs_from_split_dirs(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                input_dir=input_dir,
                match_by=match_by,
            )
            pair_source = {
                "mode": "split_dirs",
                "gt_dir": str(gt_dir),
                "pred_dir": str(pred_dir),
                "input_dir": str(input_dir) if input_dir is not None else "",
                "match_by": match_by,
            }
    else:
        pair_dir = _find_pair_dir(repo_root, dataset_name)
        if pair_dir is None or not pair_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find paired data for dataset={dataset_name}. "
                "Please set datasets[].pair_dir or datasets[].gt_dir + datasets[].pred_dir in the config."
            )
        pair_items = _discover_pairs(
            pair_dir=pair_dir,
            gt_prefix=str(dataset_cfg.get("gt_prefix", "gt_")),
            pred_prefix=str(dataset_cfg.get("pred_prefix", "pred_")),
            input_prefix=str(dataset_cfg.get("input_prefix", "input_")),
        )
        pair_source = {"mode": "shared_dir", "pair_dir": str(pair_dir)}

    max_samples = int(run_cfg.get("max_samples", -1))
    if max_samples > 0:
        pair_items = pair_items[:max_samples]

    batch_size = max(1, int(run_cfg.get("batch_size", 8)))
    hot_percent = float(run_cfg.get("hot_percent", 1.0))
    rank_size = int(run_cfg.get("rank_size", 64))
    num_vis = max(0, int(run_cfg.get("num_vis", 8)))
    save_factor_maps_all = bool(run_cfg.get("save_factor_maps_all", True))
    factor_map_cmap = str(run_cfg.get("factor_map_cmap", "inferno"))
    factor_map_dirname = str(run_cfg.get("factor_map_dirname", "factor_maps"))
    save_single_factor_maps = bool(run_cfg.get("save_single_factor_maps", True))
    save_tev_hsv = bool(run_cfg.get("save_tev_hsv", False))

    output_root = _ensure_dir((_resolve_path(run_cfg.get("output_dir", "outputs/phys_consistency"), repo_root) or (repo_root / "outputs" / "phys_consistency")) / dataset_name)
    factor_map_root = output_root / factor_map_dirname

    rows: List[Dict[str, object]] = []
    for start in tqdm(range(0, len(pair_items), batch_size), desc=f"[{dataset_name}] physics", unit="batch"):
        end = min(start + batch_size, len(pair_items))
        gt_img, pred_img, gt_teacher, pred_teacher = _batch_teacher_forward(
            teacher=teacher,
            pair_items=pair_items,
            start=start,
            end=end,
            device=device,
        )
        metrics = _collect_metrics(
            gt_img=gt_img,
            pred_img=pred_img,
            gt_teacher=gt_teacher,
            pred_teacher=pred_teacher,
            hot_percent=hot_percent,
            rank_size=rank_size,
        )
        rows.extend(_tensor_to_rows(pair_items, start, end, metrics))
        if save_factor_maps_all:
            _save_factor_maps_batch(
                pair_items=pair_items,
                start=start,
                end=end,
                gt_img=gt_img,
                pred_img=pred_img,
                gt_teacher=gt_teacher,
                pred_teacher=pred_teacher,
                out_root=factor_map_root,
                cmap_name=factor_map_cmap,
                save_single_maps=save_single_factor_maps,
                save_tev_hsv=save_tev_hsv,
            )

    per_sample_csv = output_root / "per_sample_metrics.csv"
    _save_rows_csv(per_sample_csv, rows)
    summary = {
        "dataset": dataset_name,
        "pair_source": pair_source,
        "teacher_ckpt": str(teacher_ckpt),
        "num_pairs": len(rows),
        "metrics": _numeric_summary(rows),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    compact_row = {"dataset": dataset_name, "num_pairs": len(rows)}
    for key, stats in summary["metrics"].items():
        compact_row[key] = stats["mean"]

    _save_selected_panels(
        pair_items=pair_items,
        rows=rows,
        teacher=teacher,
        device=device,
        hot_percent=hot_percent,
        out_dir=output_root / "worst_pseudo_hot",
        key_name="pseudo_hot_ratio",
        reverse=True,
        num_vis=num_vis,
    )
    _save_selected_panels(
        pair_items=pair_items,
        rows=rows,
        teacher=teacher,
        device=device,
        hot_percent=hot_percent,
        out_dir=output_root / "worst_phys_factor",
        key_name="phys_l1_weighted",
        reverse=True,
        num_vis=num_vis,
    )

    return {
        "summary": summary,
        "compact_row": compact_row,
        "output_dir": str(output_root),
    }


def main() -> None:
    args = _make_parser().parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    repo_root = Path(__file__).resolve().parents[1]

    run_cfg = dict(cfg.get("run", {}))
    teacher_cfg = dict(cfg.get("teacher", {}))
    datasets_cfg = list(cfg.get("datasets", []))
    if not datasets_cfg:
        raise ValueError("Config must contain a non-empty `datasets` list.")

    results: List[Dict[str, object]] = []
    compact_rows: List[Dict[str, object]] = []
    for dataset_cfg in datasets_cfg:
        result = _analyze_dataset(
            dataset_cfg=dict(dataset_cfg),
            teacher_cfg=teacher_cfg,
            run_cfg=run_cfg,
            repo_root=repo_root,
        )
        results.append(result["summary"])
        compact_rows.append(result["compact_row"])
        print(
            f"[DONE] {dataset_cfg['name']}: "
            f"pairs={result['summary']['num_pairs']} "
            f"output={result['output_dir']}"
        )

    output_root = _resolve_path(run_cfg.get("output_dir", "outputs/phys_consistency"), repo_root) or (repo_root / "outputs" / "phys_consistency")
    output_root = _ensure_dir(output_root)
    with (output_root / "all_datasets_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    _save_rows_csv(output_root / "all_datasets_means.csv", compact_rows)
    print(f"[INFO] Saved global summary to: {output_root}")


if __name__ == "__main__":
    main()
