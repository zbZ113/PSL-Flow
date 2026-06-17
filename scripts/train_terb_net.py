from __future__ import annotations

import argparse
import csv
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from phys_train_common import (
    PROJECT_ROOT,
    append_jsonl,
    append_metrics_csv,
    bool_flag,
    build_datamodule,
    build_eval_loaders,
    count_parameters,
    close_progress,
    device_memory_stats,
    device_summary,
    effective_max_steps,
    extract_thermal,
    prepare_stage_output_dirs,
    release_runtime_resources,
    reset_device_peak_memory,
    resolve_resume_path,
    save_json,
    save_teacher_visualization,
    set_seed,
    shutdown_dataloaders,
    InterruptHandler,
    thermal_to_01,
)
from dataloaders.norm_params import IMAGENET_MEAN_STD, NORMAL_MEAN_STD
from utils.load_cfg import load_config
from models.physics import (
    TeR_B,
    load_module_checkpoint,
    normalize_01,
    sobel_mag,
    ssim_per_sample,
    tv_loss,
    tv_weighted,
)


CSV_FIELDS = [
    "epoch",
    "global_step",
    "train_loss_total",
    "train_loss_recon",
    "train_loss_img",
    "train_loss_ssim",
    "train_psnr",
    "train_ssim_score",
    "train_loss_edge",
    "train_edge_precision",
    "train_edge_recall",
    "train_edge_f1",
    "train_loss_env_grad",
    "train_loss_e",
    "train_loss_t",
    "train_loss_r",
    "train_loss_a",
    "train_loss_a0",
    "train_a_mean",
    "train_a_abs1",
    "train_v4_entropy",
    "train_lambda_env",
    "val_loss_total",
    "val_loss_recon",
    "val_loss_img",
    "val_loss_ssim",
    "val_psnr",
    "val_ssim_score",
    "val_loss_edge",
    "val_edge_precision",
    "val_edge_recall",
    "val_edge_f1",
    "val_loss_env_grad",
    "val_loss_e",
    "val_loss_t",
    "val_loss_r",
    "val_loss_a",
    "val_loss_a0",
    "val_a_mean",
    "val_a_abs1",
    "val_v4_entropy",
    "val_lambda_env",
    "val_rec_ir_mean",
    "val_rec_vis_mean",
    "val_rec_gap",
    "val_rec_emd",
    "best_val_loss",
    "best_val_loss_recon",
    "train_steps",
    "val_steps",
    "gpu_mem_peak_gb",
    "is_best",
]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train TeR-B Net")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default="logs/physics/teacher")
    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--val-every-epochs", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=0, help="Deprecated; stage training keeps only best/last checkpoints.")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--datasets-folder", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--amp", type=str, default="true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-vis-every-epochs", type=int, default=0, help="Deprecated; visualization is updated only when best improves.")
    parser.add_argument("--num-vis-samples", type=int, default=4)
    return parser


def _extract_rgb(batch) -> torch.Tensor:
    if not isinstance(batch, (list, tuple)) or len(batch) < 1:
        raise RuntimeError(f"Unexpected batch structure: {type(batch)}")
    return batch[0]


def _rgb_to_gray_01(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB batch [B,3,H,W], got {rgb.shape}")
    if float(rgb.min().detach().cpu()) < -1.2 or float(rgb.max().detach().cpu()) > 1.2:
        mean = torch.tensor(IMAGENET_MEAN_STD["mean"], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_MEAN_STD["std"], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    else:
        mean = torch.tensor(NORMAL_MEAN_STD["mean"], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        std = torch.tensor(NORMAL_MEAN_STD["std"], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    rgb_01 = torch.clamp(rgb * std + mean, 0.0, 1.0)
    gray = 0.2989 * rgb_01[:, 0:1] + 0.5870 * rgb_01[:, 1:2] + 0.1140 * rgb_01[:, 2:3]
    return gray.clamp(0.0, 1.0)


def _psnr_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _edge_scores(pred: torch.Tensor, gt: torch.Tensor, pred_thresh: float = 0.5, gt_thresh: float = 0.3, eps: float = 1e-6):
    pred_bin = (pred >= float(pred_thresh)).float()
    gt_bin = (gt >= float(gt_thresh)).float()
    tp = (pred_bin * gt_bin).flatten(1).sum(dim=1)
    fp = (pred_bin * (1.0 - gt_bin)).flatten(1).sum(dim=1)
    fn = ((1.0 - pred_bin) * gt_bin).flatten(1).sum(dim=1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def _v4_entropy(v4: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prob = v4.clamp_min(eps)
    ent = -(prob * torch.log(prob)).sum(dim=1)
    return ent.flatten(1).mean(dim=1) / math.log(float(v4.shape[1]))


def _recon_loss_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().flatten(1).mean(dim=1) + 0.2 * (1.0 - ssim_per_sample(pred, target))


def _wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a = torch.sort(a.flatten()).values
    b = torch.sort(b.flatten()).values
    if a.numel() != b.numel():
        n = max(int(a.numel()), int(b.numel()))
        qa = torch.linspace(0.0, 1.0, n, device=a.device)
        qb = torch.linspace(0.0, 1.0, n, device=b.device)
        a = torch.quantile(a, qa)
        b = torch.quantile(b, qb)
    return float((a - b).abs().mean().detach().cpu().item())


def _compute_teacher_losses(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    s_01 = out["S_01"]
    s_phys = out["S_phys"]
    b_edge_logits = out["B_edge_logits"]
    b_edge = out["B_edge"]
    r_env = out["R_env"]
    e = out["e"]
    t_rad = out["T_rad"]
    a = out["A"]
    v4 = out["V4"]

    e_gt = normalize_01(sobel_mag(s_01))
    loss_img = F.l1_loss(s_phys, s_01)
    ssim_score = ssim_per_sample(s_phys, s_01).mean()
    loss_ssim = 1.0 - ssim_score
    loss_recon = loss_img + 0.2 * loss_ssim
    loss_edge = F.binary_cross_entropy_with_logits(b_edge_logits.float(), e_gt.float())
    loss_env_grad = ((1.0 - b_edge.detach()) * sobel_mag(r_env)).mean()
    loss_e = tv_weighted(e, 1.0 - b_edge.detach())
    loss_t = tv_loss(t_rad)
    loss_r = tv_loss(r_env)
    loss_a = tv_loss(a)
    loss_a0 = torch.mean(torch.abs(a - 1.0))
    edge_precision, edge_recall, edge_f1 = _edge_scores(b_edge, e_gt)

    loss_total = (
        1.0 * loss_img
        + 0.2 * loss_ssim
        + 0.1 * loss_edge
        + 0.05 * loss_env_grad
        + 0.05 * loss_e
        + 0.02 * loss_t
        + 0.02 * loss_r
        + 0.05 * loss_a
        + 0.02 * loss_a0
    )
    return {
        "loss_total": loss_total,
        "loss_recon": loss_recon,
        "loss_img": loss_img,
        "loss_ssim": loss_ssim,
        "psnr": _psnr_per_sample(s_phys, s_01).mean(),
        "ssim_score": ssim_score,
        "loss_edge": loss_edge,
        "edge_precision": edge_precision.mean(),
        "edge_recall": edge_recall.mean(),
        "edge_f1": edge_f1.mean(),
        "loss_env_grad": loss_env_grad,
        "loss_e": loss_e,
        "loss_t": loss_t,
        "loss_r": loss_r,
        "loss_a": loss_a,
        "loss_a0": loss_a0,
        "a_mean": a.mean(),
        "a_abs1": torch.abs(a - 1.0).mean(),
        "v4_entropy": _v4_entropy(v4).mean(),
        "lambda_env": out["lambda_env"].mean(),
    }


def _init_meter() -> Dict[str, float]:
    return {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_img": 0.0,
        "loss_ssim": 0.0,
        "psnr": 0.0,
        "ssim_score": 0.0,
        "loss_edge": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "loss_env_grad": 0.0,
        "loss_e": 0.0,
        "loss_t": 0.0,
        "loss_r": 0.0,
        "loss_a": 0.0,
        "loss_a0": 0.0,
        "a_mean": 0.0,
        "a_abs1": 0.0,
        "v4_entropy": 0.0,
        "lambda_env": 0.0,
    }


def _meter_to_metrics(prefix: str, meter: Dict[str, float], seen: int) -> Dict[str, float]:
    denom = max(1, int(seen))
    return {f"{prefix}_{k}": v / denom for k, v in meter.items()}


def _accumulate(meter: Dict[str, float], losses: Dict[str, torch.Tensor], batch_size: int) -> None:
    for key in meter.keys():
        meter[key] += float(losses[key].detach().cpu().item()) * batch_size


def _save_training_state(
    path: Path,
    teacher: TeR_B,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_val_loss_recon: float,
    best_epoch: int,
    no_improve_validations: int,
    args,
    last_completed_epoch: int | None = None,
) -> None:
    if last_completed_epoch is None:
        last_completed_epoch = int(epoch) - 1
    payload = {
        "format_version": 1,
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "next_epoch": int(epoch),
        "last_completed_epoch": int(last_completed_epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "best_val_loss_recon": float(best_val_loss_recon),
        "best_epoch": int(best_epoch),
        "no_improve_validations": int(no_improve_validations),
        "args": vars(args),
    }
    torch.save(payload, path)


def _evaluate(teacher, val_loaders, device, use_amp: bool):
    teacher.eval()
    meter = _init_meter()
    seen = 0
    steps = 0
    rec_ir_samples = []
    rec_vis_samples = []
    for loader_idx, loader in enumerate(val_loaders, start=1):
        progress = tqdm(loader, desc=f"[Teacher] Validate {loader_idx}/{len(val_loaders)}", leave=False, dynamic_ncols=True)
        for batch in progress:
            rgb = _extract_rgb(batch).to(device, non_blocking=True)
            thermal = extract_thermal(batch).to(device, non_blocking=True)
            s_01 = thermal_to_01(thermal)
            rgb_gray_01 = _rgb_to_gray_01(rgb)
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if device.type == "cuda" else nullcontext()
            with torch.no_grad():
                with amp_ctx:
                    out = teacher(s_01)
                    losses = _compute_teacher_losses(out)
                    vis_out = teacher(rgb_gray_01)
                    rec_ir = _recon_loss_sample(out["S_phys"], s_01)
                    rec_vis = _recon_loss_sample(vis_out["S_phys"], rgb_gray_01)
            batch_size = int(thermal.shape[0])
            _accumulate(meter, losses, batch_size)
            rec_ir_samples.append(rec_ir.detach().cpu())
            rec_vis_samples.append(rec_vis.detach().cpu())
            seen += batch_size
            steps += 1
    metrics = _meter_to_metrics("val", meter, seen)
    metrics["val_steps"] = int(steps)
    if rec_ir_samples and rec_vis_samples:
        rec_ir_all = torch.cat(rec_ir_samples)
        rec_vis_all = torch.cat(rec_vis_samples)
        metrics["val_rec_ir_mean"] = float(rec_ir_all.mean().item())
        metrics["val_rec_vis_mean"] = float(rec_vis_all.mean().item())
        metrics["val_rec_gap"] = float((rec_vis_all.mean() - rec_ir_all.mean()).item())
        metrics["val_rec_emd"] = _wasserstein_1d(rec_vis_all, rec_ir_all)
    else:
        metrics["val_rec_ir_mean"] = 0.0
        metrics["val_rec_vis_mean"] = 0.0
        metrics["val_rec_gap"] = 0.0
        metrics["val_rec_emd"] = 0.0
    return metrics


def _save_visual_snapshot(teacher, val_loaders, device, use_amp: bool, path: Path, num_vis_samples: int) -> str | None:
    if not val_loaders:
        return None
    try:
        batch = next(iter(val_loaders[0]))
    except StopIteration:
        return None
    thermal = extract_thermal(batch).to(device, non_blocking=True)
    s_01 = thermal_to_01(thermal)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        with amp_ctx:
            out = teacher(s_01)
    save_teacher_visualization(out, path, max_samples=num_vis_samples)
    return str(path)


def main() -> None:
    args = _make_parser().parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    physics_cfg = cfg.model.model_config.get("physics_config", {}) if isinstance(cfg.model.model_config, dict) else {}
    teacher_cfg = physics_cfg.get("teacher", {}) if isinstance(physics_cfg.get("teacher", {}), dict) else {}
    if not teacher_cfg:
        raise ValueError("model.model_config.physics_config.teacher is required.")

    run_dir = Path(args.run_dir)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (run_dir / "checkpoints")
    args.resume = resolve_resume_path(args.resume, [run_dir / "states" / "last.pth"])
    state_dir, visuals_dir = prepare_stage_output_dirs(run_dir, ckpt_dir, "Teacher", fresh_start=not bool(args.resume))
    if args.resume:
        print(f"[Teacher] Auto resume enabled: {args.resume}")
    save_json(run_dir / "args.json", vars(args))
    save_json(run_dir / "config_ref.json", {"config": str(Path(args.config))})

    dm, train_batch_size, num_workers, datasets_folder = build_datamodule(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        datasets_folder=args.datasets_folder,
    )
    train_loader = dm.train_dataloader()
    val_loaders = build_eval_loaders(dm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool_flag(args.amp) and device.type == "cuda"
    teacher = TeR_B(
        smp_model=str(teacher_cfg.get("smp_model", "Unet")),
        smp_encoder=str(teacher_cfg.get("smp_encoder", "resnet18")),
        smp_encoder_weights=teacher_cfg.get("smp_encoder_weights", None),
        vnums=int(teacher_cfg.get("vnums", 4)),
        erme_kernel=int(teacher_cfg.get("erme_kernel", 5)),
        lambda_env_init=float(teacher_cfg.get("lambda_env_init", 0.1)),
        a_low_range=tuple(teacher_cfg.get("a_low_range", [0.8, 1.2])),
    ).to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_val_recon = float("inf")
    best_epoch = -1
    no_improve_validations = 0
    if args.resume:
        resume_obj = torch.load(args.resume, map_location="cpu")
        if isinstance(resume_obj, dict) and "teacher" in resume_obj:
            legacy_epoch_format = "next_epoch" not in resume_obj and "last_completed_epoch" not in resume_obj
            teacher.load_state_dict(resume_obj["teacher"], strict=True)
            if "optimizer" in resume_obj:
                optimizer.load_state_dict(resume_obj["optimizer"])
            if use_amp and "scaler" in resume_obj:
                scaler.load_state_dict(resume_obj["scaler"])
            start_epoch = int(resume_obj.get("next_epoch", resume_obj.get("epoch", 0)))
            global_step = int(resume_obj.get("global_step", 0))
            best_val_loss = float(resume_obj.get("best_val_loss", best_val_loss))
            best_val_recon = float(resume_obj.get("best_val_loss_recon", resume_obj.get("best_val_loss", best_val_recon)))
            best_epoch = int(resume_obj.get("best_epoch", best_epoch))
            if legacy_epoch_format and best_epoch >= 0:
                best_epoch -= 1
            no_improve_validations = int(resume_obj.get("no_improve_validations", 0))
            print(f"[Teacher] Resumed training state from {args.resume} @ epoch={start_epoch}, global_step={global_step}")
        else:
            load_info = load_module_checkpoint(teacher, args.resume, strict=False)
            print(
                f"[Teacher] Loaded weights from {args.resume} "
                f"(source={load_info.get('state_source', 'unknown')}, state_tensors={load_info['num_state_tensors']}, "
                f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})."
            )

    max_steps = effective_max_steps(args.max_steps_per_epoch, getattr(cfg.training, "num_samples_per_epoch", 0), train_batch_size)
    print("[Teacher] Setup")
    print(f"  device={device}, amp={use_amp}")
    print(f"  device_summary={device_summary(device)}")
    print(f"  log_dir={run_dir}")
    print(f"  ckpt_dir={ckpt_dir}")
    print(f"  datasets_folder={datasets_folder}")
    print(f"  batch_size={train_batch_size}, num_workers={num_workers}")
    print(f"  val_loaders={len(val_loaders)}")
    print(f"  max_steps_per_epoch={max_steps if max_steps > 0 else '<full_loader>'}")
    print(f"  num_parameters={count_parameters(teacher):,}")

    history_path = run_dir / "history.jsonl"
    metrics_csv_path = run_dir / "metrics.csv"
    best_metrics_path = run_dir / "best_metrics.json"
    last_metrics_path = run_dir / "last_metrics.json"
    summary_path = run_dir / "summary.json"

    completed_epochs = start_epoch
    stopped_early = False
    interrupted = False
    progress = None
    interrupt_handler = InterruptHandler("Teacher")
    interrupt_handler.install()
    try:
        for epoch in range(start_epoch, int(args.epochs)):
            reset_device_peak_memory(device)
            teacher.train()
            meter = _init_meter()
            seen = 0
            steps = 0
            progress = tqdm(train_loader, desc=f"[Teacher] Epoch {epoch}/{max(0, int(args.epochs) - 1)}", dynamic_ncols=True)
            for batch in progress:
                thermal = extract_thermal(batch).to(device, non_blocking=True)
                s_01 = thermal_to_01(thermal)
                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if device.type == "cuda" else nullcontext()
                with amp_ctx:
                    out = teacher(s_01)
                    losses = _compute_teacher_losses(out)
                scaler.scale(losses["loss_total"]).backward()
                if float(args.grad_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(args.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()

                batch_size = int(thermal.shape[0])
                _accumulate(meter, losses, batch_size)
                seen += batch_size
                steps += 1
                global_step += 1
                if steps == 1 or steps % max(1, int(args.log_every_steps)) == 0:
                    progress.set_postfix(loss=f"{float(losses['loss_total'].detach().cpu().item()):.5f}")
                if max_steps > 0 and steps >= max_steps:
                    break
            epoch_index = int(epoch)
            next_epoch = epoch_index + 1
            completed_epochs = next_epoch

            train_metrics = _meter_to_metrics("train", meter, seen)
            train_metrics["train_steps"] = int(steps)
            metrics = {
                "epoch": epoch_index,
                "global_step": int(global_step),
                **train_metrics,
                "best_val_loss": float(best_val_loss) if best_epoch >= 0 else None,
                "best_val_loss_recon": float(best_val_recon) if best_epoch >= 0 else None,
                "val_steps": 0,
                "is_best": False,
            }

            run_validation = next_epoch % max(1, int(args.val_every_epochs)) == 0 or next_epoch == int(args.epochs)
            if run_validation:
                val_metrics = _evaluate(teacher, val_loaders, device, use_amp)
                metrics.update(val_metrics)
                val_total = float(val_metrics["val_loss_total"])
                val_recon = float(val_metrics["val_loss_recon"])
                improved = best_epoch < 0 or val_recon < (best_val_recon - float(args.early_stop_min_delta))
                if improved:
                    best_val_loss = val_total
                    best_val_recon = val_recon
                    best_epoch = epoch_index
                    no_improve_validations = 0
                    metrics["best_val_loss"] = float(best_val_loss)
                    metrics["best_val_loss_recon"] = float(best_val_recon)
                    metrics["is_best"] = True
                    _save_training_state(
                        state_dir / "best.pth",
                        teacher,
                        optimizer,
                        scaler,
                        next_epoch,
                        global_step,
                        best_val_loss,
                        best_val_recon,
                        best_epoch,
                        no_improve_validations,
                        args,
                        last_completed_epoch=epoch_index,
                    )
                    torch.save(teacher.state_dict(), ckpt_dir / "teacher_best.pth")
                else:
                    no_improve_validations += 1
                    metrics["best_val_loss"] = float(best_val_loss)
                    metrics["best_val_loss_recon"] = float(best_val_recon)

                if metrics.get("is_best", False) and int(args.num_vis_samples) > 0:
                    vis_path = _save_visual_snapshot(teacher, val_loaders, device, use_amp, visuals_dir / "best.png", int(args.num_vis_samples))
                    if vis_path is not None:
                        metrics["vis_path"] = vis_path

                if metrics.get("is_best", False):
                    save_json(best_metrics_path, metrics)

            metrics.update(device_memory_stats(device))
            print(
                f"[Teacher] Epoch {epoch_index}: train_recon={metrics['train_loss_recon']:.6f} "
                + (f"val_recon={metrics['val_loss_recon']:.6f} val_psnr={metrics['val_psnr']:.3f} val_ssim={metrics['val_ssim_score']:.4f} " if 'val_loss_recon' in metrics else "")
                + f"peak_mem={metrics['gpu_mem_peak_gb']:.3f}GB"
            )

            _save_training_state(
                state_dir / "last.pth",
                teacher,
                optimizer,
                scaler,
                next_epoch,
                global_step,
                best_val_loss,
                best_val_recon,
                best_epoch,
                no_improve_validations,
                args,
                last_completed_epoch=epoch_index,
            )

            torch.save(teacher.state_dict(), ckpt_dir / "teacher_last.pth")

            append_jsonl(history_path, metrics)
            append_metrics_csv(metrics_csv_path, metrics, CSV_FIELDS)
            save_json(last_metrics_path, metrics)

            if int(args.early_stop_patience) > 0 and no_improve_validations >= int(args.early_stop_patience):
                print(
                    f"[Teacher] Early stop at epoch={epoch_index} "
                    f"(no_improve_validations={no_improve_validations}, best_epoch={best_epoch}, best_val_recon={best_val_recon:.6f}, best_val_total={best_val_loss:.6f})"
                )
                stopped_early = True
                break

    except KeyboardInterrupt:
        interrupted = True
        close_progress(progress)
        print(f"[Teacher] Interrupted at epoch={max(-1, int(completed_epochs) - 1)} global_step={global_step}. Writing interrupt summary...")
        emergency = {
            "event": "interrupted",
            "epoch": max(-1, int(completed_epochs) - 1),
            "global_step": int(global_step),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss) if best_epoch >= 0 else None,
            "best_val_loss_recon": float(best_val_recon) if best_epoch >= 0 else None,
        }
        append_jsonl(history_path, emergency)
        save_json(run_dir / "interrupt_summary.json", emergency)
    finally:
        close_progress(progress)
        shutdown_dataloaders(train_loader)
        shutdown_dataloaders(val_loaders)
        release_runtime_resources(device)
        interrupt_handler.restore()
        if interrupted:
            os._exit(130)

    summary = {
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if best_epoch >= 0 else None,
        "best_val_loss_recon": float(best_val_recon) if best_epoch >= 0 else None,
        "stopped_early": bool(stopped_early),
        "log_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "state_dir": str(state_dir),
        "project_root": str(PROJECT_ROOT),
    }
    save_json(summary_path, summary)
    print(f"[Teacher] Finished. Summary: {summary}")


if __name__ == "__main__":
    main()
