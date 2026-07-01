from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torchvision.utils import save_image


def configure_training_runtime(training: dict[str, Any]) -> None:
    seed = training.get("seed", None)
    if seed not in (None, ""):
        pl.seed_everything(int(seed), workers=True)

    matmul_precision = str(training.get("float32_matmul_precision", "high") or "").strip()
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)

    if bool(training.get("cuda_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    plain: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().float().cpu().item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            plain[str(key)] = number
    return plain


class LocalMetricsCallback(pl.Callback):
    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir) if root_dir else None

    def _log_dir(self, trainer: pl.Trainer) -> Path:
        root = self.root_dir or Path(trainer.default_root_dir)
        path = root / "local_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, trainer: pl.Trainer, stage: str) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        metrics = scalar_metrics(dict(trainer.callback_metrics))
        if not metrics:
            return
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": metrics,
        }
        log_dir = self._log_dir(trainer)
        with (log_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        with (log_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["time", "stage", "epoch", "global_step", "key", "value"])
            if handle.tell() == 0:
                writer.writeheader()
            for key, value in sorted(metrics.items()):
                writer.writerow(
                    {
                        "time": payload["time"],
                        "stage": stage,
                        "epoch": payload["epoch"],
                        "global_step": payload["global_step"],
                        "key": key,
                        "value": value,
                    }
                )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write(trainer, "train")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write(trainer, "val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write(trainer, "test")


def _display_tensor(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    image = x.detach().float().cpu()[:max_samples]
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.shape[1] not in (1, 3):
        image = image[:, :1]
    flat = image.flatten(1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    needs_norm = bool((mins < -1e-4).any() or (maxs > 1.0001).any())
    if needs_norm:
        image = (image - mins) / (maxs - mins).clamp_min(1e-6)
    return image.clamp(0.0, 1.0)


def _as_three(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    return x.repeat(1, 3, 1, 1)


class QualitativeSampleCallback(pl.Callback):
    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
        max_images: int = 4,
        max_batches: int = 1,
        monitor: str = "val/loss_total",
        mode: str = "min",
    ):
        self.root_dir = Path(root_dir) if root_dir else None
        self.max_images = int(max_images)
        self.max_batches = int(max_batches)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.best_score: float | None = None
        self.last_val_epoch_dir: Path | None = None

    def _root(self, trainer: pl.Trainer) -> Path:
        return self.root_dir or Path(trainer.default_root_dir)

    def _save_payload(
        self,
        trainer: pl.Trainer,
        *,
        payload: dict[str, torch.Tensor],
        split: str,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking or batch_idx >= self.max_batches:
            return
        tensors = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        if not tensors:
            return
        epoch_dir = self._root(trainer) / f"{split}_samples" / f"epoch_{int(trainer.current_epoch):04d}"
        out_dir = epoch_dir / f"loader_{dataloader_idx}" / f"batch_{batch_idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_items = []
        for key, value in tensors.items():
            image = _display_tensor(value, self.max_images)
            save_image(image, out_dir / f"{key}.png")
            grid_items.append(_as_three(image))
        if grid_items:
            save_image(torch.cat(grid_items, dim=0), out_dir / "grid.png", nrow=self.max_images)
        if split == "val":
            self.last_val_epoch_dir = epoch_dir

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if isinstance(outputs, dict):
            self._save_payload(
                trainer,
                payload=outputs,
                split="val",
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if isinstance(outputs, dict):
            self._save_payload(
                trainer,
                payload=outputs,
                split="test",
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking or self.last_val_epoch_dir is None:
            return
        metrics = scalar_metrics(dict(trainer.callback_metrics))
        if self.monitor not in metrics:
            return
        score = metrics[self.monitor]
        improved = self.best_score is None
        if self.best_score is not None:
            improved = score < self.best_score if self.mode == "min" else score > self.best_score
        if not improved:
            return
        self.best_score = score
        dst = self._root(trainer) / "best_samples" / "latest"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.last_val_epoch_dir, dst)


def build_management_callbacks(
    training: dict[str, Any],
    *,
    root_dir: str | Path | None,
    monitor: str,
    mode: str,
) -> list[pl.Callback]:
    callbacks: list[pl.Callback] = [LocalMetricsCallback(root_dir=root_dir)]
    if bool(training.get("export_samples", True)):
        callbacks.append(
            QualitativeSampleCallback(
                root_dir=root_dir,
                max_images=int(training.get("num_sample_images", 4)),
                max_batches=int(training.get("num_sample_batches", 1)),
                monitor=monitor,
                mode=mode,
            )
        )
    return callbacks


def copy_best_checkpoint_alias(callbacks: list[pl.Callback], root_dir: str | Path) -> None:
    root = Path(root_dir)
    target = root / "checkpoints" / "best.ckpt"
    for callback in callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_path = getattr(callback, "best_model_path", "")
            if best_path:
                source = Path(best_path)
                if source.is_file() and source.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    print(f"[checkpoint] copied best checkpoint: {source} -> {target}")
