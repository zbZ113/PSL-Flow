from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml

from psl_flow.data import build_data_module, thermal_to_01
from psl_flow.models.psl_vae.losses import PSLVAELoss
from psl_flow.models.psl_vae.psl_vae import PSLVAE, build_terb_teacher
from psl_flow.utils.checkpoint import load_state_dict_flexible


class PSLVAELightningModule(pl.LightningModule):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = dict(config.get("model", {}).get("model_config", {}))
        loss_cfg = dict(config.get("training", {}).get("loss", {}).get("config", {}))
        teacher_cfg = dict(loss_cfg.get("teacher", config.get("teacher", {})))
        teacher_ckpt = str(teacher_cfg.get("ckpt", ""))
        self.model = PSLVAE(model_cfg)
        if teacher_ckpt:
            teacher, info = build_terb_teacher(
                teacher_cfg,
                ckpt_path=teacher_ckpt,
                strict=bool(teacher_cfg.get("strict_load", False)),
            )
            print(f"[PSL-VAE] Loaded frozen TeR-B teacher: {info}")
            self.model.attach_teacher(teacher)
        self.loss_fn = PSLVAELoss(loss_cfg)
        self.optimizer_cfg = dict(config.get("training", {}).get("optimizer", {"name": "AdamW", "lr": 6e-5}))

    def forward(self, thermal_01: torch.Tensor):
        return self.model.forward_from_ir(thermal_01, sample=bool(self.training))

    def _step(self, batch, prefix: str):
        thermal_01 = thermal_to_01(batch[1])
        outputs = self(thermal_01)
        losses = self.loss_fn(outputs, thermal_01)
        self.log(f"{prefix}/loss_total", losses["loss_total"], prog_bar=True, sync_dist=True)
        for key, value in losses.items():
            if key != "loss_total":
                self.log(f"{prefix}/{key}", value, sync_dist=True)
        self.log(f"{prefix}/latent_std", outputs["z_phys"].std(unbiased=False), sync_dist=True)
        self.log(f"{prefix}/latent_mean", outputs["z_phys"].mean(), sync_dist=True)
        return losses["loss_total"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        lr = float(self.optimizer_cfg.get("lr", 6e-5))
        weight_decay = float(self.optimizer_cfg.get("weight_decay", 1e-3))
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


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


def main() -> None:
    parser = argparse.ArgumentParser("Train paper-aligned PSL-VAE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["fit", "validate"], default="fit")
    parser.add_argument("--default-root-dir", default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data = build_data_module(config)
    training = dict(config.get("training", {}))
    module = PSLVAELightningModule(config)
    if args.load:
        print(load_state_dict_flexible(module.model, args.load, strict=False, strip_prefixes=("model.", "module.", "psl_vae.")))
    callbacks = []
    if args.mode == "fit":
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=None,
                save_last=True,
                monitor=str(training.get("checkpoint_monitor", "val/loss_total")),
                mode=str(training.get("checkpoint_mode", "min")),
                save_top_k=int(training.get("checkpoint_save_top_k", 1)),
            )
        )
    trainer = pl.Trainer(
        max_epochs=int(training.get("num_epochs", 100)),
        max_steps=args.max_steps if args.max_steps is not None else int(training.get("max_steps", -1)),
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
        precision="16-mixed" if bool(training.get("mixed_precision", False)) else "32-true",
        callbacks=callbacks,
        logger=False if bool(training.get("disable_logger", False)) else True,
        limit_val_batches=args.limit_val_batches
        if args.limit_val_batches is not None
        else training.get("limit_val_batches", None),
        check_val_every_n_epoch=args.check_val_every_n_epoch
        if args.check_val_every_n_epoch is not None
        else int(training.get("check_val_every_n_epoch", 1)),
        accumulate_grad_batches=int(training.get("gradient_accumulation", 1)),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
    )
    if args.mode == "fit":
        trainer.fit(module, datamodule=data, ckpt_path=args.resume_from)
        if trainer.is_global_zero:
            final_ckpt = Path(args.default_root_dir or trainer.default_root_dir) / "checkpoints" / "last.ckpt"
            final_ckpt.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(final_ckpt))
            print(f"[PSL-VAE] saved final checkpoint: {final_ckpt}")
    else:
        results = trainer.validate(module, datamodule=data, ckpt_path=args.ckpt)
        if trainer.is_global_zero:
            _emit_metrics("PSL-VAE validation", results, args.metrics_json)


if __name__ == "__main__":
    main()
