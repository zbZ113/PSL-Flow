import argparse
import csv
import glob
import json
import os
import zipfile

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from psl_flow import PSLFlow
from utils.load_cfg import load_config, load_datasets_config
from dataloaders.GenericDataloader import GenericDataModule
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models


def _normalize_cli_string(value: str) -> str:
    return value.strip().lower()


def _is_zip_checkpoint(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and zipfile.is_zipfile(path)


def _iter_checkpoint_fallbacks(ckpt_path: str):
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    run_dir = os.path.dirname(ckpt_dir)
    basename = os.path.basename(ckpt_path).lower()

    if basename == "best.ckpt":
        best_info_path = os.path.join(run_dir, "best_samples", "best_info.json")
        if os.path.isfile(best_info_path):
            try:
                with open(best_info_path, "r", encoding="utf-8") as handle:
                    best_info = json.load(handle)
            except Exception:
                best_info = {}
            best_model_path = str(best_info.get("best_model_path", "") or "")
            if best_model_path:
                yield best_model_path

    try:
        sibling_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    except OSError:
        sibling_ckpts = []

    for candidate in sibling_ckpts:
        candidate_name = os.path.basename(candidate).lower()
        if candidate_name in {"best.ckpt", "last.ckpt"}:
            continue
        yield candidate


def _resolve_eval_checkpoint_path(ckpt_path: str | None) -> str | None:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return ckpt_path

    basename = os.path.basename(ckpt_path).lower()
    if basename not in {"best.ckpt", "last.ckpt"}:
        return ckpt_path
    if _is_zip_checkpoint(ckpt_path):
        return ckpt_path

    print(f"[WARN] Checkpoint alias looks corrupted or incomplete: {ckpt_path}")
    seen = {os.path.abspath(ckpt_path)}
    for candidate in _iter_checkpoint_fallbacks(ckpt_path):
        candidate_abs = os.path.abspath(candidate)
        if candidate_abs in seen or not _is_zip_checkpoint(candidate):
            continue
        print(f"[INFO] Fallback to valid checkpoint: {candidate}")
        return candidate
    raise RuntimeError(
        "Checkpoint alias is unreadable and no valid fallback checkpoint was found. "
        f"Please re-export the alias or pass a concrete checkpoint file instead: {ckpt_path}"
    )


def _scalarize_metrics(metrics):
    scalar_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            scalar_metrics[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, bool, str)):
            scalar_metrics[key] = value
    return scalar_metrics


class LocalMetricsCallback(pl.Callback):
    def __init__(self, root_dir: str):
        super().__init__()
        self.log_dir = os.path.join(root_dir, "local_logs")
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.jsonl_path = os.path.join(self.log_dir, "history.jsonl")
        self._rows = []
        self._fieldnames = []
        os.makedirs(self.log_dir, exist_ok=True)

    def _write_csv(self):
        if not self._rows:
            return
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

    def _record_event(self, trainer, event_name: str):
        if not trainer.is_global_zero:
            return
        record = {
            "event": event_name,
            "epoch": int(trainer.current_epoch) + 1,
            "global_step": int(trainer.global_step),
        }
        record.update(_scalarize_metrics(trainer.callback_metrics))
        for key in record.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(record)
        self._write_csv()
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_validation_end(self, trainer, pl_module):
        self._record_event(trainer, "validation_end")

    def on_test_end(self, trainer, pl_module):
        self._record_event(trainer, "test_end")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args.add_argument('--devices', type=int, default=1)
    args.add_argument('--num-nodes', type=int, default=1)
    args.add_argument('--accelerator', type=str, default='gpu')
    args.add_argument('--strategy', type=str, default='auto')
    args.add_argument('--disable-wandb', action='store_true')
    args.add_argument('--vae-path', type=str, default=None)
    args.add_argument('--rgb-vae-path', type=str, default=None)
    args.add_argument('--default-root-dir', type=str, default='./logs/')
    args.add_argument('--eval-splits', type=_normalize_cli_string, default='test', choices=['val', 'test', 'both'])
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    if args.vae_path is not None and args.vae_path != "":
        train_cfg.model.model_config["vae_path"] = args.vae_path
        print(f"[INFO] Override thermal VAE path: {args.vae_path}")
    if args.rgb_vae_path is not None and args.rgb_vae_path != "":
        train_cfg.model.model_config["rgb_vae_path"] = args.rgb_vae_path
        print(f"[INFO] Override RGB VAE path: {args.rgb_vae_path}")
    disable_wandb_env = str(os.environ.get("WANDB_MODE", "")).lower() == "disabled" or str(os.environ.get("WANDB_DISABLED", "")).lower() in {"1", "true", "yes", "y", "on"}
    wandb_logger = None if (args.disable_wandb or disable_wandb_env) else WandbLogger(name=args.config.split('/')[-1].split('.')[0], entity="unistgl", project="PSL-Flow")
    logger = wandb_logger if wandb_logger is not None else False
    datamodule = GenericDataModule(
        datasets_folder=train_cfg.datasets.datasets_folder,
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
        mixed_precision=True if train_cfg.training.mixed_precision else False,
    )
    
    model = PSLFlow(
        #---- Encoder
        model_arch=train_cfg.model.model_arch,
        model_config=train_cfg.model.model_config,
        lr=train_cfg.training.optimizer["lr"],
        optimizer=train_cfg.training.optimizer["name"],
        weight_decay=train_cfg.training.optimizer["weight_decay"], # 0.001 for sgd and 0 for adam,
        momentum=train_cfg.training.optimizer["momentum"],
        lr_sched=train_cfg.training.scheduler["name"],
        lr_sched_args = train_cfg.training.scheduler["args"],

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=train_cfg.training.loss["name"],
        loss_config=train_cfg.training.loss["config"],
        validation_type=train_cfg.training.validation_type
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=f'{train_cfg.datasets.target_val_dataset}_val/FID',
        filename=f'{model.model_arch}' + '_{epoch:02d}_FID[{' + f'{train_cfg.datasets.target_val_dataset}' + '_val/FID:.4f}]_LPIPS[{' + f'{train_cfg.datasets.target_val_dataset}' + '_val/LPIPS:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        save_last=True,
        mode='min'
    )

    local_metrics_cb = LocalMetricsCallback(args.default_root_dir)
    callbacks = [checkpoint_cb, local_metrics_cb]
    if wandb_logger is not None:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    mixed_precision_setting = "32"
    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.default_root_dir, # Tensorflow can be used to viz 
        num_nodes=args.num_nodes,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=mixed_precision_setting,
        max_epochs=train_cfg.training.num_epochs,
        check_val_every_n_epoch=train_cfg.training.val_freq, # run validation every epoch
        callbacks=callbacks,# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        strategy=args.strategy,
        logger=logger,
    )

    ckpt_path = None if (not hasattr(train_cfg.training, "load") or train_cfg.training.load == "None") else train_cfg.training.load
    ckpt_path = _resolve_eval_checkpoint_path(ckpt_path)
    if ckpt_path is None:
        print("Warning: Dummy model evaluation")
    else:
        print(f"[INFO] Evaluation checkpoint: {ckpt_path}")
    if args.eval_splits in {"val", "both"}:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    if args.eval_splits in {"test", "both"}:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    if trainer.is_global_zero:
        print(f"[INFO] Local test metrics CSV: {local_metrics_cb.csv_path}")
        print(f"[INFO] Local test metrics history: {local_metrics_cb.jsonl_path}")




