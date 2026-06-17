import argparse
import csv
import json
import os
import shutil
import tempfile

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from psl_flow import PSLFlow
from utils.load_cfg import load_config, load_datasets_config
from dataloaders.GenericDataloader import GenericDataModule
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models


def _copy_checkpoint_alias(src_path: str, dst_path: str, alias_name: str) -> None:
    if not src_path or not os.path.isfile(src_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        fd, tmp_path = tempfile.mkstemp(prefix=f".{alias_name}_", suffix=".ckpt.tmp", dir=os.path.dirname(dst_path))
        os.close(fd)
        try:
            shutil.copy2(src_path, tmp_path)
            os.replace(tmp_path, dst_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    print(f"[INFO] Saved {alias_name} checkpoint alias: {dst_path}")


def _scalarize_metrics(metrics, include_fn=None):
    scalar_metrics = {}
    for key, value in metrics.items():
        if include_fn is not None and not include_fn(str(key)):
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            scalar_metrics[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, bool, str)):
            scalar_metrics[key] = value
    return scalar_metrics


def _is_validation_metric(name: str) -> bool:
    return '/val' in name or name.startswith('val_') or name.startswith('val/') or '_val/' in name


def _is_test_metric(name: str) -> bool:
    return '/test' in name or name.startswith('test_') or name.startswith('test/') or '_test/' in name


def _include_train_metric(name: str) -> bool:
    return not _is_validation_metric(name) and not _is_test_metric(name)


def _include_validation_metric(name: str) -> bool:
    return _is_validation_metric(name) or name in {'latent_std', 'latent_mean', 'latent_normalizer'}


def _include_test_metric(name: str) -> bool:
    return _is_test_metric(name)


class LocalMetricsCallback(pl.Callback):
    def __init__(self, root_dir: str):
        super().__init__()
        self.log_dir = os.path.join(root_dir, "local_logs")
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.jsonl_path = os.path.join(self.log_dir, "history.jsonl")
        self._rows = []
        self._fieldnames = []
        self._seen = set()
        self._load_existing()

    @staticmethod
    def _record_key(record):
        return (
            str(record.get("event")),
            str(record.get("epoch")),
            str(record.get("global_step")),
        )

    def _load_existing(self):
        os.makedirs(self.log_dir, exist_ok=True)
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self._fieldnames = list(reader.fieldnames or [])
                self._rows = list(reader)
        for row in self._rows:
            self._seen.add(self._record_key(row))

    def _write_csv(self):
        if not self._rows:
            return
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

    def _append_record(self, record):
        record_key = self._record_key(record)
        if record_key in self._seen:
            return
        self._seen.add(record_key)
        for key in record.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(record)
        self._write_csv()
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_record(self, trainer, event, include_fn=None):
        record = {
            "event": event,
            "epoch": int(trainer.current_epoch) + 1,
            "global_step": int(trainer.global_step),
        }
        record.update(_scalarize_metrics(trainer.callback_metrics, include_fn=include_fn))
        return record

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._append_record(self._build_record(trainer, "train_epoch_end", include_fn=_include_train_metric))

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        self._append_record(self._build_record(trainer, "validation_end", include_fn=_include_validation_metric))

    def on_test_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._append_record(self._build_record(trainer, "test_end", include_fn=_include_test_metric))


class BestSampleExportCallback(pl.Callback):
    def __init__(self, checkpoint_cb: pl.callbacks.ModelCheckpoint):
        super().__init__()
        self.checkpoint_cb = checkpoint_cb
        self._last_best_model_path = ""

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        best_model_path = str(getattr(self.checkpoint_cb, "best_model_path", "") or "")
        if not best_model_path or best_model_path == self._last_best_model_path:
            return

        epoch_tag = f"epoch_{int(trainer.current_epoch) + 1:04d}"
        src_dir = os.path.join(trainer.default_root_dir, "val_samples", epoch_tag)
        if not os.path.isdir(src_dir):
            print(f"[WARN] Best checkpoint updated but validation samples not found: {src_dir}")
            self._last_best_model_path = best_model_path
            return

        best_root = os.path.join(trainer.default_root_dir, "best_samples")
        latest_dir = os.path.join(best_root, "latest")
        archive_dir = os.path.join(best_root, epoch_tag)
        os.makedirs(best_root, exist_ok=True)

        if os.path.isdir(archive_dir):
            shutil.rmtree(archive_dir)
        shutil.copytree(src_dir, archive_dir)

        if os.path.isdir(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(src_dir, latest_dir)

        best_score = getattr(self.checkpoint_cb, "best_model_score", None)
        if best_score is not None:
            try:
                best_score = float(best_score.detach().cpu().item())
            except Exception:
                try:
                    best_score = float(best_score)
                except Exception:
                    best_score = None

        with open(os.path.join(best_root, "best_info.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "epoch": int(trainer.current_epoch) + 1,
                    "best_model_path": best_model_path,
                    "best_model_score": best_score,
                    "sample_source_dir": src_dir,
                    "latest_dir": latest_dir,
                    "archive_dir": archive_dir,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        self._last_best_model_path = best_model_path
        print(f"[INFO] Saved best validation samples to: {latest_dir}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args.add_argument('--limit-train-batches', type=float, default=None)
    args.add_argument('--limit-val-batches', type=float, default=None)
    args.add_argument('--check-val-every-n-epoch', type=int, default=None)
    args.add_argument('--devices', type=int, default=None)
    args.add_argument('--num-nodes', type=int, default=1)
    args.add_argument('--accelerator', type=str, default='gpu')
    args.add_argument('--strategy', type=str, default='ddp')
    args.add_argument('--disable-wandb', action='store_true')
    args.add_argument('--resume-from', type=str, default=None)
    args.add_argument('--default-root-dir', type=str, default='./logs/')
    args.add_argument('--vae-path', type=str, default=None)
    args.add_argument('--rgb-vae-path', type=str, default=None)
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    if args.vae_path is not None and args.vae_path != "":
        train_cfg.model.model_config["vae_path"] = args.vae_path
        print(f"[INFO] Override thermal VAE path: {args.vae_path}")
    if args.rgb_vae_path is not None and args.rgb_vae_path != "":
        train_cfg.model.model_config["rgb_vae_path"] = args.rgb_vae_path
        print(f"[INFO] Override RGB VAE path: {args.rgb_vae_path}")
    wandb_logger = None if args.disable_wandb else WandbLogger(name=args.config.split('/')[-1].split('.')[0], entity="unistgl", project="PSL-Flow")
    if wandb_logger is not None:
        logger = wandb_logger
    else:
        logger = CSVLogger(save_dir=args.default_root_dir, name="lightning_csv")
        print(f"[INFO] W&B disabled; using local CSV logger at {logger.log_dir}")
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
    
    if hasattr(train_cfg.training, "load") and train_cfg.training.load_type == "finetune":
        model = PSLFlow.load_from_checkpoint(train_cfg.training.load,
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
                                                training_stage=train_cfg.training.training_stage if hasattr(train_cfg.training, "training_stage") else "full",
                                                strict=False)
    else:
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
            training_stage=train_cfg.training.training_stage if hasattr(train_cfg.training, "training_stage") else "full",
            gradient_accumulation=train_cfg.training.gradient_accumulation if hasattr(train_cfg.training, "gradient_accumulation") else 1,
            calculate_stats=train_cfg.training.calculate_stats if hasattr(train_cfg.training, "calculate_stats") else False,
        )

    if train_cfg.training.mixed_precision:
        mixed_precision_setting = "16-mixed"
    else:
        mixed_precision_setting = "32"

    limit_train_batches = args.limit_train_batches
    if limit_train_batches is None:
        limit_train_batches = float(train_cfg.training.limit_train_batches) if hasattr(train_cfg.training, "limit_train_batches") else 1.0

    limit_val_batches = args.limit_val_batches
    if limit_val_batches is None:
        limit_val_batches = float(train_cfg.training.limit_val_batches) if hasattr(train_cfg.training, "limit_val_batches") else 1.0

    check_val_every_n_epoch = args.check_val_every_n_epoch
    if check_val_every_n_epoch is None:
        if hasattr(train_cfg.training, "check_val_every_n_epoch"):
            check_val_every_n_epoch = int(train_cfg.training.check_val_every_n_epoch)
        elif hasattr(train_cfg.training, "val_freq"):
            check_val_every_n_epoch = int(train_cfg.training.val_freq)
        else:
            check_val_every_n_epoch = 1

    target_val_dataset = train_cfg.datasets.target_val_dataset
    checkpoint_monitor = train_cfg.training.checkpoint_monitor if hasattr(train_cfg.training, "checkpoint_monitor") else f"{target_val_dataset}_val/FID"
    checkpoint_mode = train_cfg.training.checkpoint_mode if hasattr(train_cfg.training, "checkpoint_mode") else "min"
    checkpoint_save_top_k = int(train_cfg.training.checkpoint_save_top_k) if hasattr(train_cfg.training, "checkpoint_save_top_k") else 1
    checkpoint_save_last = bool(train_cfg.training.checkpoint_save_last) if hasattr(train_cfg.training, "checkpoint_save_last") else True
    val_enabled = float(limit_val_batches) > 0
    checkpoint_dir = os.path.join(args.default_root_dir, "checkpoints")

    if hasattr(train_cfg.training, "checkpoint_filename"):
        filename = str(train_cfg.training.checkpoint_filename)
    elif str(checkpoint_monitor).endswith("/FID"):
        monitor_root = str(checkpoint_monitor).rsplit("/", 1)[0]
        filename = (
            f"{model.model_arch}"
            + "_{epoch:02d}_FID[{"
            + f"{checkpoint_monitor}:.4f"
            + "}]_LPIPS[{"
            + f"{monitor_root}/LPIPS:.4f"
            + "}]"
        )
    else:
        filename = f"{model.model_arch}" + "_{epoch:02d}"

    best_checkpoint_cb = None
    if val_enabled and checkpoint_save_top_k != 0:
        best_checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_top_k=checkpoint_save_top_k,
            filename=filename,
            auto_insert_metric_name=False,
            save_weights_only=False,
            save_on_train_epoch_end=False,
            save_last=False,
        )
    elif checkpoint_save_top_k != 0:
        if checkpoint_save_top_k != 0:
            print("[WARN] Validation is disabled (`limit_val_batches=0`), skip best-checkpoint monitoring and only save `last`.")

    last_checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=0,
        filename=f"{model.model_arch}" + "_{epoch:02d}",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=checkpoint_save_last,
    )

    local_metrics_cb = LocalMetricsCallback(args.default_root_dir)
    callbacks = [last_checkpoint_cb, local_metrics_cb]
    if best_checkpoint_cb is not None:
        callbacks.append(best_checkpoint_cb)
    if wandb_logger is not None:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    if best_checkpoint_cb is not None:
        callbacks.append(BestSampleExportCallback(best_checkpoint_cb))

    detect_anomaly = bool(train_cfg.training.detect_anomaly) if hasattr(train_cfg.training, "detect_anomaly") else False
    devices = args.devices if args.devices is not None else torch.cuda.device_count()
    strategy_name = args.strategy.lower()
    if strategy_name in {"ddp", "ddp_find_unused_parameters_true"}:
        strategy = DDPStrategy(find_unused_parameters=True) if int(devices) > 1 else "auto"
    elif strategy_name == "auto":
        if int(devices) > 1 and model.automatic_optimization is False:
            strategy = DDPStrategy(find_unused_parameters=True)
            print("[INFO] Using DDPStrategy(find_unused_parameters=True) for multi-GPU manual optimization.")
        else:
            strategy = "auto"
    else:
        strategy = args.strategy
    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        default_root_dir=args.default_root_dir, # Tensorflow can be used to viz 
        num_nodes=args.num_nodes,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=mixed_precision_setting,
        max_epochs=train_cfg.training.num_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,# we only run the checkpointing callback (you can add more)
        log_every_n_steps=20,
        strategy=strategy,
        detect_anomaly=detect_anomaly,
        logger=logger,
    )
    print(f"[INFO] Trainer check_val_every_n_epoch={check_val_every_n_epoch}")

    # we call the trainer, we give it the model and the datamodule
    # trainer.validate(model=model, datamodule=datamodule)
    if args.resume_from is not None and args.resume_from != "":
        print(f"RESUME FROM CKPT (CLI): {args.resume_from}")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_from)
    elif hasattr(train_cfg.training, "load"):
        print(f"Loading Model: {train_cfg.training.load}")
        if train_cfg.training.load_type == "resume":
            print("RESUME FROM CKPT")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=train_cfg.training.load)
        elif train_cfg.training.load_type == "finetune":
            print("FINETUNE FROM CKPT")
            trainer.fit(model=model, datamodule=datamodule)
    else:
        print("Training from scratch")
        trainer.fit(model=model, datamodule=datamodule)

    if trainer.is_global_zero:
        checkpoint_alias_dir = os.path.join(args.default_root_dir, "checkpoints")
        _copy_checkpoint_alias(last_checkpoint_cb.last_model_path, os.path.join(checkpoint_alias_dir, "last.ckpt"), "last")
        if best_checkpoint_cb is not None:
            _copy_checkpoint_alias(best_checkpoint_cb.best_model_path, os.path.join(checkpoint_alias_dir, "best.ckpt"), "best")
        print(f"[INFO] Local metrics CSV: {local_metrics_cb.csv_path}")
        print(f"[INFO] Local metrics history: {local_metrics_cb.jsonl_path}")

    # torch.distributed.destroy_process_group()
    # if trainer.is_global_zero:
    #     trainer = pl.Trainer(
    #         accelerator='gpu',
    #         devices=1,
    #         default_root_dir=f'./logs/', # Tensorflow can be used to viz 
    #         num_nodes=1,
    #         precision=mixed_precision_setting,
    #         logger=wandb_logger,
    #     )
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path="last")
