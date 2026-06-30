from __future__ import annotations

import glob
import json
import os
import random
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
import yaml
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import v2


IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
THERMAL_MEAN_STD = {"mean": [0.5], "std": [0.5]}
NORMAL_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def _to_image_tensor(x):
    if hasattr(v2, "ToImage"):
        return v2.ToImage()(x)
    if isinstance(x, torch.Tensor):
        if x.ndim == 2:
            return x.unsqueeze(0)
        return x
    if isinstance(x, Image.Image):
        return TF.pil_to_tensor(x)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            t = t.permute(2, 0, 1)
        return t
    return TF.pil_to_tensor(Image.fromarray(np.asarray(x)))


def _to_dtype(dtype: torch.dtype, scale: bool = True):
    if hasattr(v2, "ToDtype"):
        try:
            return v2.ToDtype(dtype, scale=scale)
        except TypeError:
            if scale and hasattr(v2, "ConvertImageDtype"):
                return v2.ConvertImageDtype(dtype)
            return v2.ToDtype(dtype)
    if scale and hasattr(v2, "ConvertImageDtype"):
        return v2.ConvertImageDtype(dtype)
    return T.ConvertImageDtype(dtype)


def _ensure_float_image(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        return x.float().div(255.0)
    if x.numel() > 0:
        x_max = float(x.max().detach().cpu())
        x_min = float(x.min().detach().cpu())
        if x_max > 1.0 or x_min < 0.0:
            return x.div(255.0)
    return x


def _identity_nodesplitter(src):
    return src


class RandomChainDataset(torch.utils.data.ChainDataset):
    def __init__(self, datasets, num_samples_per_epoch=None, probs=None, longest=True):
        super().__init__(datasets)
        self.probs = probs
        self.longest = longest
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self):
        sources = [iter(d) for d in self.datasets]
        return self.random_samples(sources, self.probs, longest=self.longest)

    def random_samples(self, sources, probs=None, longest=False):
        if probs is None:
            probs = [1] * len(sources)
        else:
            probs = list(probs)
        while len(sources) > 0:
            cum = (np.array(probs) / np.sum(probs)).cumsum()
            r = random.random()
            i = np.searchsorted(cum, r)
            try:
                yield next(sources[i])
            except StopIteration:
                if longest:
                    del sources[i]
                    del probs[i]
                else:
                    break

    def __len__(self):
        if self.num_samples_per_epoch is not None:
            return self.num_samples_per_epoch
        total = 0
        for dataset in self.datasets:
            assert isinstance(dataset, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(dataset)
        return total


class PaperWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets_folder: str = "./datasets_preprocess",
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        train_image_size: tuple[int, int] = (256, 256),
        num_workers: int = 8,
        dataset_names=None,
        train_cfg_training=None,
        mixed_precision: bool = False,
        image_norm: str = "normal",
        dataset_config_root: str | None = None,
    ):
        super().__init__()
        self.datasets_folder = str(datasets_folder)
        self.train_batch_size = int(train_batch_size)
        self.test_batch_size = int(test_batch_size)
        self.train_image_size = tuple(train_image_size)
        self.num_workers = int(num_workers)
        self.image_norm = str(image_norm)
        self.dataset_config_root = Path(dataset_config_root) if dataset_config_root else (
            Path(__file__).resolve().parents[1] / "configs" / "paper" / "datasets"
        )

        if self.image_norm == "normal":
            image_mean_std = NORMAL_MEAN_STD
        elif self.image_norm == "imagenet":
            image_mean_std = IMAGENET_MEAN_STD
        else:
            raise ValueError(f"Unsupported image_norm: {self.image_norm}")

        self.mean_image = image_mean_std["mean"]
        self.std_image = image_mean_std["std"]
        self.mean_thermal = THERMAL_MEAN_STD["mean"]
        self.std_thermal = THERMAL_MEAN_STD["std"]
        self.train_dataset_names = list(dataset_names.train_datasets)
        self.val_dataset_names = list(dataset_names.val_datasets)
        self.test_dataset_names = list(dataset_names.test_datasets)
        self.train_cfg_training = train_cfg_training
        self.train_datasets_cfg = self._load_dataset_configs(self.train_dataset_names)
        self.val_datasets_cfg = self._load_dataset_configs(self.val_dataset_names)
        self.test_datasets_cfg = self._load_dataset_configs(self.test_dataset_names)

        dtype = torch.float16 if mixed_precision else torch.float32
        self.train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    self.train_image_size,
                    scale=(0.5, 1.0),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                _to_dtype(dtype, scale=True),
            ]
        )
        self.val_transform = v2.Compose(
            [
                v2.Resize(
                    self.train_image_size,
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                _to_dtype(dtype, scale=True),
            ]
        )
        self.test_transform = self.val_transform

    def _load_dataset_configs(self, names: list[str]) -> dict[str, Namespace]:
        loaded: dict[str, Namespace] = {}
        for name in names:
            path = self.dataset_config_root / f"{name}.yml"
            if not path.exists():
                raise FileNotFoundError(f"Missing dataset config: {path}")
            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            loaded[name] = Namespace(**payload)
        return loaded

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_datasets = []
            for dataset_name in self.train_dataset_names:
                cfg = self.train_datasets_cfg[dataset_name]
                self.train_datasets.append(
                    self._build_dataset(
                        "train",
                        cfg.train,
                        self.train_transform,
                        int(cfg.dataset_index),
                        dataset_name,
                        num_samples_per_epoch=int(self.train_cfg_training.num_samples_per_epoch),
                    )
                )

        if stage in (None, "fit", "validate"):
            self.val_datasets = []
            for dataset_name in self.val_dataset_names:
                cfg = self.val_datasets_cfg[dataset_name]
                self.val_datasets.append(
                    self._build_dataset("val", cfg.val, self.val_transform, int(cfg.dataset_index), dataset_name)
                )

        if stage in (None, "fit", "test"):
            self.test_datasets = []
            for dataset_name in self.test_dataset_names:
                cfg = self.test_datasets_cfg[dataset_name]
                self.test_datasets.append(
                    self._build_dataset("test", cfg.test, self.test_transform, int(cfg.dataset_index), dataset_name)
                )

        print(
            {
                "train_datasets": self.train_datasets_cfg,
                "val_datasets": self.val_datasets_cfg,
                "test_datasets": self.test_datasets_cfg,
            }
        )

    def _build_dataset(
        self,
        split: str,
        dataset_configs: list[dict[str, Any]],
        transform,
        dataset_index: int,
        dataset_name: str,
        num_samples_per_epoch: int | None = None,
    ):
        dataset_list = []
        for dataset_config in dataset_configs:
            datafolder_name = dataset_config["datafolder_name"]
            shard_root = os.path.join(self.datasets_folder, datafolder_name)
            allshards = sorted(glob.glob(os.path.join(shard_root, "dataset-*.tar")))
            if len(allshards) == 0:
                raise FileNotFoundError(f"No shards found under {shard_root}")

            is_train = split == "train"
            dataset = wds.WebDataset(
                allshards,
                nodesplitter=wds.split_by_node if is_train else _identity_nodesplitter,
                resampled=is_train,
                shardshuffle=is_train,
                empty_check=False,
            )
            tuple_pattern = ["color.png", "thermal.png", "__key__"]
            map_fn = lambda sample, index=dataset_index: self._preprocess_pair(sample, transform, index)
            if is_train:
                dataset = (
                    dataset.shuffle(1000)
                    .decode("pil")
                    .to_tuple(*tuple_pattern)
                    .map(map_fn)
                    .with_epoch(num_samples_per_epoch)
                    .with_length(num_samples_per_epoch)
                )
            else:
                metadata_path = os.path.join(shard_root, "metadata.json")
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                dataset = dataset.decode("pil").to_tuple(*tuple_pattern).map(map_fn).with_length(metadata["num_samples"])
                dataset.total_len = metadata["num_samples"]

            dataset.split = split
            dataset.dataset_name = dataset_name
            dataset.my_shard_num = len(allshards)
            dataset_list.append(dataset)
            print(f"< {dataset.__class__.__name__}, {dataset.dataset_name} - #samples: {len(dataset)} >")

        if len(dataset_list) > 1:
            if split == "train":
                dataset = RandomChainDataset(dataset_list, num_samples_per_epoch=num_samples_per_epoch)
                dataset.split = dataset_list[-1].split
            else:
                dataset = torch.utils.data.ChainDataset(dataset_list)
                dataset.split = dataset_list[-1].split
            dataset.dataset_name = dataset_name
            dataset.my_shard_num = sum(getattr(item, "my_shard_num", 1) for item in dataset_list)
            return dataset
        return dataset_list[0]

    def _preprocess_pair(self, sample, transform, dataset_index: int):
        rgb_in = _to_image_tensor(sample[0])
        thermal_in = v2.Grayscale(num_output_channels=1)(_to_image_tensor(sample[1]))
        if rgb_in.shape[-2:] != thermal_in.shape[-2:]:
            thermal_in = v2.Resize(rgb_in.shape[-2:])(thermal_in)

        combined = torch.cat([rgb_in, thermal_in], dim=0)
        combined = transform(combined)
        rgb = _ensure_float_image(combined[:3, :, :])
        thermal = _ensure_float_image(combined[3:4, :, :])
        rgb = v2.Normalize(mean=self.mean_image, std=self.std_image)(rgb)
        thermal = v2.Normalize(mean=self.mean_thermal, std=self.std_thermal)(thermal)
        return rgb, thermal, dataset_index

    def train_dataloader(self):
        mixed_dataset = RandomChainDataset(
            self.train_datasets,
            num_samples_per_epoch=int(self.train_cfg_training.num_samples_per_epoch),
        )
        return DataLoader(
            dataset=mixed_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=None,
        )

    def _eval_loader(self, dataset):
        workers = max(min(int(getattr(dataset, "my_shard_num", 1)), self.num_workers), 1)
        return DataLoader(
            dataset=dataset,
            batch_size=self.test_batch_size,
            num_workers=workers,
            drop_last=False,
            pin_memory=True,
            shuffle=None,
        )

    def val_dataloader(self):
        return [self._eval_loader(dataset) for dataset in self.val_datasets]

    def test_dataloader(self):
        return [self._eval_loader(dataset) for dataset in self.test_datasets]


def thermal_to_01(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if torch.is_floating_point(x):
        x_min = float(x.detach().amin().cpu())
        x_max = float(x.detach().amax().cpu())
        if x_min < -0.05 or x_max > 1.05:
            return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
        return torch.clamp(x, 0.0, 1.0)
    return x.float().div(255.0).clamp(0.0, 1.0)


def build_data_module(config: dict[str, Any]) -> PaperWebDataModule:
    datasets = dict(config.get("datasets", {}))
    training = dict(config.get("training", {}))
    dataset_names = SimpleNamespace(
        train_datasets=list(datasets.get("train_datasets", ["AVIID"])),
        val_datasets=list(datasets.get("val_datasets", ["AVIID"])),
        test_datasets=list(datasets.get("test_datasets", datasets.get("val_datasets", ["AVIID"]))),
    )
    train_size = training.get("train_image_size", [256, 256])
    train_cfg_training = SimpleNamespace(num_samples_per_epoch=int(training.get("num_samples_per_epoch", 10000)))
    return PaperWebDataModule(
        datasets_folder=str(datasets.get("datasets_folder", "./datasets_preprocess")),
        train_batch_size=int(training.get("train_batch_size", 16)),
        test_batch_size=int(training.get("test_batch_size", 16)),
        train_image_size=tuple(train_size),
        num_workers=int(training.get("num_workers", 8)),
        dataset_names=dataset_names,
        train_cfg_training=train_cfg_training,
        mixed_precision=bool(training.get("mixed_precision", False)),
        image_norm=str(training.get("image_norm", "normal")),
        dataset_config_root=datasets.get("dataset_config_root"),
    )
