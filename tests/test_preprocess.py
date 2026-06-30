from __future__ import annotations

import json
import tarfile

import pytest


Image = pytest.importorskip("PIL.Image")
pytest.importorskip("PIL.ImageOps")
format_dataset = pytest.importorskip("tools.preprocess.format_dataset")


def _write_rgb(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (16, 32, 48)).save(path)


def _write_gray(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), 96).save(path)


def test_aviid_raw_folder_is_formatted_as_webdataset(tmp_path):
    raw_root = tmp_path / "datasets_raw"
    output_root = tmp_path / "datasets_preprocess"

    for split in ("train", "test"):
        _write_rgb(raw_root / "AVIID" / split / "vis" / "0001.png")
        _write_gray(raw_root / "AVIID" / split / "ir" / "0001.png")

    format_dataset._process_one("AVIID", raw_root, output_root, shard_size=1000, overwrite=True)

    train_dir = output_root / "AVIID" / "train"
    with (train_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata == {"num_samples": 1}

    with tarfile.open(train_dir / "dataset-00000.tar", "r") as archive:
        names = sorted(archive.getnames())
    assert names == ["0001.color.png", "0001.thermal.png"]


def test_cart_preprocess_creates_validation_from_test_when_missing(tmp_path):
    raw_root = tmp_path / "datasets_raw"
    output_root = tmp_path / "datasets_preprocess"

    for split in ("train", "test"):
        _write_rgb(raw_root / "CART" / split / "RGB" / "0001_rgb-0.png")
        _write_gray(raw_root / "CART" / split / "Thermal" / "0001_thermal-0.png")

    format_dataset._process_one("CART", raw_root, output_root, shard_size=1000, overwrite=True)

    for split in ("train", "val", "test"):
        assert (output_root / "CART" / split / "dataset-00000.tar").is_file()
        with (output_root / "CART" / split / "metadata.json").open("r", encoding="utf-8") as handle:
            assert json.load(handle) == {"num_samples": 1}
