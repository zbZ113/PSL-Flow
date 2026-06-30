from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


DATASET_CHOICES = ("AVIID", "CART", "DroneVehicle_day", "DroneVehicle_night")
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    sample_key: str
    color_path: Path
    thermal_path: Path


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS)


def _stem_index(folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _list_images(folder):
        if path.stem in index:
            raise RuntimeError(f"Duplicate sample stem `{path.stem}` in {folder}")
        index[path.stem] = path
    return index


def _normalized_cart_id(path: Path) -> str:
    return path.stem.replace("_eo-", "_").replace("_thermal-", "_").replace("_rgb-", "_")


def _cart_index(folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _list_images(folder):
        pair_id = _normalized_cart_id(path)
        if pair_id in index:
            raise RuntimeError(f"Duplicate CART pair id `{pair_id}` in {folder}")
        index[pair_id] = path
    return index


def _resolve_aviid_split(raw_root: Path, split: str) -> tuple[Path, Path]:
    dataset_root = raw_root / "AVIID"
    modern_color = dataset_root / split / "vis"
    modern_thermal = dataset_root / split / "ir"
    if modern_color.is_dir() and modern_thermal.is_dir():
        return modern_color, modern_thermal

    legacy_root = dataset_root / "AVIID"
    legacy_color = legacy_root / f"{split}A"
    legacy_thermal = legacy_root / f"{split}B"
    if legacy_color.is_dir() and legacy_thermal.is_dir():
        return legacy_color, legacy_thermal

    raise FileNotFoundError(
        f"Cannot find AVIID split `{split}`. Checked {modern_color} + {modern_thermal} "
        f"and {legacy_color} + {legacy_thermal}."
    )


def _collect_aviid(raw_root: Path, split: str) -> list[ImagePair]:
    color_dir, thermal_dir = _resolve_aviid_split(raw_root, split)
    color_index = _stem_index(color_dir)
    thermal_index = _stem_index(thermal_dir)
    common = sorted(set(color_index) & set(thermal_index))
    if not common:
        raise RuntimeError(f"No paired AVIID samples found for split `{split}`.")
    return [ImagePair(stem, color_index[stem], thermal_index[stem]) for stem in common]


def _collect_cart(raw_root: Path, split: str) -> list[ImagePair]:
    split_root = raw_root / "CART" / split
    color_index = _cart_index(split_root / "RGB")
    thermal_index = _cart_index(split_root / "Thermal")
    common = sorted(set(color_index) & set(thermal_index))
    if not common:
        raise RuntimeError(f"No paired CART samples found for split `{split}`.")
    return [ImagePair(stem, color_index[stem], thermal_index[stem]) for stem in common]


def _collect_dronevehicle(raw_root: Path, subset: str, split: str) -> list[ImagePair]:
    subset_dir = "DayDrone" if subset == "day" else "NightDrone"
    color_dir_name, thermal_dir_name = ("trainA", "trainB") if split == "train" else ("testA", "testB")
    subset_root = raw_root / "DroneVehicle" / subset_dir
    color_index = _stem_index(subset_root / color_dir_name)
    thermal_index = _stem_index(subset_root / thermal_dir_name)
    common = sorted(set(color_index) & set(thermal_index))
    if not common:
        raise RuntimeError(f"No paired DroneVehicle samples found for subset `{subset}` split `{split}`.")
    return [
        ImagePair(f"{subset}_{split}_{stem}", color_index[stem], thermal_index[stem])
        for stem in common
    ]


def _encode_png(path: Path, role: str) -> bytes:
    with Image.open(path) as image:
        if role == "color":
            out = image.convert("RGB")
        elif image.mode in {"RGB", "RGBA", "P"}:
            out = ImageOps.grayscale(image)
        else:
            out = image.copy()
        buffer = BytesIO()
        out.save(buffer, format="PNG")
        return buffer.getvalue()


def _write_member(sink: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    sink.addfile(info, BytesIO(payload))


def _clear_existing(output_dir: Path) -> None:
    for path in output_dir.glob("dataset-*.tar"):
        path.unlink()
    metadata = output_dir / "metadata.json"
    if metadata.exists():
        metadata.unlink()


def _write_shards(pairs: list[ImagePair], output_dir: Path, shard_size: int, overwrite: bool, desc: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("dataset-*.tar")) or (output_dir / "metadata.json").exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to rebuild it.")
        _clear_existing(output_dir)

    valid_count = 0
    num_shards = (len(pairs) + shard_size - 1) // shard_size
    with tqdm(total=len(pairs), desc=desc, unit="sample") as progress:
        for shard_index in range(num_shards):
            shard_pairs = pairs[shard_index * shard_size : (shard_index + 1) * shard_size]
            shard_path = output_dir / f"dataset-{shard_index:05d}.tar"
            with tarfile.open(shard_path, "w") as sink:
                for pair in shard_pairs:
                    color_png = _encode_png(pair.color_path, "color")
                    thermal_png = _encode_png(pair.thermal_path, "thermal")
                    _write_member(sink, f"{pair.sample_key}.color.png", color_png)
                    _write_member(sink, f"{pair.sample_key}.thermal.png", thermal_png)
                    valid_count += 1
                    progress.update(1)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"num_samples": valid_count}, handle)
    print(f"[OK] {desc}: wrote {valid_count} samples -> {output_dir}")


def _process_one(dataset: str, raw_root: Path, output_root: Path, shard_size: int, overwrite: bool) -> None:
    if dataset == "AVIID":
        for split in ("train", "test"):
            _write_shards(
                _collect_aviid(raw_root, split),
                output_root / "AVIID" / split,
                shard_size,
                overwrite,
                f"AVIID {split}",
            )
        return

    if dataset == "CART":
        cart_root = raw_root / "CART"
        for split in ("train", "test"):
            if (cart_root / split).is_dir():
                _write_shards(
                    _collect_cart(raw_root, split),
                    output_root / "CART" / split,
                    shard_size,
                    overwrite,
                    f"CART {split}",
                )
        validation_source = "val" if (cart_root / "val").is_dir() else "test"
        if (cart_root / validation_source).is_dir():
            _write_shards(
                _collect_cart(raw_root, validation_source),
                output_root / "CART" / "val",
                shard_size,
                overwrite,
                f"CART val",
            )
        return

    if dataset == "DroneVehicle_day":
        subset = "day"
    elif dataset == "DroneVehicle_night":
        subset = "night"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for split in ("train", "test"):
        _write_shards(
            _collect_dronevehicle(raw_root, subset, split),
            output_root / "DroneVehicle" / split / subset,
            shard_size,
            overwrite,
            f"DroneVehicle {subset} {split}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw visible-infrared datasets into WebDataset shards.")
    parser.add_argument(
        "--dataset",
        choices=(*DATASET_CHOICES, "all"),
        required=True,
        help="Dataset key to preprocess.",
    )
    parser.add_argument("--raw-root", default="datasets_raw", help="Root containing downloaded raw datasets.")
    parser.add_argument("--output-root", default="datasets_preprocess", help="Root for generated WebDataset shards.")
    parser.add_argument("--shard-size", type=int, default=1000, help="Samples per output tar shard.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing shards for the selected dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_size = max(1, int(args.shard_size))
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    selected = DATASET_CHOICES if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        _process_one(dataset, raw_root, output_root, shard_size, bool(args.overwrite))


if __name__ == "__main__":
    main()
