from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


DEFAULT_QUERY = "datasets_raw/DroneVehicle/DayDrone/testA/000076.jpg"
DEFAULT_SEARCH_DIR = (
    "logs/psl_flow_ablation/DroneVehicle_day/full/test_samples/"
    "epoch_0001/DroneVehicle_day"
)


@dataclass
class MatchResult:
    path: str
    width: int
    height: int
    mse: float
    mae: float
    psnr: float
    cosine: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find which logged `input_bXXXXX_sXXX.png` best matches a given raw RGB image. "
            "The comparison is content-based and can run on GPU."
        )
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help=f"Query image path. Default: {DEFAULT_QUERY}")
    parser.add_argument(
        "--search-dir",
        default=DEFAULT_SEARCH_DIR,
        help=f"Directory that contains logged input images. Default: {DEFAULT_SEARCH_DIR}",
    )
    parser.add_argument(
        "--pattern",
        default="input_b*.png",
        help="Glob pattern under --search-dir. Default: input_b*.png",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search under --search-dir instead of only the top level.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0 ... Default: auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of candidate images compared per batch. Default: 16",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many top matches to print. Default: 10")
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional JSON file path to save the ranked results.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def iter_candidate_paths(search_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if not search_dir.is_dir():
        raise FileNotFoundError(f"Search directory not found: {search_dir}")
    candidates = list(search_dir.rglob(pattern) if recursive else search_dir.glob(pattern))
    candidates = sorted(path for path in candidates if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No files matched pattern `{pattern}` under `{search_dir}`. "
            "Check the log path or add --recursive."
        )
    return candidates


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def get_image_hw(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def resize_tensor(image_chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    image_nchw = image_chw.unsqueeze(0)
    try:
        resized = F.interpolate(
            image_nchw,
            size=size_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        resized = F.interpolate(
            image_nchw,
            size=size_hw,
            mode="bilinear",
            align_corners=False,
        )
    return resized.squeeze(0)


def group_candidates_by_size(candidate_paths: Sequence[Path]) -> Dict[Tuple[int, int], List[Path]]:
    groups: Dict[Tuple[int, int], List[Path]] = {}
    for path in candidate_paths:
        size_hw = get_image_hw(path)
        groups.setdefault(size_hw, []).append(path)
    return groups


def batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@torch.no_grad()
def compare_group(
    query_rgb: torch.Tensor,
    candidate_paths: Sequence[Path],
    size_hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> List[MatchResult]:
    query_resized = resize_tensor(query_rgb, size_hw).unsqueeze(0).to(device=device, dtype=torch.float32)
    query_flat = query_resized.flatten(1)
    results: List[MatchResult] = []

    for batch_paths in batched(candidate_paths, batch_size):
        batch = torch.stack([load_rgb_tensor(path) for path in batch_paths], dim=0).to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        )
        diff = batch - query_resized
        mse = diff.square().mean(dim=(1, 2, 3))
        mae = diff.abs().mean(dim=(1, 2, 3))
        cosine = F.cosine_similarity(batch.flatten(1), query_flat.expand(batch.shape[0], -1), dim=1)
        psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))

        mse_cpu = mse.detach().cpu().tolist()
        mae_cpu = mae.detach().cpu().tolist()
        cosine_cpu = cosine.detach().cpu().tolist()
        psnr_cpu = psnr.detach().cpu().tolist()

        for idx, path in enumerate(batch_paths):
            results.append(
                MatchResult(
                    path=str(path.resolve()),
                    width=size_hw[1],
                    height=size_hw[0],
                    mse=float(mse_cpu[idx]),
                    mae=float(mae_cpu[idx]),
                    psnr=float(psnr_cpu[idx]),
                    cosine=float(cosine_cpu[idx]),
                )
            )

    return results


def rank_matches(results: Sequence[MatchResult]) -> List[MatchResult]:
    return sorted(results, key=lambda item: (item.mse, item.mae, -item.cosine, item.path))


def summarize_best(results: Sequence[MatchResult]) -> str:
    best = results[0]
    second = results[1] if len(results) > 1 else None
    margin = math.inf if second is None else (second.mse - best.mse)
    return (
        f"best={best.path} | mse={best.mse:.8f} | mae={best.mae:.8f} | "
        f"psnr={best.psnr:.4f} | cosine={best.cosine:.8f} | mse_margin={margin:.8f}"
    )


def save_results_json(path: Path, ranked_results: Sequence[MatchResult], args: argparse.Namespace, device: torch.device) -> None:
    payload = {
        "query": str(Path(args.query).resolve()),
        "search_dir": str(Path(args.search_dir).resolve()),
        "pattern": args.pattern,
        "recursive": bool(args.recursive),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "results": [asdict(result) for result in ranked_results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    query_path = Path(args.query)
    search_dir = Path(args.search_dir)

    if not query_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    candidate_paths = iter_candidate_paths(search_dir, args.pattern, args.recursive)
    size_groups = group_candidates_by_size(candidate_paths)
    query_rgb = load_rgb_tensor(query_path)

    print(f"[INFO] query      : {query_path.resolve()}")
    print(f"[INFO] search_dir : {search_dir.resolve()}")
    print(f"[INFO] candidates : {len(candidate_paths)}")
    print(f"[INFO] size_groups : {len(size_groups)}")
    print(f"[INFO] device     : {device}")

    all_results: List[MatchResult] = []
    for size_hw, paths in sorted(size_groups.items()):
        print(f"[INFO] compare    : {len(paths)} images at {size_hw[1]}x{size_hw[0]}")
        all_results.extend(
            compare_group(
                query_rgb=query_rgb,
                candidate_paths=paths,
                size_hw=size_hw,
                batch_size=args.batch_size,
                device=device,
            )
        )

    ranked_results = rank_matches(all_results)
    print("[INFO] summary    :", summarize_best(ranked_results))
    print("")
    print(f"Top {min(args.top_k, len(ranked_results))} matches:")
    for rank, result in enumerate(ranked_results[: args.top_k], start=1):
        print(
            f"{rank:02d}. {result.path} | "
            f"mse={result.mse:.8f} | mae={result.mae:.8f} | "
            f"psnr={result.psnr:.4f} | cosine={result.cosine:.8f}"
        )

    if args.save_json:
        save_path = Path(args.save_json)
        save_results_json(save_path, ranked_results, args, device)
        print("")
        print(f"[INFO] saved json: {save_path.resolve()}")


if __name__ == "__main__":
    main()
