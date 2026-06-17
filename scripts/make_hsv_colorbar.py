from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone HSV bar for TeV-style HSV visualization. "
            "Default output is a single composite HSV strip."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/factor_colorbars",
        help="Directory to save the generated HSV colorbar.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Colorbar orientation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("composite", "hue", "saturation", "value"),
        default="composite",
        help=(
            "composite: vary H/S/V together; "
            "hue: vary H only with fixed S,V; "
            "saturation: vary S only with fixed H,V; "
            "value: vary V only with fixed H,S."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width for horizontal bar or thickness for vertical bar.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Output height for horizontal bar or width for vertical bar.",
    )
    parser.add_argument(
        "--hue",
        type=int,
        default=90,
        help="Fixed OpenCV hue value in [0, 179] when mode is saturation/value.",
    )
    parser.add_argument(
        "--saturation",
        type=int,
        default=255,
        help="Fixed saturation value in [0, 255] when mode is hue/value.",
    )
    parser.add_argument(
        "--value",
        type=int,
        default=255,
        help="Fixed value channel in [0, 255] when mode is hue/saturation.",
    )
    return parser


def _clip_uint8(x: int, upper: int) -> int:
    return int(np.clip(x, 0, upper))


def _make_hsv_bar(
    width: int,
    height: int,
    mode: str,
    hue: int,
    saturation: int,
    value: int,
    orientation: str,
) -> np.ndarray:
    length = int(width if orientation == "horizontal" else height)
    thickness = int(height if orientation == "horizontal" else width)

    ramp_h = np.linspace(0, 179, length, dtype=np.float32)
    ramp_sv = np.linspace(0, 255, length, dtype=np.float32)

    if mode == "composite":
        h_line = ramp_h
        s_line = ramp_sv
        v_line = ramp_sv
    elif mode == "hue":
        h_line = ramp_h
        s_line = np.full(length, _clip_uint8(saturation, 255), dtype=np.float32)
        v_line = np.full(length, _clip_uint8(value, 255), dtype=np.float32)
    elif mode == "saturation":
        h_line = np.full(length, _clip_uint8(hue, 179), dtype=np.float32)
        s_line = ramp_sv
        v_line = np.full(length, _clip_uint8(value, 255), dtype=np.float32)
    elif mode == "value":
        h_line = np.full(length, _clip_uint8(hue, 179), dtype=np.float32)
        s_line = np.full(length, _clip_uint8(saturation, 255), dtype=np.float32)
        v_line = ramp_sv
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    hsv_line = np.stack([h_line, s_line, v_line], axis=1).astype(np.uint8)
    if orientation == "horizontal":
        hsv = np.tile(hsv_line[None, :, :], (thickness, 1, 1))
    else:
        hsv = np.tile(hsv_line[:, None, :], (1, thickness, 1))

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def main() -> None:
    args = _make_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = _make_hsv_bar(
        width=int(args.width),
        height=int(args.height),
        mode=str(args.mode),
        hue=int(args.hue),
        saturation=int(args.saturation),
        value=int(args.value),
        orientation=str(args.orientation),
    )

    out_path = output_dir / "HSV_colorbar.png"
    Image.fromarray(rgb).save(out_path)
    print(f"[INFO] Saved HSV colorbar to: {out_path}")


if __name__ == "__main__":
    main()
