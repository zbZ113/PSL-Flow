from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate standalone colorbars for T, e, and R_env factor maps."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/factor_colorbars",
        help="Directory to save the generated colorbar images.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap name. Default matches current factor-map visualization.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Colorbar orientation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved image DPI.",
    )
    return parser


def _save_colorbar(
    out_path: Path,
    cmap_name: str,
    orientation: str,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = (4.2, 0.42) if orientation == "horizontal" else (0.42, 4.2)
    fig, ax = plt.subplots(figsize=fig_size)

    gradient = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    if orientation == "horizontal":
        image = np.tile(gradient[None, :], (24, 1))
    else:
        image = np.tile(gradient[:, None], (1, 24))

    ax.imshow(
        image,
        cmap=cmap_name,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)


def main() -> None:
    args = _make_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()

    for name in ("T", "e", "R_env"):
        _save_colorbar(
            out_path=output_dir / f"{name}_colorbar.png",
            cmap_name=args.cmap,
            orientation=args.orientation,
            dpi=int(args.dpi),
        )

    print(f"[INFO] Saved colorbars to: {output_dir}")


if __name__ == "__main__":
    main()
