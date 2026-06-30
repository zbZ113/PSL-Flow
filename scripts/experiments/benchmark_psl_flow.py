from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Print a compact PSL-Flow benchmark summary")
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    path = Path(args.summary_csv)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    print("route,stage,step,elapsed_hms,peak_gpu_gib,checkpoint")
    for row in rows:
        print(
            ",".join(
                [
                    row.get("route", ""),
                    row.get("stage", ""),
                    row.get("step", ""),
                    row.get("elapsed_hms", ""),
                    row.get("peak_gpu_gib", ""),
                    row.get("checkpoint", ""),
                ]
            )
        )


if __name__ == "__main__":
    main()
