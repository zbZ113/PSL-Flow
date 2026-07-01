from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_indices_are_compact_public_ids():
    expected = {
        "AVIID.yml": "dataset_index: 0",
        "CART.yml": "dataset_index: 1",
        "DroneVehicle_day.yml": "dataset_index: 2",
        "DroneVehicle_night.yml": "dataset_index: 3",
    }
    dataset_dir = ROOT / "psl_flow" / "configs" / "experiments" / "datasets"
    for filename, first_line in expected.items():
        assert (dataset_dir / filename).read_text(encoding="utf-8").splitlines()[0] == first_line
