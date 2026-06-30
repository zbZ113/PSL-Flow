from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "datasets",
    "datasets_preprocess",
    "datasets_raw",
    "logs",
    "outputs",
}
EXCLUDED_NAMES = {"LICENSE"}


def _factor_markers() -> list[str]:
    upper = chr(65)
    return [
        "a_low" + "_range",
        "a_low" + "_head",
        upper + "8",
        '"' + upper + '"',
        "'" + upper + "'",
        "loss" + "_a",
        "w" + "_a",
        "normalize" + "_a",
        "denormalize" + "_a",
    ]


def _history_markers() -> list[str]:
    return [
        "generic" + "_sit",
        "TeV" + "Net",
        "phys" + "_vae" + "_r",
        "latent" + " surrogate",
        "physics" + " proxy",
        "FL" + "IR",
        "K" + "AIST",
        "pix" + "2pix",
        "cycle" + "gan",
        "vq" + "gan",
        "dc" + "ae",
        "write" + "_grsl" + "_docx.py",
    ]


def _public_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def test_public_tree_has_no_removed_factor_markers():
    markers = _factor_markers()
    hits: list[str] = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in markers:
            if marker in text:
                hits.append(f"{path.relative_to(ROOT)}: {marker}")
    assert hits == []


def test_public_tree_has_no_history_route_or_dataset_markers():
    markers = _history_markers()
    hits: list[str] = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lower_text = text.lower()
        for marker in markers:
            if marker.lower() in lower_text:
                hits.append(f"{path.relative_to(ROOT)}: {marker}")
    assert hits == []
