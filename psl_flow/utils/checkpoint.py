from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn


def _extract_state(payload: object) -> dict:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise RuntimeError("Checkpoint does not contain a recognizable state dict.")


def _strip_prefix(key: str, prefixes: Iterable[str]) -> str:
    out = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
    return out


def load_state_dict_flexible(
    module: nn.Module,
    ckpt_path: str | Path,
    *,
    strict: bool = False,
    strip_prefixes: Iterable[str] = ("model.", "module.", "net.", "teacher.", "psl_vae."),
) -> dict:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = _extract_state(payload)
    own_state = module.state_dict()
    cleaned = {}
    ignored = []
    for key, value in state.items():
        new_key = _strip_prefix(str(key), strip_prefixes)
        if new_key in own_state and tuple(own_state[new_key].shape) == tuple(value.shape):
            cleaned[new_key] = value
        else:
            ignored.append(str(key))
    result = module.load_state_dict(cleaned, strict=strict)
    return {
        "path": str(ckpt_path),
        "loaded": len(cleaned),
        "ignored": ignored,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }

