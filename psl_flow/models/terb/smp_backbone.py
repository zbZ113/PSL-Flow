from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch
from torch import nn


SMP_VERSION = "0.3.4"


def maybe_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return None if x in ("None", "none", "null", "NULL", "") else x


def require_smp(import_context: str = "psl_flow.models.terb"):
    try:
        return importlib.import_module("segmentation_models_pytorch")
    except ImportError as exc:
        raise ImportError(
            f"{import_context} requires segmentation-models-pytorch=={SMP_VERSION}. "
            "Install the release environment before training TeR-B."
        ) from exc


def build_smp_model(
    *,
    smp_model: str,
    encoder_name: str,
    encoder_weights: Optional[str],
    in_channels: int,
    out_channels: int,
    import_context: str,
) -> nn.Module:
    smp = require_smp(import_context)
    if not hasattr(smp, smp_model):
        raise AttributeError(f"segmentation_models_pytorch has no model `{smp_model}`.")
    return getattr(smp, smp_model)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


class SMPWrapper(nn.Module):
    def __init__(
        self,
        *,
        smp_model: str,
        encoder_name: str,
        encoder_weights: Optional[str],
        in_channels: int,
        out_channels: int,
        import_context: str = "psl_flow.models.terb",
    ):
        super().__init__()
        self.net = build_smp_model(
            smp_model=smp_model,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            out_channels=out_channels,
            import_context=import_context,
        )
        for attr in ("encoder", "decoder", "segmentation_head"):
            if not hasattr(self.net, attr):
                raise AttributeError(f"smp.{smp_model} does not expose `{attr}`.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        feats = self.net.encoder(x)
        dec = self.net.decoder(*feats)
        logits = self.net.segmentation_head(dec)
        return logits, dec, feats
