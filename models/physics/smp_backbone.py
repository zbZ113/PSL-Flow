from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


SMP_VERSION = "0.3.4"


def maybe_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return None if x in ("None", "none", "null", "NULL", "") else x


def require_smp(import_context: str = "models.physics"):
    try:
        return importlib.import_module("segmentation_models_pytorch")
    except ImportError as exc:
        raise ImportError(
            f"{import_context} requires `segmentation-models-pytorch=={SMP_VERSION}`. "
            "Install physics extras with `pip install -r requirements-physics.txt` "
            "or pin the same version in your environment."
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
        raise AttributeError(
            f"segmentation_models_pytorch has no model `{smp_model}`. "
            "Use a supported architecture such as Unet, FPN, or PAN."
        )
    return getattr(smp, smp_model)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def _first_conv_in_channels(module: nn.Module) -> Optional[int]:
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Conv2d):
            return int(sub_module.in_channels)
    return None


class SMPWrapper(nn.Module):
    """Wrap segmentation_models_pytorch models to expose decoder features."""

    def __init__(
        self,
        *,
        smp_model: str,
        encoder_name: str,
        encoder_weights: Optional[str],
        in_channels: int,
        out_channels: int,
        import_context: str = "models.physics",
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
                raise AttributeError(
                    f"smp.{smp_model} does not expose `{attr}`; "
                    "use a model like Unet/FPN/PAN that supports encoder/decoder."
                )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        feats = self.net.encoder(x)
        dec = self.net.decoder(*feats)
        logits = self.net.segmentation_head(dec)
        return logits, dec, feats


def infer_decoder_channels(backbone: "SMPWrapper", in_channels: int) -> int:
    seg_head = backbone.net.segmentation_head
    dec = backbone.net.decoder

    seg_in = getattr(seg_head, "in_channels", None)
    if isinstance(seg_in, int) and seg_in > 0:
        return int(seg_in)

    seg_conv_in = _first_conv_in_channels(seg_head)
    if seg_conv_in is not None:
        return seg_conv_in

    dec_out = getattr(dec, "out_channels", None)
    if isinstance(dec_out, int) and dec_out > 0:
        return int(dec_out)
    if isinstance(dec_out, (list, tuple)) and len(dec_out) > 0:
        return int(dec_out[-1])

    dec_channels = getattr(dec, "channels", None)
    if isinstance(dec_channels, (list, tuple)) and len(dec_channels) > 0:
        return int(dec_channels[-1])

    was_training = backbone.training
    try:
        backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_channels), 64, 64)
            _, dec_feat, _ = backbone(dummy)
        if dec_feat.dim() == 4:
            return int(dec_feat.shape[1])
    finally:
        if was_training:
            backbone.train()

    raise AttributeError(
        "Cannot infer decoder channels from SMP model. "
        "Tried segmentation_head.in_channels, first conv in segmentation_head, "
        "decoder.out_channels, decoder.channels, and dummy-forward fallback."
    )


__all__ = [
    "SMP_VERSION",
    "SMPWrapper",
    "build_smp_model",
    "infer_decoder_channels",
    "maybe_none",
    "require_smp",
]
