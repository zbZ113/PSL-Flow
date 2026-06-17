"""TeV decomposition network (IR -> e/T/V).

This repo originally trained TeVNet with a *single* reconstruction constraint.
That makes the decomposition under-determined: e/T/V can absorb imaging effects
(atmosphere, sensor blur) and boundary artifacts.

This file keeps the original TeVNet (baseline) and adds a physics-regularized
variant (TeVNetTherNet) inspired by TherNet's physical-property modules.

Key ideas implemented (still TeV, not TeR):
1) **Convex environment weights**: V is normalized with softmax so env becomes a
   convex combination of pooled blocks (keeps PID's V·Ŝ form, but reduces DOF).
2) **Imaging heads**: predict atmospheric transmittance tau and path radiance A.
   These are used in the training loss as a forward imaging model.
3) **Material prototypes for emissivity**: emissivity is expressed as
   (prototype prior + residual) in logit space.

The training-side losses are implemented in `TeVNet/utils.py`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.physics.smp_backbone import SMPWrapper, build_smp_model, infer_decoder_channels, maybe_none


class ThermalInertiaBlur(nn.Module):
    """A lightweight, energy-preserving blur (camera/sensor inertia proxy).

    Implemented as a learnable kernel normalized with softmax.
    This is stable and differentiable, and does not require per-sample kernels.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        # Start near a delta kernel (no blur).
        init = torch.zeros(kernel_size * kernel_size)
        init[kernel_size * kernel_size // 2] = 3.0
        self.kernel_logits = nn.Parameter(init.view(1, 1, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W)
        b, c, h, w = x.shape
        assert c == 1, "ThermalInertiaBlur expects single-channel input"
        k = F.softmax(self.kernel_logits.view(1, -1), dim=1).view(1, 1, self.kernel_size, self.kernel_size)
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, k, bias=None, stride=1, padding=0)


class AtmosphericHead(nn.Module):
    """Predict per-pixel atmospheric transmittance tau and path radiance A."""

    def __init__(self, in_ch: int, tau_min: float = 0.05):
        super().__init__()
        self.tau_min = float(tau_min)
        mid = max(32, in_ch // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.SiLU(),
        )
        self.tau_head = nn.Conv2d(mid, 1, 1)
        self.a_head = nn.Conv2d(mid, 1, 1)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = out_hw
        y = self.net(feat)
        tau = torch.sigmoid(self.tau_head(y))
        a = torch.sigmoid(self.a_head(y))
        tau = F.interpolate(tau, size=(h, w), mode="bilinear", align_corners=False)
        a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
        # Keep tau away from 0 to avoid division explosion.
        tau = tau * (1.0 - self.tau_min) + self.tau_min
        return tau, a


class MaterialPrototypeEmissivity(nn.Module):
    """Material-inspired emissivity head.

    We do not assume material labels; instead we learn K prototypes. For each
    pixel, we predict an assignment over K, get a prototype embedding, and map
    it to an emissivity logit prior. A residual branch refines it.
    """

    def __init__(self, feat_ch: int, num_prototypes: int = 16, proto_dim: int = 32):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.proto_dim = int(proto_dim)

        self.assign_head = nn.Conv2d(feat_ch, self.num_prototypes, 1)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim) * 0.02)
        self.prior_head = nn.Sequential(
            nn.Conv2d(self.proto_dim, self.proto_dim, 1),
            nn.SiLU(),
            nn.Conv2d(self.proto_dim, 1, 1),
        )
        self.res_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch // 2, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # feat: (B,C,H,W)
        logits = self.assign_head(feat)
        p = torch.softmax(logits, dim=1)

        # Compute per-pixel prototype embedding: E = sum_k p_k * proto_k
        # p: (B,K,H,W) -> (B,H,W,K)
        p_hwk = p.permute(0, 2, 3, 1).contiguous()
        e_hw_d = torch.matmul(p_hwk, self.prototypes)  # (B,H,W,D)
        e_dhw = e_hw_d.permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)

        e_prior_logit = self.prior_head(e_dhw)
        e_delta_logit = self.res_head(feat)
        e = torch.sigmoid(e_prior_logit + e_delta_logit)

        aux = {
            "mrm_p": p,
            "mrm_e_prior_logit": e_prior_logit,
            "mrm_e_delta_logit": e_delta_logit,
        }
        return e, aux


class TeVNet(nn.Module):
    """Baseline TeVNet (original)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 6, args=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        smp_model = getattr(args, "smp_model", "PAN")
        smp_encoder = getattr(args, "smp_encoder", "resnet50")
        smp_encoder_weights = maybe_none(getattr(args, "smp_encoder_weights", "imagenet"))

        self.tevnet = build_smp_model(
            smp_model=smp_model,
            encoder_name=smp_encoder,
            encoder_weights=smp_encoder_weights,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            import_context="TeVNet",
        )

        self.v_softmax = bool(getattr(args, "v_softmax", False))
        self.vnums = int(getattr(args, "vnums", max(0, self.out_channels - 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.tevnet(x)
        # e in [0,1]
        e = torch.sigmoid(preds[:, 0:1])
        # T >= 0
        t = F.relu(preds[:, 1:2])
        rest = preds[:, 2:]
        if self.v_softmax and rest.shape[1] == self.vnums and self.vnums > 0:
            rest = torch.softmax(rest, dim=1)
        return torch.cat([e, t, rest], dim=1)


class TeVNetTherNet(nn.Module):
    """Physics-regularized TeV decomposition network.

    Return:
        - out: (B, 2+vnums, H, W) as [e, T, V...]
        - aux (optional): dict with tau/A, material priors, etc.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 6, args=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.vnums = int(getattr(args, "vnums", max(0, self.out_channels - 2)))

        smp_model = getattr(args, "smp_model", "PAN")
        smp_encoder = getattr(args, "smp_encoder", "resnet50")
        smp_encoder_weights = maybe_none(getattr(args, "smp_encoder_weights", "imagenet"))

        self.backbone = SMPWrapper(
            smp_model=smp_model,
            encoder_name=smp_encoder,
            encoder_weights=smp_encoder_weights,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            import_context="TeVNetTherNet",
        )

        # Heads operating on decoder features.
        # SMP API differs across versions; infer decoder channels robustly.
        dec_ch = infer_decoder_channels(self.backbone, self.in_channels)

        # Material emissivity head (MRM-inspired).
        self.mrm = MaterialPrototypeEmissivity(
            feat_ch=int(dec_ch),
            num_prototypes=int(getattr(args, "mrm_k", 16)),
            proto_dim=int(getattr(args, "mrm_d", 32)),
        )

        # Temperature head.
        self.t_head = nn.Sequential(
            nn.Conv2d(int(dec_ch), int(dec_ch // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(dec_ch // 2), 1, 1),
        )

        # Environment weight head (V) with convex constraint.
        self.v_head = nn.Sequential(
            nn.Conv2d(int(dec_ch), int(dec_ch // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(int(dec_ch // 2), self.vnums, 1),
        )

        # Atmospheric head on the deepest encoder feature.
        enc_out_ch = getattr(self.backbone.net.encoder, "out_channels", None)
        self.tau_min = float(getattr(args, "tau_min", 0.05))
        if enc_out_ch is None:
            self.atm = None
        else:
            self.atm = AtmosphericHead(in_ch=int(enc_out_ch[-1]), tau_min=self.tau_min)

        self.tim = ThermalInertiaBlur(kernel_size=int(getattr(args, "tim_kernel", 5)))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        logits, dec, feats = self.backbone(x)
        b, _, h, w = logits.shape

        aux: Dict[str, torch.Tensor] = {}

        # e from material-prototype prior + residual
        e, mrm_aux = self.mrm(dec)
        aux.update(mrm_aux)

        # T head
        t = F.relu(self.t_head(dec))

        # V head -> convex weights
        if self.vnums > 0:
            v_logits = self.v_head(dec)
            v = torch.softmax(v_logits, dim=1)
        else:
            v = logits[:, 2:]  # empty

        if e.shape[-2:] != (h, w):
            e = F.interpolate(e, size=(h, w), mode="bilinear", align_corners=False)
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        if self.vnums > 0 and v.shape[-2:] != (h, w):
            v = F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False)

        # tau/A
        feat_deep = feats[-1]
        if self.atm is None:
            self.atm = AtmosphericHead(in_ch=int(feat_deep.shape[1]), tau_min=self.tau_min).to(feat_deep.device)
        tau, a = self.atm(feat_deep, out_hw=(h, w))
        aux["atm_tau"] = tau
        aux["atm_A"] = a

        # tim is used in loss; expose kernel for logging.
        aux["tim_kernel_logits"] = self.tim.kernel_logits

        out = torch.cat([e, t, v], dim=1)
        return (out, aux) if return_aux else out


def build_tevnet(args, in_channels: int = 3, out_channels: int = 6) -> nn.Module:
    """Factory to build TeVNet variant by args.arch."""
    arch = getattr(args, "arch", "baseline")
    if arch in ("thernet", "pp", "phys"):
        return TeVNetTherNet(in_channels=in_channels, out_channels=out_channels, args=args)
    return TeVNet(in_channels=in_channels, out_channels=out_channels, args=args)
