from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _is_tensor_state_dict(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(isinstance(key, str) for key in obj.keys())
        and all(torch.is_tensor(value) for value in obj.values())
    )


def _extract_state_dict_from_checkpoint(state_obj: object) -> Tuple[Dict[str, torch.Tensor], str]:
    if not isinstance(state_obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_obj)}")

    candidates = (
        "state_dict",
        "student",
        "model",
        "model_state_dict",
        "module",
    )
    for key in candidates:
        value = state_obj.get(key, None)
        if _is_tensor_state_dict(value):
            return value, key

    if _is_tensor_state_dict(state_obj):
        return state_obj, "root"

    for key, value in state_obj.items():
        if _is_tensor_state_dict(value):
            return value, key

    preview_keys = list(state_obj.keys())[:10]
    raise ValueError(
        "Checkpoint does not contain a tensor state_dict. "
        f"Top-level keys preview: {preview_keys}"
    )


def load_module_checkpoint(
    module: nn.Module,
    ckpt_path: str,
    strict: bool = False,
    strip_prefixes: Tuple[str, ...] = ("module.", "model."),
) -> Dict[str, object]:
    state_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict, state_source = _extract_state_dict_from_checkpoint(state_obj)

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in strip_prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value

    missing, unexpected = module.load_state_dict(cleaned, strict=strict)
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "num_state_tensors": len(cleaned),
        "state_source": state_source,
    }


def sample_perturbed_latent(
    z: torch.Tensor,
    sigma_min: float = 0.0,
    sigma_max: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sigma_max < sigma_min:
        raise ValueError(f"Invalid sigma range: [{sigma_min}, {sigma_max}]")
    sigma = torch.empty(
        z.shape[0], 1, 1, 1, device=z.device, dtype=z.dtype
    ).uniform_(float(sigma_min), float(sigma_max))
    return z + sigma * torch.randn_like(z), sigma


def _norm_groups(channels: int, max_groups: int = 16) -> int:
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _l2n(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_feat_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch in cosine_feat_loss: {pred.shape} vs {tgt.shape}")
    b, c, _, _ = pred.shape
    p = _l2n(pred.view(b, c, -1), dim=1, eps=eps)
    t = _l2n(tgt.view(b, c, -1), dim=1, eps=eps)
    return (1.0 - (p * t).sum(dim=1)).mean()


def grad_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch in grad_loss: {pred.shape} vs {tgt.shape}")
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = tgt[..., :, 1:] - tgt[..., :, :-1]
    tdy = tgt[..., 1:, :] - tgt[..., :-1, :]
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def grad_align_loss(
    z: torch.Tensor,
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    proj: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not z.requires_grad:
        raise RuntimeError("grad_align_loss requires `z.requires_grad=True`.")
    if feat_s.shape != feat_t.shape:
        raise ValueError(f"Shape mismatch in grad_align_loss: {feat_s.shape} vs {feat_t.shape}")

    if proj is None:
        proj = torch.randn_like(feat_s)

    s = (feat_s * proj).sum()
    t = (feat_t * proj).sum()

    g_s = torch.autograd.grad(s, z, retain_graph=True, create_graph=True)[0]
    g_t = torch.autograd.grad(t, z, retain_graph=True, create_graph=False)[0]

    gs = g_s.view(g_s.shape[0], -1)
    gt = g_t.view(g_t.shape[0], -1)
    gs = gs / (gs.norm(dim=1, keepdim=True) + eps)
    gt = gt / (gt.norm(dim=1, keepdim=True) + eps)
    return (1.0 - (gs * gt).sum(dim=1)).mean()


class PooledGlobalAttn(nn.Module):
    def __init__(self, ch: int, heads: int = 4, head_dim: int = 32, pool: int = 2):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.pool = int(pool)
        inner = self.heads * self.head_dim
        self.q = nn.Conv2d(ch, inner, 1, bias=False)
        self.k = nn.Conv2d(ch, inner, 1, bias=False)
        self.v = nn.Conv2d(ch, inner, 1, bias=False)
        self.o = nn.Conv2d(inner, ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.q(x)
        xp = F.avg_pool2d(x, self.pool, self.pool) if self.pool > 1 else x
        k = self.k(xp)
        v = self.v(xp)

        inner = self.heads * self.head_dim
        q = q.view(b, self.heads, self.head_dim, h * w).transpose(2, 3)
        hp, wp = xp.shape[-2:]
        k = k.view(b, self.heads, self.head_dim, hp * wp)
        v = v.view(b, self.heads, self.head_dim, hp * wp).transpose(2, 3)

        scale = self.head_dim ** -0.5
        attn = torch.softmax((q @ (k * scale)), dim=-1)
        out = attn @ v
        out = out.transpose(2, 3).contiguous().view(b, inner, h, w)
        return self.o(out)


class ResBlock(nn.Module):
    def __init__(self, ch: int, drop: float = 0.0):
        super().__init__()
        groups = _norm_groups(ch)
        self.n1 = nn.GroupNorm(groups, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.drop = nn.Dropout(float(drop)) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.drop(h)
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class ResAttnBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        use_attn: bool = True,
        heads: int = 4,
        head_dim: int = 32,
        pool: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        self.res = ResBlock(ch, drop)
        self.use_attn = bool(use_attn)
        if self.use_attn:
            self.n = nn.GroupNorm(_norm_groups(ch), ch)
            self.attn = PooledGlobalAttn(ch, heads, head_dim, pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        if self.use_attn:
            x = x + self.attn(F.silu(self.n(x)))
        return x


@dataclass
class SurCfg:
    z_ch: int = 4
    width: int = 192
    depth: int = 8
    attn_every: int = 2
    heads: int = 4
    head_dim: int = 32
    pool: int = 2
    drop: float = 0.0
    feat_dim: int = 64
    vnums: int = 4


class LatentSurrogate(nn.Module):
    def __init__(self, cfg: SurCfg):
        super().__init__()
        self.cfg = cfg
        width = int(cfg.width)
        self.stem = nn.Conv2d(int(cfg.z_ch), width, 3, padding=1)
        blocks = []
        for index in range(int(cfg.depth)):
            blocks.append(
                ResAttnBlock(
                    width,
                    use_attn=(index % int(cfg.attn_every) == 0),
                    heads=int(cfg.heads),
                    head_dim=int(cfg.head_dim),
                    pool=int(cfg.pool),
                    drop=float(cfg.drop),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        groups = _norm_groups(width)
        self.dec_head = nn.Sequential(nn.GroupNorm(groups, width), nn.SiLU(), nn.Conv2d(width, int(cfg.feat_dim), 1))
        self.deep_head = nn.Sequential(nn.GroupNorm(groups, width), nn.SiLU(), nn.Conv2d(width, int(cfg.feat_dim), 1))
        self.phys_head = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, 2 + int(cfg.vnums), 1),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.stem(z)
        h = self.blocks(h)
        return {
            "feat_dec": self.dec_head(h),
            "feat_deep": self.deep_head(h),
            "phys_logits": self.phys_head(h),
        }


class FixedRand1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, seed: int = 1234):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        w = torch.randn(int(out_ch), int(in_ch), 1, 1, generator=generator) / (float(in_ch) ** 0.5)
        self.register_buffer("w", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w if self.w.dtype == x.dtype else self.w.to(dtype=x.dtype)
        return F.conv2d(x, w)


class TeacherComposite(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        tevnet: nn.Module,
        feat_dim: int = 64,
        vnums: int = 4,
        seed: int = 1234,
    ):
        super().__init__()
        self.decoder = decoder
        self.tevnet = tevnet
        self.feat_dim = int(feat_dim)
        self.vnums = int(vnums)
        self.seed = int(seed)
        self.proj_dec: Optional[FixedRand1x1] = None
        self.proj_deep: Optional[FixedRand1x1] = None

    def _expected_teacher_in_channels(self) -> int:
        in_ch = getattr(self.tevnet, "in_channels", None)
        if isinstance(in_ch, int) and in_ch > 0:
            return int(in_ch)
        backbone = getattr(self.tevnet, "backbone", None)
        if backbone is not None:
            net = getattr(backbone, "net", None)
            if net is not None and hasattr(net, "encoder"):
                encoder = net.encoder
                if hasattr(encoder, "conv1") and hasattr(encoder.conv1, "in_channels"):
                    return int(encoder.conv1.in_channels)
        return 3

    def _prepare_teacher_input(self, y: torch.Tensor) -> torch.Tensor:
        # Keep behavior aligned with the TeR-B auxiliary path in PSL-Flow:
        # decode output in [-1,1] -> [0,1], then adapt channels for TeR-B Net.
        y = torch.clamp(y * 0.5 + 0.5, 0, 1)
        expected_channels = self._expected_teacher_in_channels()
        if y.shape[1] == expected_channels:
            return y
        if y.shape[1] == 1 and expected_channels == 3:
            return y.repeat(1, 3, 1, 1)
        if y.shape[1] == 3 and expected_channels == 1:
            return y.mean(dim=1, keepdim=True)
        raise RuntimeError(
            f"TeacherComposite channel mismatch: decoded={y.shape[1]}, "
            f"TeVNet expects={expected_channels}."
        )

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if hasattr(self.decoder, "decode"):
            decoded = self.decoder.decode(z)
        else:
            decoded = self.decoder(z)
        if isinstance(decoded, (tuple, list)):
            decoded = decoded[0]
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        if not isinstance(decoded, torch.Tensor):
            raise RuntimeError(f"Unsupported decoder output type: {type(decoded)}")
        return decoded

    def _extract_raw(self, y: torch.Tensor):
        required = ("backbone", "mrm", "t_head", "v_head")
        for attr in required:
            if not hasattr(self.tevnet, attr):
                raise AttributeError(
                    f"TeacherComposite requires TeVNetTherNet-like teacher with `{attr}`."
                )
        logits, dec, feats = self.tevnet.backbone(y)
        deep = feats[-1]
        e, _ = self.tevnet.mrm(dec)
        t = F.relu(self.tevnet.t_head(dec))
        v_logits = self.tevnet.v_head(dec) if self.vnums > 0 else logits[:, 2:]
        return dec, deep, e, t, v_logits

    def forward(self, z: torch.Tensor, requires_grad_teacher: bool = False) -> Dict[str, torch.Tensor]:
        hw = (z.shape[-2], z.shape[-1])
        context = nullcontext() if requires_grad_teacher else torch.no_grad()
        with context:
            y = self._decode(z)
            y = self._prepare_teacher_input(y)
            dec, deep, e, t, v_logits = self._extract_raw(y)

            if self.proj_dec is None:
                self.proj_dec = FixedRand1x1(dec.shape[1], self.feat_dim, seed=self.seed).to(dec.device)
            if self.proj_deep is None:
                self.proj_deep = FixedRand1x1(deep.shape[1], self.feat_dim, seed=self.seed + 1).to(deep.device)

            feat_dec = self.proj_dec(dec)
            feat_deep = self.proj_deep(deep)
            feat_dec = F.interpolate(feat_dec, size=hw, mode="bilinear", align_corners=False)
            feat_deep = F.interpolate(feat_deep, size=hw, mode="bilinear", align_corners=False)

            e = F.interpolate(e.clamp(0, 1), size=hw, mode="bilinear", align_corners=False)
            logt = torch.log1p(t.clamp(min=0))
            logt = F.interpolate(logt, size=hw, mode="bilinear", align_corners=False)
            if self.vnums > 0:
                v_logits = F.interpolate(v_logits, size=hw, mode="bilinear", align_corners=False)
                phys = torch.cat([e, logt, v_logits], dim=1)
            else:
                phys = torch.cat([e, logt], dim=1)

        return {"feat_dec": feat_dec, "feat_deep": feat_deep, "phys": phys}


@dataclass
class KDcfg:
    w_l1: float = 1.0
    w_cos: float = 0.5
    w_grad: float = 0.5
    w_tv_t: float = 0.05
    w_entropy_v: float = 0.01
    w_grad_align: float = 0.1
    grad_align_prob: float = 0.25


class KDLoss(nn.Module):
    def __init__(self, cfg: KDcfg, vnums: int = 4):
        super().__init__()
        self.cfg = cfg
        self.vnums = int(vnums)

    def _match(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        loss = float(self.cfg.w_l1) * F.smooth_l1_loss(s, t)
        loss = loss + float(self.cfg.w_cos) * cosine_feat_loss(s, t)
        loss = loss + float(self.cfg.w_grad) * grad_loss(s, t)
        return loss

    def forward(
        self,
        z: torch.Tensor,
        stu: Dict[str, torch.Tensor],
        tea_detached: Dict[str, torch.Tensor],
        tea_phys_for_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=z.device, dtype=stu["feat_dec"].dtype)
        loss = loss + self._match(stu["feat_dec"], tea_detached["feat_dec"])
        loss = loss + self._match(stu["feat_deep"], tea_detached["feat_deep"])
        loss = loss + self._match(stu["phys_logits"], tea_detached["phys"])

        logt = stu["phys_logits"][:, 1:2]
        if float(self.cfg.w_tv_t) > 0:
            loss = loss + float(self.cfg.w_tv_t) * tv_loss(logt)

        if self.vnums > 0 and float(self.cfg.w_entropy_v) > 0:
            v_logits = stu["phys_logits"][:, 2 : 2 + self.vnums]
            v = torch.softmax(v_logits, dim=1)
            ent = -(v * torch.log(v + 1e-8)).sum(dim=1).mean()
            loss = loss - float(self.cfg.w_entropy_v) * ent

        run_grad_align = (
            tea_phys_for_grad is not None
            and float(self.cfg.w_grad_align) > 0
            and z.requires_grad
            and bool(torch.rand((), device=z.device) < float(self.cfg.grad_align_prob))
        )
        if run_grad_align:
            loss = loss + float(self.cfg.w_grad_align) * grad_align_loss(
                z, stu["phys_logits"], tea_phys_for_grad
            )

        return loss


__all__ = [
    "SurCfg",
    "LatentSurrogate",
    "KDcfg",
    "KDLoss",
    "TeacherComposite",
    "sample_perturbed_latent",
    "load_module_checkpoint",
    "grad_align_loss",
]
