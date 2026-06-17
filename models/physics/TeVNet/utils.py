from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class TeVloss:
    """Original PID-style TeV reconstruction loss.

    The label `x` is the *same* IR image (self-supervised):
    we train e/T/V so that

        rec = e*T + (1-e)*env

    matches the grayscale IR.
    """

    def __init__(self, vnums: int = 4, loss_type: str = "MSE"):
        self.vnums = int(vnums)
        if loss_type == "L1":
            self.loss = nn.L1Loss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def loss_rec(self, preds: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)
        rec_img = self.rec(preds, x)
        return self.loss(rec_img, x_mean)

    def rec(self, preds: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=1)  # (B,H,W)
        e = self.rec_e(preds)
        t = self.rec_T(preds)
        env = self.rec_env(preds, x_mean)
        return e * t + (1 - e) * env

    def rec_e(self, preds: torch.Tensor) -> torch.Tensor:
        return preds[:, 0:1]

    def rec_T(self, preds: torch.Tensor) -> torch.Tensor:
        return preds[:, 1:2]

    def rec_env(self, preds: torch.Tensor, x_mean: torch.Tensor) -> torch.Tensor:
        """Environment term env = x_beta @ V.

        Args:
            preds: (B, 2+vnums, H, W)
            x_mean: (B, H, W) OR (B,1,H,W)
        """
        if x_mean.dim() == 4:
            x_map = x_mean[:, 0:1]
        elif x_mean.dim() == 3:
            x_map = x_mean.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported x_mean ndim={x_mean.dim()}, expected 3 or 4.")
        b, _, h, w = preds.shape
        v = preds[:, 2 : 2 + self.vnums]
        h_split_nums = int(math.sqrt(self.vnums))
        w_split_nums = self.vnums // h_split_nums
        assert h_split_nums * w_split_nums == self.vnums

        # Align scene map resolution to prediction map before pooling.
        if x_map.shape[-2:] != (h, w):
            x_map = F.interpolate(x_map, size=(h, w), mode="bilinear", align_corners=False)

        # x_beta: (B,1,vnums), robust to non-divisible shapes.
        x_beta = F.adaptive_avg_pool2d(x_map, (h_split_nums, w_split_nums)).reshape(b, 1, self.vnums)
        v_pred = v.reshape(b, self.vnums, h * w)
        env = torch.matmul(x_beta, v_pred).view(b, 1, h, w)
        return env


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Sobel edge magnitude in [0, +inf).

    Args:
        x: (B,1,H,W)
    """
    device = x.device
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _tv(x: torch.Tensor) -> torch.Tensor:
    """Isotropic TV (mean)."""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (dx.abs().mean() + dy.abs().mean())


class TeVTherNetLoss:
    """Physics-regularized TeV loss.

    Mechanism:
        x_mean  ≈  TIM( tau * S_scene + (1-tau) * A )
    and compute env using tau/A-corrected low-frequency prior.

    Regularizers:
        - emissivity piecewise smooth away from edges (MBM spirit)
        - prototype-residual regularization for emissivity (MRM spirit)
        - smooth assignments for prototypes
    """

    def __init__(
        self,
        vnums: int = 4,
        loss_type: str = "MSE",
        w_rec: float = 1.0,
        w_scene: float = 0.2,
        w_e_smooth: float = 0.05,
        w_mrm_res: float = 0.01,
        w_mrm_assign_tv: float = 0.01,
        edge_thresh: float = 0.08,
    ):
        self.base = TeVloss(vnums=vnums, loss_type=loss_type)
        self.vnums = int(vnums)
        self.w_rec = float(w_rec)
        self.w_scene = float(w_scene)
        self.w_e_smooth = float(w_e_smooth)
        self.w_mrm_res = float(w_mrm_res)
        self.w_mrm_assign_tv = float(w_mrm_assign_tv)
        self.edge_thresh = float(edge_thresh)
        self.rec_loss = nn.MSELoss() if loss_type == "MSE" else nn.L1Loss()

    def __call__(self, preds: torch.Tensor, x: torch.Tensor, aux: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        aux = aux or {}
        x_mean = torch.mean(x, dim=1, keepdim=True)

        tau = aux.get("atm_tau", None)
        a = aux.get("atm_A", None)
        if tau is None or a is None:
            return self.base.loss_rec(preds, x)

        # Keep all scene/imaging computations at the same spatial resolution as tau/A.
        if x_mean.shape[-2:] != tau.shape[-2:]:
            x_mean = F.interpolate(x_mean, size=tau.shape[-2:], mode="bilinear", align_corners=False)

        # Remove atmosphere (do not invert TIM here).
        x_scene_est = (x_mean - (1.0 - tau) * a) / (tau + 1e-6)
        x_scene_est = x_scene_est.clamp(0.0, 1.0)

        # Scene radiance via TeV, but env uses scene_est as low-frequency prior.
        e = self.base.rec_e(preds)
        t = self.base.rec_T(preds)
        env = self.base.rec_env(preds, x_scene_est)
        s_scene = e * t + (1.0 - e) * env

        # Forward imaging: atmosphere then inertia blur.
        x_hat_pre = tau * s_scene + (1.0 - tau) * a
        tim = aux.get("tim_module", None)
        x_hat = tim(x_hat_pre) if tim is not None else x_hat_pre

        loss_rec = self.rec_loss(x_hat, x_mean)
        loss_scene = self.rec_loss(s_scene, x_scene_est)

        # Edge-aware emissivity smoothness
        edge = _sobel_edges(x_scene_est)
        edge_mask = (edge > self.edge_thresh).float()
        non_edge = 1.0 - edge_mask

        dx = (e[..., :, 1:] - e[..., :, :-1]).abs()
        dy = (e[..., 1:, :] - e[..., :-1, :]).abs()
        non_edge_x = non_edge[..., :, 1:]
        non_edge_y = non_edge[..., 1:, :]
        loss_e_smooth = (dx * non_edge_x).mean() + (dy * non_edge_y).mean()

        e_delta_logit = aux.get("mrm_e_delta_logit", None)
        loss_mrm_res = e_delta_logit.abs().mean() if e_delta_logit is not None else torch.tensor(0.0, device=x.device)

        p = aux.get("mrm_p", None)
        loss_p_tv = _tv(p) if p is not None else torch.tensor(0.0, device=x.device)

        total = (
            self.w_rec * loss_rec
            + self.w_scene * loss_scene
            + self.w_e_smooth * loss_e_smooth
            + self.w_mrm_res * loss_mrm_res
            + self.w_mrm_assign_tv * loss_p_tv
        )
        return total


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)
