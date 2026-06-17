from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "KDcfg": (".latent_surrogate", "KDcfg"),
    "KDLoss": (".latent_surrogate", "KDLoss"),
    "LatentSurrogate": (".latent_surrogate", "LatentSurrogate"),
    "SurCfg": (".latent_surrogate", "SurCfg"),
    "TeacherComposite": (".latent_surrogate", "TeacherComposite"),
    "grad_align_loss": (".latent_surrogate", "grad_align_loss"),
    "load_module_checkpoint": (".latent_surrogate", "load_module_checkpoint"),
    "sample_perturbed_latent": (".latent_surrogate", "sample_perturbed_latent"),
    "PhysDecoder": (".phys_decoder", "PhysDecoder"),
    "cosine_per_sample": (".phys_losses", "cosine_per_sample"),
    "l1_per_sample": (".phys_losses", "l1_per_sample"),
    "normalize_01": (".phys_losses", "normalize_01"),
    "sobel_mag": (".phys_losses", "sobel_mag"),
    "ssim_per_sample": (".phys_losses", "ssim_per_sample"),
    "tv_loss": (".phys_losses", "tv_loss"),
    "tv_weighted": (".phys_losses", "tv_weighted"),
    "PhysCPEN": (".phys_proxy", "PhysCPEN"),
    "PhysSur": (".phys_surrogate", "PhysSur"),
    "TeR_B": (".terb_core", "TeR_B"),
    "build_lowres_targets": (".phys_utils", "build_lowres_targets"),
    "ramp_weight": (".phys_utils", "ramp_weight"),
    "set_requires_grad": (".phys_utils", "set_requires_grad"),
    "split_proxy_tensor": (".phys_utils", "split_proxy_tensor"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
