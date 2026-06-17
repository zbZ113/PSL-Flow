import argparse
import copy
import inspect
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import yaml
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.generative_models.sit_networks import sit_networks
from models.generative_models.sit_networks.transport import Sampler, create_transport
from models.psl_vae import PSL_VAE, build_psl_vae_terb_q

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real PSL-Flow and KLVAE->SiT inference routes."
    )
    parser.add_argument("--ckpt", type=str, default="", help="Full SiT Lightning checkpoint path.")
    parser.add_argument("--image", type=str, default="", help="Input RGB image path.")
    parser.add_argument(
        "--route",
        type=str,
        default="auto",
        choices=["auto", "sit", "psl_flow"],
        help="Route to benchmark. `auto` infers from checkpoint config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device, e.g. cuda:0 or cpu. Default: auto.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional dataset name, e.g. AVIID. If empty, infer from paths.",
    )
    parser.add_argument(
        "--dataset-idx",
        type=int,
        default=None,
        help="Optional dataset index override.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs before timing. Default: 5.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of timed runs. Default: 20.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast(fp16) during route inference.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="dopri5",
        help="ODE sampling method. Default: dopri5.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of ODE integration outputs. Default: 50.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="ODE absolute tolerance. Default: 1e-6.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="ODE relative tolerance. Default: 1e-3.",
    )
    parser.add_argument(
        "--include-teacher-encode",
        action="store_true",
        help=(
            "Also run `TeR-B Net -> PSL-VAE.encode_from_ir` before SiT sampling. "
            "This is not part of the default generation route unless you explicitly want to count it."
        ),
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="",
        help="Optional override for teacher checkpoint when --include-teacher-encode is used.",
    )
    return parser.parse_args()


def prompt_if_empty(value: str, prompt: str) -> str:
    if str(value).strip():
        return str(value).strip()
    return input(prompt).strip()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(ckpt_path: str) -> dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")
    return checkpoint


def state_dict_from_checkpoint(checkpoint: dict) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a usable state_dict.")
    return state_dict


def has_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state_dict.keys())


def extract_prefixed_state(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    prefix_len = len(prefix)
    return {key[prefix_len:]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_dataset_index_map() -> dict[str, int]:
    mapping: dict[str, int] = {}
    cfg_dir = REPO_ROOT / "configs" / "datasets"
    for cfg_path in sorted(cfg_dir.glob("*.yml")):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "dataset_index" in data:
            mapping[cfg_path.stem] = int(data["dataset_index"])
    return mapping


def infer_dataset_name(paths: list[str], candidates: list[str]) -> str:
    lowered_paths = [str(Path(path)).replace("\\", "/").lower() for path in paths if path]
    for candidate in sorted(candidates, key=len, reverse=True):
        token = candidate.lower()
        for lowered_path in lowered_paths:
            if token in lowered_path:
                return candidate
    return ""


def normalize_rgb(tensor: torch.Tensor) -> torch.Tensor:
    return TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize_gray_signed(tensor: torch.Tensor) -> torch.Tensor:
    return TF.normalize(tensor, mean=[0.5], std=[0.5])


def default_rgb_vae_config(latent_channels: int = 4) -> dict:
    return {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "act_fn": "silu",
        "latent_channels": int(latent_channels),
        "norm_num_groups": 32,
        "sample_size": 256,
        "force_upcast": True,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "mid_block_add_attention": True,
    }


def build_rgb_vae_config(model_config: dict) -> dict:
    if "rgb_vae_config" in model_config and isinstance(model_config["rgb_vae_config"], dict):
        cfg = copy.deepcopy(model_config["rgb_vae_config"])
    else:
        cfg = default_rgb_vae_config(
            latent_channels=int(model_config.get("rgb_vae_latent_channels", 4))
        )
    cfg["in_channels"] = 3
    cfg["out_channels"] = 3
    return cfg


def build_autoencoder_kl(cfg: dict, tag: str) -> AutoencoderKL:
    cfg = dict(cfg)
    try:
        sig = inspect.signature(AutoencoderKL.__init__)
        valid_keys = {k for k in sig.parameters.keys() if k not in {"self", "args", "kwargs"}}
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    except Exception:
        pass
    return AutoencoderKL(**cfg)


def load_plain_module(module: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = False) -> tuple[list[str], list[str]]:
    incompat = module.load_state_dict(state, strict=strict)
    missing_keys = list(getattr(incompat, "missing_keys", []))
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []))
    return missing_keys, unexpected_keys


def build_sit_model(model_config: dict, route: str) -> tuple[nn.Module, int, int]:
    injection_args = copy.deepcopy(model_config.get("injection_args", {}))
    num_classes = 1000
    if route == "sit":
        latent_channels = 4
        downsample_factor = int(model_config.get("vae_divisible", 8))
        injection_args.setdefault("rgb_in_chans", 4)
    elif route == "psl_flow":
        latent_channels = int(model_config["vae_config"]["latent_channels"])
        downsample_factor = 2 ** max(0, len(model_config["vae_config"]["block_out_channels"]) - 1)
        injection_args.setdefault("rgb_in_chans", int(model_config.get("rgb_vae_latent_channels", 4)))
        if int(injection_args.get("phys_in_chans", 0) or 0) > 0:
            injection_args["phys_in_chans"] = latent_channels
    else:
        raise ValueError(f"Unsupported route: {route}")

    sit_model = sit_networks.SiT_models[f"SiT-{model_config['arch']}/{model_config['patch_size']}"](
        in_channels=latent_channels,
        num_classes=num_classes,
        injection_args=injection_args,
        learn_sigma=True if "pretrain_load" in model_config else False,
        repa=True if model_config.get("repa", False) else False,
    )
    sit_model.eval()
    sit_model.requires_grad_(False)
    return sit_model, latent_channels, downsample_factor


def load_sit_weights(
    sit_model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    prefer_ema: bool,
) -> tuple[str, list[str], list[str]]:
    if prefer_ema and has_prefix(state_dict, "ema."):
        prefix = "ema."
    elif has_prefix(state_dict, "model."):
        prefix = "model."
    elif has_prefix(state_dict, "ema."):
        prefix = "ema."
    else:
        raise KeyError("Neither `model.` nor `ema.` weights were found in the checkpoint.")

    missing_keys, unexpected_keys = load_plain_module(
        sit_model,
        extract_prefixed_state(state_dict, prefix),
        strict=False,
    )
    return prefix.rstrip("."), missing_keys, unexpected_keys


def build_rgb_vae(model_config: dict, state_dict: dict[str, torch.Tensor]) -> tuple[AutoencoderKL, list[str], list[str]]:
    if str(model_config.get("rgb_vae_model", "klvae")).lower() != "klvae":
        raise NotImplementedError("This benchmarking script currently supports RGB KLVAE only.")
    rgb_state = extract_prefixed_state(state_dict, "RGB_vae.")
    if not rgb_state:
        raise KeyError("RGB_vae weights were not found in the full SiT checkpoint.")
    rgb_vae = build_autoencoder_kl(build_rgb_vae_config(model_config), tag="RGB VAE")
    missing_keys, unexpected_keys = load_plain_module(
        rgb_vae,
        rgb_state,
        strict=False,
    )
    rgb_vae.eval()
    rgb_vae.requires_grad_(False)
    return rgb_vae, missing_keys, unexpected_keys


def build_thermal_vae(
    model_config: dict,
    state_dict: dict[str, torch.Tensor],
    route: str,
) -> tuple[nn.Module, list[str], list[str]]:
    thermal_state = extract_prefixed_state(state_dict, "thermal_vae.")
    if not thermal_state:
        raise KeyError("thermal_vae weights were not found in the full SiT checkpoint.")
    if route == "sit":
        thermal_vae = build_autoencoder_kl(model_config["vae_config"], tag="Thermal KLVAE")
    elif route == "psl_flow":
        thermal_vae = PSL_VAE(model_config["vae_config"])
    else:
        raise ValueError(f"Unsupported route: {route}")

    missing_keys, unexpected_keys = load_plain_module(
        thermal_vae,
        thermal_state,
        strict=False,
    )
    thermal_vae.eval()
    thermal_vae.requires_grad_(False)
    return thermal_vae, missing_keys, unexpected_keys


def build_teacher_if_needed(
    model_config: dict,
    args: argparse.Namespace,
) -> nn.Module | None:
    if not args.include_teacher_encode:
        return None
    if str(model_config.get("vae_model", "")).lower() not in {"phys_factor_vae", "psl_vae"}:
        raise ValueError("--include-teacher-encode is only valid for the PSL-Flow route.")
    teacher_cfg = dict(model_config.get("teacher", {}))
    if args.teacher_ckpt:
        teacher_cfg["ckpt"] = args.teacher_ckpt
    if not str(teacher_cfg.get("ckpt", "")).strip():
        raise ValueError("Teacher encode was requested, but no teacher checkpoint is configured.")
    teacher, _ = build_psl_vae_terb_q(
        teacher_cfg,
        ckpt_path=str(teacher_cfg.get("ckpt", "")),
        strict=bool(teacher_cfg.get("strict_load", False)),
    )
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


class SiTRoute(nn.Module):
    def __init__(
        self,
        *,
        route_name: str,
        model_config: dict,
        sit_model: nn.Module,
        rgb_vae: AutoencoderKL,
        thermal_vae: nn.Module,
        latent_channels: int,
        downsample_factor: int,
        dataset_idx: int,
        sampling_method: str,
        num_steps: int,
        atol: float,
        rtol: float,
        include_teacher_encode: bool,
        teacher: nn.Module | None = None,
    ):
        super().__init__()
        self.route_name = route_name
        self.model_config = copy.deepcopy(model_config)
        self.sit_model = sit_model
        self.rgb_vae = rgb_vae
        self.thermal_vae = thermal_vae
        self.teacher = teacher
        self.dataset_idx = int(dataset_idx)
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)
        self.include_teacher_encode = bool(include_teacher_encode)
        self.transport = create_transport(**self.model_config["transport_config"])
        self.sampler = Sampler(self.transport)
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        self.use_cfg = float(self.model_config.get("cfg_scale", 1.0)) > 1.0
        self.cfg_scale = float(self.model_config.get("cfg_scale", 1.0))
        self.thermal_normalizer = self.model_config.get("thermal_normalizer", None)
        self.rgb_normalizer = self.model_config.get("RGB_normalizer", None)
        self.rgb_vae_model = str(self.model_config.get("rgb_vae_model", "klvae")).lower()
        injection_args = self.model_config.get("injection_args", {})
        self.phys_in_chans = int(injection_args.get("phys_in_chans", 0) or 0)

    def encode_rgb_latent(self, rgb_signed: torch.Tensor) -> torch.Tensor:
        rgb_in = rgb_signed
        if self.rgb_vae_model != "klvae":
            raise NotImplementedError("Only RGB KLVAE is supported.")
        x_rgb = self.rgb_vae.encode(rgb_in).latent_dist.sample()
        if self.rgb_normalizer is not None:
            x_rgb = x_rgb * float(self.rgb_normalizer)
        return x_rgb

    def decode_thermal_latent(self, z_latent: torch.Tensor) -> torch.Tensor:
        z = z_latent
        if self.thermal_normalizer is not None:
            z = z / float(self.thermal_normalizer)
        if self.route_name == "sit":
            return self.thermal_vae.decode(z).sample
        decoded = self.thermal_vae.decode_latents(z, recompose_mode="full")
        y_hat = decoded["y_hat"]
        return torch.clamp(y_hat, 0.0, 1.0) * 2.0 - 1.0

    def encode_teacher_phys_latent(self, gray_01: torch.Tensor) -> torch.Tensor | None:
        if self.route_name != "psl_flow" or not self.include_teacher_encode:
            return None
        if self.teacher is None:
            raise RuntimeError("Teacher branch requested but teacher is not initialized.")
        if not isinstance(self.thermal_vae, PSL_VAE):
            raise RuntimeError("Teacher branch requires PSL-VAE.")
        self.thermal_vae.attach_teacher(self.teacher)
        encoded = self.thermal_vae.encode_from_ir(gray_01, sample=False)
        z_phys = encoded["z_phys"]
        if self.thermal_normalizer is not None:
            z_phys = z_phys * float(self.thermal_normalizer)
        return z_phys

    def forward(self, rgb_signed: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = rgb_signed.shape
        latent_h = height // self.downsample_factor
        latent_w = width // self.downsample_factor
        zs = torch.randn(
            batch_size,
            self.latent_channels,
            latent_h,
            latent_w,
            device=rgb_signed.device,
            dtype=rgb_signed.dtype,
        )
        dataset_idx = torch.full(
            (batch_size,),
            fill_value=self.dataset_idx,
            device=rgb_signed.device,
            dtype=torch.long,
        )

        x_rgb = self.encode_rgb_latent(rgb_signed)
        x_phys = None
        if self.include_teacher_encode and self.route_name == "psl_flow":
            gray_01 = torch.clamp(rgb_signed.mean(dim=1, keepdim=True) * 0.5 + 0.5, 0.0, 1.0)
            x_phys = self.encode_teacher_phys_latent(gray_01)

        if self.use_cfg:
            cfg_condition = str(self.model_config.get("cfg_condition", "label")).lower()
            zs_model = torch.cat([zs, zs], dim=0)
            if cfg_condition == "label":
                y_null = torch.full_like(dataset_idx, 1000)
                y_model = torch.cat([dataset_idx, y_null], dim=0)
                x_rgb_model = torch.cat([x_rgb, x_rgb], dim=0)
            elif cfg_condition == "rgb":
                y_model = torch.cat([dataset_idx, dataset_idx], dim=0)
                x_rgb_model = torch.cat([x_rgb, torch.zeros_like(x_rgb)], dim=0)
            elif cfg_condition == "both":
                y_null = torch.full_like(dataset_idx, 1000)
                y_model = torch.cat([dataset_idx, y_null], dim=0)
                x_rgb_model = torch.cat([x_rgb, torch.zeros_like(x_rgb)], dim=0)
            else:
                raise ValueError(f"Unknown cfg_condition: {cfg_condition}")
            x_phys_model = None if x_phys is None else torch.cat([x_phys, x_phys], dim=0)
            samples = self.sample_fn(
                zs_model,
                self.sit_model.forward_with_cfg,
                y=y_model,
                x_RGB=x_rgb_model,
                cfg_scale=self.cfg_scale,
                x_phys=x_phys_model,
            )[-1]
            samples, _ = samples.chunk(2, dim=0)
        else:
            if self.model_config.get("force_un", False):
                y_model = torch.full_like(dataset_idx, 1000)
            else:
                y_model = dataset_idx
            kwargs = {"y": y_model, "x_RGB": x_rgb}
            if x_phys is not None and self.phys_in_chans > 0:
                kwargs["x_phys"] = x_phys
            samples = self.sample_fn(
                zs,
                self.sit_model.forward,
                **kwargs,
            )[-1]

        return self.decode_thermal_latent(samples)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def benchmark_runtime(
    route_module: nn.Module,
    rgb_input: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
    use_amp: bool,
) -> float:
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            with autocast_context(device, use_amp):
                _ = route_module(rgb_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(runs, 1)):
            with autocast_context(device, use_amp):
                _ = route_module(rgb_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return elapsed / max(runs, 1)


def profile_flops(
    route_module: nn.Module,
    rgb_input: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> int:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            with autocast_context(device, use_amp):
                _ = route_module(rgb_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    total_flops = 0
    for event in prof.key_averages():
        total_flops += int(getattr(event, "flops", 0) or 0)
    return total_flops


def main() -> None:
    args = parse_args()
    ckpt_path = os.path.abspath(prompt_if_empty(args.ckpt, "Enter full SiT checkpoint path: "))
    image_path = os.path.abspath(prompt_if_empty(args.image, "Enter RGB image path: "))

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint = load_checkpoint(ckpt_path)
    state_dict = state_dict_from_checkpoint(checkpoint)
    hyper = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyper, dict):
        raise ValueError("Checkpoint does not contain `hyper_parameters`.")
    model_config = copy.deepcopy(hyper.get("model_config", {}))
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint hyper_parameters does not contain a usable `model_config`.")
    if str(hyper.get("model_arch", "")).lower() != "sit":
        raise ValueError(f"This script expects a SiT checkpoint, but got model_arch={hyper.get('model_arch')}.")

    ckpt_vae_model = str(model_config.get("vae_model", "")).lower()
    inferred_route = {
        "klvae": "sit",
        "phys_factor_vae": "psl_flow",
        "psl_vae": "psl_flow",
    }.get(ckpt_vae_model, "")
    if args.route == "auto":
        if not inferred_route:
            raise ValueError(
                f"Unable to infer route from vae_model={ckpt_vae_model}. "
                "Use --route sit or --route psl_flow explicitly."
            )
        route = inferred_route
    else:
        route = args.route

    if route == "sit" and ckpt_vae_model != "klvae":
        raise ValueError(f"--route sit expects vae_model=klvae, but checkpoint has vae_model={ckpt_vae_model}.")
    if route == "psl_flow" and ckpt_vae_model not in {"phys_factor_vae", "psl_vae"}:
        raise ValueError(
            f"--route psl_flow expects vae_model=psl_vae, but checkpoint has vae_model={ckpt_vae_model}."
        )

    dataset_index_map = load_dataset_index_map()
    dataset_name = args.dataset_name.strip()
    if not dataset_name:
        dataset_name = infer_dataset_name([ckpt_path, image_path], list(dataset_index_map.keys()))
    if args.dataset_idx is not None:
        dataset_idx = int(args.dataset_idx)
    elif dataset_name and dataset_name in dataset_index_map:
        dataset_idx = int(dataset_index_map[dataset_name])
    else:
        raw = input("Unable to infer dataset_idx. Enter an integer [default: 0]: ").strip()
        dataset_idx = int(raw) if raw else 0

    device = resolve_device(args.device)
    image = Image.open(image_path).convert("RGB")
    rgb_input = normalize_rgb(TF.to_tensor(image)).unsqueeze(0).to(device)

    sit_model, latent_channels, downsample_factor = build_sit_model(model_config, route)
    sit_weight_source, sit_missing, sit_unexpected = load_sit_weights(
        sit_model,
        state_dict,
        prefer_ema=str(hyper.get("validation_type", "ema")).lower() == "ema",
    )

    rgb_vae, rgb_missing, rgb_unexpected = build_rgb_vae(model_config, state_dict)
    thermal_vae, thermal_missing, thermal_unexpected = build_thermal_vae(model_config, state_dict, route)
    teacher = build_teacher_if_needed(model_config, args)

    route_module = SiTRoute(
        route_name=route,
        model_config=model_config,
        sit_model=sit_model,
        rgb_vae=rgb_vae,
        thermal_vae=thermal_vae,
        latent_channels=latent_channels,
        downsample_factor=downsample_factor,
        dataset_idx=dataset_idx,
        sampling_method=args.sampling_method,
        num_steps=args.num_steps,
        atol=args.atol,
        rtol=args.rtol,
        include_teacher_encode=args.include_teacher_encode,
        teacher=teacher,
    ).to(device)
    route_module.eval()

    params_total = sum(param.numel() for param in route_module.parameters())
    flops_total = profile_flops(route_module, rgb_input, device, use_amp=args.amp)
    rt_avg = benchmark_runtime(
        route_module,
        rgb_input,
        device,
        warmup=args.warmup,
        runs=args.runs,
        use_amp=args.amp,
    )

    print("=" * 80)
    print(f"Checkpoint         : {ckpt_path}")
    print(f"Image              : {image_path}")
    print(f"Route              : {route}")
    print(f"Checkpoint vae_model: {ckpt_vae_model}")
    print(f"SiT weights source : {sit_weight_source}")
    print(f"Dataset            : {dataset_name or '<unknown>'}")
    print(f"Dataset idx        : {dataset_idx}")
    print(f"Device             : {device}")
    print(f"Image size         : {rgb_input.shape[-2]} x {rgb_input.shape[-1]}")
    print(f"ODE sampler        : {args.sampling_method}")
    print(f"ODE steps          : {args.num_steps}")
    print(f"Teacher encode     : {args.include_teacher_encode}")
    if route == "psl_flow" and not args.include_teacher_encode:
        print("Teacher note       : TeR-B Net is not on the default generation path for this route.")
    print("-" * 80)
    print(f"FLOPs/G            : {flops_total / 1e9:.4f}")
    print(f"Params/M           : {params_total / 1e6:.4f}")
    print(f"RT/s               : {rt_avg:.6f}")
    print("-" * 80)
    print(
        "Load stats         : "
        f"sit(m={len(sit_missing)},u={len(sit_unexpected)}) "
        f"rgb_vae(m={len(rgb_missing)},u={len(rgb_unexpected)}) "
        f"thermal_vae(m={len(thermal_missing)},u={len(thermal_unexpected)})"
    )
    if flops_total == 0:
        print("[WARN] torch.profiler returned 0 FLOPs. Some operators are not covered.")
    print("=" * 80)


if __name__ == "__main__":
    main()
