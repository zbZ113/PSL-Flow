import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import torchvision
import torch.nn.functional as F
import os
import numpy as np
import itertools
import inspect
import matplotlib

# Force a non-interactive backend so validation image export is safe on headless
# servers and inside dataloader / trainer worker contexts.
matplotlib.use("Agg")

from utils.losses import get_loss
from utils.metrics import calculate_psnr, calculate_ssim, calculate_fid, calculate_lpips
from models.generative_models.pix2pix_networks.networks import UnetGenerator, NLayerDiscriminator, get_norm_layer, ResnetGenerator
from models.generative_models.pix2pixHD_networks.networks import define_G, define_D
from models.generative_models.vqgan_networks.networks import VQGAN
from diffusers.models import AutoencoderKL
try:
    from diffusers.models import AutoencoderDC
except Exception:
    AutoencoderDC = None
from models.generative_models.sit_networks import sit_networks
from models.generative_models.sit_networks.transport import create_transport, Sampler
from models.physics import (
    PhysCPEN,
    PhysDecoder,
    PhysSur,
    TeR_B,
    build_lowres_targets,
    cosine_per_sample,
    l1_per_sample,
    load_module_checkpoint,
    normalize_01,
    ramp_weight,
    set_requires_grad,
    sobel_mag,
    ssim_per_sample,
    tv_loss,
    tv_weighted,
)
from models.phys_vae_r import (
    PhysVAER,
    PhysVAERLoss,
    weighted_joint_flow_loss,
)
from models.psl_vae import (
    PSL_VAE,
    PSL_VAELoss,
    build_psl_vae_terb_q,
)
from dataloaders.GenericDataloader import IMAGENET_MEAN_STD, NORMAL_MEAN_STD
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from collections import OrderedDict
import copy


def _canonical_token(value) -> str:
    return str(value).strip().lower().replace("-", "_")


def _canonical_model_arch(value):
    token = _canonical_token(value)
    if token in {"psl_vae", "pslvae"}:
        return "phys_factor_vae"
    return value


def _canonical_vae_model(value):
    token = _canonical_token(value)
    if token in {"psl_vae", "pslvae"}:
        return "phys_factor_vae"
    return value


def _canonical_loss_name(value):
    token = _canonical_token(value)
    if token in {"psl_vae", "pslvae"}:
        return "phys_factor_vae"
    return value


def _build_autoencoder_kl(cfg: dict, tag: str = "AutoencoderKL"):
    cfg = dict(cfg)
    try:
        sig = inspect.signature(AutoencoderKL.__init__)
        valid_keys = {k for k in sig.parameters.keys() if k not in {"self", "args", "kwargs"}}
        filtered_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        dropped = [k for k in cfg.keys() if k not in filtered_cfg]
    except Exception:
        filtered_cfg = cfg
        dropped = []

    if dropped:
        print(f"[WARN] {tag}: ignore unsupported AutoencoderKL args for current diffusers: {dropped}")
    return AutoencoderKL(**filtered_cfg)


def _load_autoencoder_kl_pretrained(path_or_repo: str, local_files_only: bool = False):
    return AutoencoderKL.from_pretrained(path_or_repo, local_files_only=local_files_only)


class PSLFlow(pl.LightningModule):
    """Lightning module for PSL-Flow and its KLVAE->SiT ablation route."""

    def __init__(self,
        #---- Backbone
        model_arch='pix2pix',
        model_config={
            "G_arch": "unet",
            "D_arch": "patchGAN",
        },
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
        },
        
        #----- Loss
        loss_name='pix2pix', 
        loss_config = {
            'G_mode': 'lsgan',
            'G_loss_lambda': 100.0,
        },
        training_stage='full',
        gradient_accumulation=1,
        calculate_stats=False,
        validation_type="ema",
    ):
        super().__init__()

        model_arch = _canonical_model_arch(model_arch)
        loss_name = _canonical_loss_name(loss_name)
        model_config = copy.deepcopy(model_config)
        if isinstance(model_config, dict) and "vae_model" in model_config:
            model_config["vae_model"] = _canonical_vae_model(model_config["vae_model"])

        # Disable Auto Optim for GAN
        self.automatic_optimization = False
        # Backbone
        self.model_arch = model_arch
        self.model_config = model_config

        # Train hyperparameters
        self.lr = float(lr)
        self.optimizer = optimizer
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.loss_config = loss_config
        self.training_stage = training_stage
        self.gradient_accumulation = gradient_accumulation
        self.calculate_stats = calculate_stats
        self.validation_type = validation_type
        self.eval_vis_num = int(self.model_config.get("eval_vis_num", 4))
        self.save_eval_images_local = bool(self.model_config.get("save_eval_images_local", True))
        self.save_all_eval_samples = str(self.model_config.get("save_all_eval_samples", False)).lower() in {"1", "true", "yes", "y", "on"}
        self.psl_recompose_mode = str(
            self.model_config.get(
                "psl_recompose_mode",
                self.model_config.get("phys_factor_recompose_mode", "full"),
            )
            or "full"
        ).lower()
        self.phys_factor_recompose_mode = self.psl_recompose_mode
        
        self.save_hyperparameters() # write hyperparams into a file
        self.fail_on_nan = bool(self.model_config.get("fail_on_nan", False))
        self._last_eval_image_dir = None
        
        if self.loss_name == "pix2pix" or self.loss_name == "cyclegan" or self.loss_name == "pix2pixHD":
            self.loss_fn_GAN, self.loss_fn_L1 = get_loss(loss_name, loss_config)
        else:
            self.loss_fn = get_loss(loss_name, loss_config)
        
        # ----------------------------------
        # get the backbone and the aggregator
        if self.model_arch == "pix2pix":
            norm_layer = get_norm_layer(norm_type=self.model_config['GAN_norm'])
            if self.model_config["G_arch"] == "unet":
                self.model = UnetGenerator(3, 1, 8, norm_layer=norm_layer, upsample=self.model_config['GAN_upsample'])
                self.model.divisible = model_config['divisible']
                self.output_type = "tanh"
            else:
                raise NotImplementedError()
            if self.model_config["D_arch"] == "patchGAN":
                self.discriminator = NLayerDiscriminator(3+1, norm_layer=norm_layer)
            else:
                raise NotImplementedError()
        elif self.model_arch == "cyclegan":
            norm_layer = get_norm_layer(norm_type=self.model_config['GAN_norm'])
            if self.model_config["G_arch"] == "resnet":
                self.model_A = ResnetGenerator(1, 3, norm_layer=norm_layer, upsample=self.model_config['GAN_upsample'], n_blocks=9)
                self.model = ResnetGenerator(3, 1, norm_layer=norm_layer, upsample=self.model_config['GAN_upsample'], n_blocks=9)
                self.model.divisible = model_config['divisible']
                self.output_type = "tanh"
            else:
                raise NotImplementedError()
            if self.model_config["D_arch"] == "patchGAN":
                self.discriminator_A = NLayerDiscriminator(3, norm_layer=norm_layer)
                self.discriminator_B = NLayerDiscriminator(1, norm_layer=norm_layer)
            else:
                raise NotImplementedError()
        elif self.model_arch == "pix2pixHD":
            if self.model_config["G_arch"] == "global":
                self.model = define_G(3, 1, 64, "global", n_downsample_global=self.model_config["n_downsample_global"], norm=self.model_config['GAN_norm'], upsample=self.model_config['GAN_upsample'])
                self.model.divisible = model_config['divisible']
                self.output_type = "tanh"
            else:
                raise NotImplementedError()
            if self.model_config["D_arch"] == "patchGAN":
                self.discriminator = define_D(3+1, 64, self.model_config["n_layers_D"], num_D=self.model_config["num_D"], getIntermFeat=True, norm=self.model_config['GAN_norm'])
            else:
                raise NotImplementedError()
        elif self.model_arch == "vqgan":
            self.model = VQGAN(self.model_config)
            self.model.divisible = model_config['divisible']
            self.output_type = "normal"
        elif self.model_arch == "klvae" or self.model_arch == "klvae_RGB":
            divisible = self.model_config['divisible']
            self.model_config.pop('divisible')
            self.model = _build_autoencoder_kl(self.model_config, tag="model_arch=klvae")
            self.model.divisible = divisible
            self.output_type = "normal"
        elif self.model_arch == "phys_vae_r":
            model_cfg = dict(self.model_config)
            divisible = int(model_cfg.pop('divisible', 8))
            self.model = PhysVAER(model_cfg)
            self.model.divisible = divisible
            self.output_type = "normal"
            self.phys_vae_r_stage = self._resolve_phys_vae_r_stage(self.training_stage)
            self.model.set_stage(self.phys_vae_r_stage)
            teacher_cfg = self.loss_config.get("teacher", {}) if isinstance(self.loss_config, dict) else {}
            teacher_ckpt = ""
            if isinstance(teacher_cfg, dict):
                teacher_ckpt = str(teacher_cfg.get("ckpt", ""))
            if not teacher_ckpt:
                raise ValueError("phys_vae_r loss config requires teacher.ckpt.")
            self.phys_vae_r_loss = PhysVAERLoss(
                teacher=self._build_phys_teacher_q(teacher_cfg),
                config=self.loss_config,
            )
        elif self.model_arch == "phys_factor_vae":
            model_cfg = dict(self.model_config)
            divisible = int(model_cfg.pop('divisible', 8))
            self.model = PSL_VAE(model_cfg)
            self.model.divisible = divisible
            self.output_type = "normal"
            teacher_cfg = self.loss_config.get("teacher", {}) if isinstance(self.loss_config, dict) else {}
            teacher_ckpt = ""
            if isinstance(teacher_cfg, dict):
                teacher_ckpt = str(teacher_cfg.get("ckpt", ""))
            if not teacher_ckpt:
                raise ValueError("PSL-VAE loss config requires teacher.ckpt.")
            teacher, load_info = build_psl_vae_terb_q(
                teacher_cfg,
                ckpt_path=teacher_ckpt,
                strict=bool(teacher_cfg.get("strict_load", False)),
            )
            print(
                f"[INFO] Loaded PSL-VAE TeR-B Net from {teacher_ckpt} "
                f"(source={load_info.get('state_source', 'unknown')}, state_tensors={load_info.get('num_state_tensors', 0)}, "
                f"missing={len(load_info.get('missing_keys', []))}, unexpected={len(load_info.get('unexpected_keys', []))})"
            )
            self.model.attach_teacher(teacher)
            self.phys_factor_vae_loss = PSL_VAELoss(config=self.loss_config)
        elif self.model_arch == "dcae":
            if AutoencoderDC is None:
                raise ImportError("AutoencoderDC is not available in your installed diffusers. Please install/upgrade diffusers with DC-AE support, or use klvae.")
            divisible = self.model_config['divisible']
            self.model_config.pop('divisible')
            self.model = AutoencoderDC(**self.model_config)
            self.model.divisible = divisible
            self.output_type = "normal"
        elif self.model_arch == "sit":
            num_classes = 1000
            injection_args = copy.deepcopy(self.model_config['injection_args'])
            if self.model_config['vae_model'] == "klvae":
                sit_in_channels = 4
                rgb_in_channels = 4
                injection_args.setdefault('rgb_in_chans', rgb_in_channels)
                self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](in_channels=sit_in_channels, num_classes=num_classes, injection_args=injection_args, learn_sigma=True if 'pretrain_load' in self.model_config else False, repa=True if 'repa' in self.model_config and self.model_config['repa'] else False)
                self.thermal_vae = _build_autoencoder_kl(self.model_config['vae_config'], tag="sit.vae_config")
                self.thermal_vae = self.load_pretrained(
                    self.thermal_vae,
                    allow_missing=bool(self.model_config.get("allow_missing_vae_path", False)),
                )
                self.RGB_vae = self.load_rgb_vae_kl()
                self.thermal_latent_channels = sit_in_channels
                self.thermal_downsample_factor = int(self.model_config.get('vae_divisible', 8))
            elif self.model_config['vae_model'] == "dcae":
                if AutoencoderDC is None:
                    raise ImportError("vae_model=dcae requires AutoencoderDC from diffusers, but it is unavailable in current environment.")
                self.thermal_vae = AutoencoderDC(**self.model_config['vae_config'])
                self.thermal_vae = self.load_pretrained(self.thermal_vae)
                if model_config['divisible'] == 32:
                    sit_in_channels = 32
                    rgb_in_channels = 32
                    injection_args.setdefault('rgb_in_chans', rgb_in_channels)
                    self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](in_channels=sit_in_channels, num_classes=num_classes, injection_args=injection_args, repa=True if 'repa' in self.model_config and self.model_config['repa'] else False)
                    self.RGB_vae = AutoencoderDC().from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
                    self.thermal_latent_channels = sit_in_channels
                    self.thermal_downsample_factor = 32
                elif model_config['divisible'] == 64:
                    sit_in_channels = 128
                    rgb_in_channels = 128
                    injection_args.setdefault('rgb_in_chans', rgb_in_channels)
                    self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](in_channels=sit_in_channels, num_classes=num_classes, injection_args=injection_args, repa=True if 'repa' in self.model_config and self.model_config['repa'] else False)
                    self.RGB_vae = AutoencoderDC().from_pretrained(f"mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers")
                    self.thermal_latent_channels = sit_in_channels
                    self.thermal_downsample_factor = 64
                else:
                    raise NotImplementedError()
            elif self.model_config['vae_model'] == "phys_vae_r":
                self.thermal_vae = PhysVAER(self.model_config['vae_config'])
                self.thermal_vae = self.load_pretrained_phys_vae_r(
                    self.thermal_vae,
                    allow_missing=bool(self.model_config.get("allow_missing_vae_path", False)),
                )
                self.RGB_vae = self.load_rgb_vae_kl()
                self.thermal_latent_channels = int(self.thermal_vae.joint_channels)
                self.thermal_downsample_factor = int(getattr(self.thermal_vae, "downsample_factor", 8))
                injection_args.setdefault('rgb_in_chans', int(self.model_config.get('rgb_vae_latent_channels', 4)))
                if injection_args.get('phys_in_chans', 0) > 0:
                    injection_args['phys_in_chans'] = int(self.thermal_vae.joint_channels)
                self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](
                    in_channels=int(self.thermal_vae.joint_channels),
                    num_classes=num_classes,
                    injection_args=injection_args,
                    learn_sigma=True if 'pretrain_load' in self.model_config else False,
                    repa=True if 'repa' in self.model_config and self.model_config['repa'] else False,
                )
            elif self.model_config['vae_model'] == "phys_factor_vae":
                self.thermal_vae = PSL_VAE(self.model_config['vae_config'])
                teacher_cfg = dict(self.model_config.get('teacher', {}))
                teacher_ckpt = str(teacher_cfg.get('ckpt', ''))
                if not teacher_ckpt:
                    raise ValueError("PSL-Flow with vae_model=psl_vae requires model_config.teacher.ckpt.")
                teacher, load_info = build_psl_vae_terb_q(
                    teacher_cfg,
                    ckpt_path=teacher_ckpt,
                    strict=bool(teacher_cfg.get('strict_load', False)),
                )
                print(
                    f"[INFO] Loaded PSL-Flow TeR-B Net from {teacher_ckpt} "
                    f"(source={load_info.get('state_source', 'unknown')}, state_tensors={load_info.get('num_state_tensors', 0)}, "
                    f"missing={len(load_info.get('missing_keys', []))}, unexpected={len(load_info.get('unexpected_keys', []))})"
                )
                self.thermal_vae.attach_teacher(teacher)
                self.thermal_vae = self.load_pretrained_phys_factor_vae(
                    self.thermal_vae,
                    allow_missing=bool(self.model_config.get("allow_missing_vae_path", False)),
                )
                self.RGB_vae = self.load_rgb_vae_kl()
                self.thermal_latent_channels = int(self.thermal_vae.latent_channels)
                self.thermal_downsample_factor = int(getattr(self.thermal_vae, "downsample_factor", 8))
                injection_args.setdefault('rgb_in_chans', int(self.model_config.get('rgb_vae_latent_channels', 4)))
                if injection_args.get('phys_in_chans', 0) > 0:
                    injection_args['phys_in_chans'] = int(self.thermal_vae.latent_channels)
                self.model = sit_networks.SiT_models[f"SiT-{self.model_config['arch']}/{self.model_config['patch_size']}"](
                    in_channels=int(self.thermal_vae.latent_channels),
                    num_classes=num_classes,
                    injection_args=injection_args,
                    learn_sigma=True if 'pretrain_load' in self.model_config else False,
                    repa=True if 'repa' in self.model_config and self.model_config['repa'] else False,
                )
            else:
                raise NotImplementedError()
            if self.model.repa:
                self.repa_encoder = self.load_encoder(enc_type="dinov2")
                self.repa_input = model_config['repa_input']
                self.repa_encoder.eval()
                self.repa_encoder.requires_grad_(False)
            self.transport = create_transport(**self.model_config['transport_config'])
            self.sampler = Sampler(self.transport)
            self.ema = copy.deepcopy(self.model)
            self.ema.eval()
            self.ema.requires_grad_(False)
            self.use_cfg = self.model_config['cfg_scale'] > 1.0
            self.thermal_normalizer = self.model_config['thermal_normalizer'] if 'thermal_normalizer' in self.model_config else None
            self.RGB_normalizer = self.model_config['RGB_normalizer'] if 'RGB_normalizer' in self.model_config else None
            self.model.divisible = model_config['divisible']
            self.RGB_encoder_training = model_config['RGB_encoder_training'] if 'RGB_encoder_training' in model_config else False
            self.style_finetuning = model_config['style_finetuning'] if 'style_finetuning' in model_config else False
            self.kl_training = model_config['kl_training'] if 'kl_training' in model_config else False
            self.thermal_vae.eval()
            self.thermal_vae.requires_grad_(False)
            if self.RGB_encoder_training:
                self.RGB_vae.train()
                self.RGB_vae.requires_grad_(True)
            else:
                self.RGB_vae.eval()
                self.RGB_vae.requires_grad_(False)
            if self.kl_training:
                assert self.RGB_encoder_training and self.model_config['vae_model'] == "klvae" # kl training should only be used when RGB encoder is training
            if 'cache_rate' in self.model_config and self.model_config['cache_rate'] > 1:
                self.latent_cache = []
                if not self.RGB_encoder_training:
                    self.RGB_latent_cache = []
                else:
                    self.RGB_cache = []
                self.latent_cache_init = False

            # Optional TeVNet auxiliary loss (ported from PID).
            # Enable by adding `tevnet_config` under `model_config`.
            self.use_tev_loss = False
            self.tev_weight = 0.0
            self.tev_rec_weight = 0.0
            self.tev_t_max = 1.0
            self.tevnet_stage_model = None
            self.tev_loss_obj = None
            self.use_latent_surrogate_loss = False
            self.latent_surrogate_model = None
            self.latent_surrogate_loss_obj = None
            self.latent_surrogate_weight = 0.0
            self.latent_surrogate_t_min = 0.7
            if 'tevnet_config' in self.model_config and isinstance(self.model_config['tevnet_config'], dict):
                cfg = self.model_config['tevnet_config']
                if cfg.get('enable', False):
                    try:
                        import types
                        from models.physics.TeVNet.models import TeVNet, TeVNetTherNet
                        from models.physics.TeVNet.utils import TeVloss
                        tev_args = types.SimpleNamespace(**cfg.get('args', {}))
                        variant = str(cfg.get('variant', 'baseline')).lower()
                        thernet_variants = {'thernet', 'tevnet_thernet', 'tevnetthernet', 'pp', 'phys'}
                        NetCls = TeVNetTherNet if variant in thernet_variants else TeVNet
                        in_ch = int(cfg.get('in_channels', 3))
                        vnums = int(getattr(tev_args, 'vnums', cfg.get('vnums', 4)))
                        out_ch = int(cfg.get('out_channels', 2 + vnums))
                        self.tevnet_stage_model = NetCls(in_channels=in_ch, out_channels=out_ch, args=tev_args)
                        ckpt = cfg.get('ckpt', None)
                        if ckpt:
                            state = torch.load(ckpt, map_location='cpu')
                            if isinstance(state, dict) and 'state_dict' in state:
                                state = state['state_dict']
                            # allow both plain and lightning-style keys
                            new_state = {k.replace('model.', '').replace('module.', ''): v for k, v in state.items()}
                            self.tevnet_stage_model.load_state_dict(new_state, strict=False)
                        self.tevnet_stage_model.eval()
                        for p in self.tevnet_stage_model.parameters():
                            p.requires_grad = False
                        self.tev_loss_obj = TeVloss(vnums=vnums, loss_type=str(cfg.get('loss_type', 'MSE')))
                        self.use_tev_loss = True
                        self.tev_weight = float(cfg.get('tev_weight', 0.0))
                        self.tev_rec_weight = float(cfg.get('tev_rec_weight', 0.0))
                        self.tev_t_max = float(cfg.get('tev_t_max', 1.0))
                    except Exception as e:
                        print(f"[WARN] TeVNet init failed, disable TeV loss. Reason: {e}")
                        self.use_tev_loss = False
                        self.tevnet_stage_model = None
                        self.tev_loss_obj = None

            # Optional latent surrogate loss (P3): no decode+TeV in the SiT graph.
            # This expects a pre-trained surrogate checkpoint distilled from TeVNetTherNet.
            if 'latent_surrogate_config' in self.model_config and isinstance(self.model_config['latent_surrogate_config'], dict):
                ls_cfg = self.model_config['latent_surrogate_config']
                if ls_cfg.get('enable', False):
                    try:
                        from models.physics.latent_surrogate import (
                            KDcfg,
                            KDLoss,
                            LatentSurrogate,
                            SurCfg,
                            load_module_checkpoint,
                        )

                        sur_cfg_dict = ls_cfg.get('surrogate', {}) if isinstance(ls_cfg.get('surrogate', {}), dict) else {}
                        kd_cfg_dict = ls_cfg.get('kd_loss', {}) if isinstance(ls_cfg.get('kd_loss', {}), dict) else {}

                        sur_cfg = SurCfg()
                        for key, value in sur_cfg_dict.items():
                            if hasattr(sur_cfg, key):
                                setattr(sur_cfg, key, value)
                        if 'z_ch' not in sur_cfg_dict:
                            sur_cfg.z_ch = int(self.model_config.get('vae_config', {}).get('latent_channels', sur_cfg.z_ch))
                        if 'vnums' not in sur_cfg_dict:
                            tev_vnums = 4
                            tev_cfg = self.model_config.get('tevnet_config', {})
                            if isinstance(tev_cfg, dict):
                                tev_vnums = int(tev_cfg.get('vnums', tev_vnums))
                            sur_cfg.vnums = int(tev_vnums)

                        self.latent_surrogate_model = LatentSurrogate(sur_cfg)
                        ckpt = ls_cfg.get('ckpt', None)
                        if not ckpt:
                            raise ValueError("latent_surrogate_config.ckpt is required when enable=true")
                        load_info = load_module_checkpoint(
                            self.latent_surrogate_model,
                            ckpt,
                            strict=False,
                            strip_prefixes=('model.', 'module.', 'student.'),
                        )
                        state_source = str(load_info.get("state_source", "unknown"))
                        print(
                            f"[INFO] Loaded latent surrogate from {ckpt} "
                            f"(source={state_source}, state_tensors={load_info['num_state_tensors']}, "
                            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})"
                        )
                        if len(load_info["missing_keys"]) > 0:
                            print(
                                "[WARN] Latent surrogate checkpoint was only partially loaded. "
                                "Please verify surrogate architecture and checkpoint compatibility."
                            )
                        self.latent_surrogate_model.eval()
                        for parameter in self.latent_surrogate_model.parameters():
                            parameter.requires_grad = False

                        kd_cfg = KDcfg()
                        for key, value in kd_cfg_dict.items():
                            if hasattr(kd_cfg, key):
                                setattr(kd_cfg, key, value)
                        # Main SiT training runs value distillation only by default.
                        if 'grad_align_prob' not in kd_cfg_dict:
                            kd_cfg.grad_align_prob = 0.0
                        self.latent_surrogate_loss_obj = KDLoss(kd_cfg, vnums=int(sur_cfg.vnums))
                        self.latent_surrogate_weight = float(ls_cfg.get('weight', 1.0))
                        self.latent_surrogate_t_min = float(ls_cfg.get('t_min', 0.7))
                        self.use_latent_surrogate_loss = self.latent_surrogate_weight > 0

                        disable_online_tev = bool(ls_cfg.get('disable_online_tev', True))
                        if self.use_latent_surrogate_loss and disable_online_tev:
                            self.use_tev_loss = False
                            self.tevnet_stage_model = None
                            self.tev_loss_obj = None
                            print("[INFO] Latent surrogate enabled, disable online TeV decode loss.")
                    except Exception as e:
                        print(f"[WARN] Latent surrogate init failed, disable surrogate loss. Reason: {e}")
                        self.use_latent_surrogate_loss = False
                        self.latent_surrogate_model = None
                        self.latent_surrogate_loss_obj = None

            self.physics_config = None
            self.physics_train_cfg = {}
            self.use_physics_proxy = False
            self.phys_teacher = None
            self.p_cpen = None
            self.phys_sur = None
            self.phys_decoder = None
            if 'physics_config' in self.model_config and isinstance(self.model_config['physics_config'], dict):
                phys_cfg = self.model_config['physics_config']
                if phys_cfg.get('enabled', False):
                    teacher_cfg = phys_cfg.get('teacher', {}) if isinstance(phys_cfg.get('teacher', {}), dict) else {}
                    proxy_cfg = phys_cfg.get('proxy', {}) if isinstance(phys_cfg.get('proxy', {}), dict) else {}
                    self.physics_train_cfg = phys_cfg.get('train', {}) if isinstance(phys_cfg.get('train', {}), dict) else {}
                    p_map_ch = int(proxy_cfg.get('p_map_ch', 64))
                    p_token_num = int(proxy_cfg.get('p_token_num', 4))
                    p_token_dim = int(proxy_cfg.get('p_token_dim', 64))
                    a_low_range = tuple(teacher_cfg.get('a_low_range', [0.8, 1.2]))
                    allow_partial_physics_ckpt = bool(phys_cfg.get('allow_partial_physics_ckpt', False))

                    self.phys_teacher = TeR_B(
                        smp_model=str(teacher_cfg.get('smp_model', 'Unet')),
                        smp_encoder=str(teacher_cfg.get('smp_encoder', 'resnet18')),
                        smp_encoder_weights=teacher_cfg.get('smp_encoder_weights', None),
                        vnums=int(teacher_cfg.get('vnums', 4)),
                        erme_kernel=int(teacher_cfg.get('erme_kernel', 5)),
                        lambda_env_init=float(teacher_cfg.get('lambda_env_init', 0.1)),
                        a_low_range=(float(a_low_range[0]), float(a_low_range[1])),
                    )
                    self.p_cpen = PhysCPEN(
                        in_ch=5,
                        base_ch=p_map_ch,
                        p_token_num=p_token_num,
                        p_token_dim=p_token_dim,
                    )
                    self.phys_sur = PhysSur(
                        in_ch=int(self.model_config.get('vae_config', {}).get('latent_channels', 4)),
                        base_ch=p_map_ch,
                        p_token_num=p_token_num,
                        p_token_dim=p_token_dim,
                    )
                    self.phys_decoder = PhysDecoder(
                        map_ch=p_map_ch,
                        p_token_num=p_token_num,
                        p_token_dim=p_token_dim,
                        a_low_range=(float(a_low_range[0]), float(a_low_range[1])),
                    )

                    teacher_ckpt = str(teacher_cfg.get('ckpt', ''))
                    pcpen_ckpt = str(proxy_cfg.get('pcpen_ckpt', ''))
                    physdecoder_ckpt = str(proxy_cfg.get('physdecoder_ckpt', ''))
                    physsur_ckpt = str(proxy_cfg.get('physsur_ckpt', ''))
                    if not teacher_ckpt or not pcpen_ckpt or not physdecoder_ckpt or not physsur_ckpt:
                        raise ValueError('physics_config requires teacher/pcpen/physdecoder/physsur checkpoints.')

                    self._load_named_module_checkpoint(
                        self.phys_teacher,
                        teacher_ckpt,
                        'TeR_B',
                        strict=(not allow_partial_physics_ckpt),
                    )
                    self._load_named_module_checkpoint(
                        self.p_cpen,
                        pcpen_ckpt,
                        'PhysCPEN',
                        strict=(not allow_partial_physics_ckpt),
                    )
                    self._load_named_module_checkpoint(
                        self.phys_decoder,
                        physdecoder_ckpt,
                        'PhysDecoder',
                        strict=(not allow_partial_physics_ckpt),
                    )
                    self._load_named_module_checkpoint(
                        self.phys_sur,
                        physsur_ckpt,
                        'PhysSur',
                        strict=(not allow_partial_physics_ckpt),
                    )

                    set_requires_grad(self.phys_teacher, not bool(teacher_cfg.get('freeze', True)))
                    set_requires_grad(self.p_cpen, not bool(proxy_cfg.get('freeze_pcpen', True)))
                    set_requires_grad(self.phys_decoder, not bool(proxy_cfg.get('freeze_physdecoder', True)))
                    set_requires_grad(self.phys_sur, not bool(proxy_cfg.get('freeze_physsur', True)))

                    self.physics_config = phys_cfg
                    self.use_physics_proxy = True
                    print('[INFO] Physics proxy branch enabled.')
            self.output_type = "normal"
        else:
            self.output_type = "normal"
            raise NotImplementedError()

        # For validation in Lightning v2.0.0
        self.log_img_first_iter_train = False
        self.log_img_first_iter_val = False
        self.log_img_first_iter_test = False

    # the forward pass of the lightning model
    def forward(self, RGB, dataset_idx=None, Thermal=None, Phys=None, Training=False):
        # Pad RGB and Thermal
        RGB_padded = self.pad_to_divisble(RGB, multiple=self.model.divisible)
        if Thermal is not None:
            Thermal_padded = self.pad_to_divisble(Thermal, multiple=self.model.divisible)
        else:
            Thermal_padded = None
        if Phys is not None:
            Phys_padded = self.pad_to_divisble(Phys, multiple=self.model.divisible)
        else:
            Phys_padded = None

        dataset_idx_model = dataset_idx
        if (
            dataset_idx_model is not None
            and self.model_arch == "sit"
            and getattr(self, "physics_config", None) is not None
            and bool(self.physics_config.get("collapse_style", False))
        ):
            dataset_idx_model = torch.zeros_like(dataset_idx_model)

        if self.model_arch == "pix2pix" or self.model_arch == "pix2pixHD":
            Pred_Thermal_padded = self.model(RGB_padded)
            results = Pred_Thermal_padded
        elif self.model_arch == "cyclegan":
            if Training:
                assert Thermal is not None
                Thermal_pred_padded = self.model(RGB_padded)
                RGB_rec_padded = self.model_A(Thermal_pred_padded)
                RGB_pred_padded = self.model_A(Thermal_padded)
                Thermal_rec_padded = self.model(RGB_pred_padded)
                results = [Thermal_pred_padded, RGB_rec_padded, Thermal_rec_padded, RGB_pred_padded]
            else: # Evaluation
                Pred_Thermal_padded = self.model(RGB_padded)
                results = Pred_Thermal_padded
        elif self.model_arch == "vqgan":
            if Training:
                Pred_Thermal_padded, qloss = self.model(RGB_padded)
                results = [Pred_Thermal_padded, qloss]
            else:
                Pred_Thermal_padded, _ = self.model(RGB_padded)
                results = Pred_Thermal_padded
        elif self.model_arch == "klvae":
            if Training:
                posterior = self.model.encode(Thermal_padded).latent_dist
                z = posterior.sample(generator=None)
                qloss = posterior.kl()
                Pred_Thermal_padded = self.model.decode(z).sample
                self.latent_list = torch.cat([self.latent_list, z.detach().flatten().cpu()])
                results = [Pred_Thermal_padded, qloss]
            else:
                Pred_Thermal_padded = self.model(Thermal_padded).sample
                results = Pred_Thermal_padded
        elif self.model_arch == "klvae_RGB":
            if Training:
                posterior = self.model.encode(RGB_padded).latent_dist
                z = posterior.sample(generator=None)
                qloss = posterior.kl()
                Pred_RGB_padded = self.model.decode(z).sample
                self.latent_list = torch.cat([self.latent_list, z.detach().flatten().cpu()])
                results = [Pred_RGB_padded, qloss]
            else:
                Pred_RGB_padded = self.model(RGB_padded).sample
                results = Pred_RGB_padded
        elif self.model_arch == "dcae":
            if Training:
                z = self.model.encode(Thermal_padded, return_dict=False)[0]
                Pred_Thermal_padded = self.model.decode(z, return_dict=False)[0]
                self.latent_list = torch.cat([self.latent_list, z.detach().flatten().cpu()])
                results = Pred_Thermal_padded
            else:
                Pred_Thermal_padded = self.model(Thermal_padded).sample
                results = Pred_Thermal_padded
        elif self.model_arch == "phys_vae_r":
            thermal_01 = self._thermal_to_zero_one(Thermal_padded)
            sample_posterior = bool(self.model_config.get("sample_posterior", Training))
            detach_base_for_residual = bool(self.model_config.get("detach_base_for_residual", True))
            outputs = self.model(
                thermal_01,
                sample=sample_posterior if Training else False,
                detach_base_for_residual=detach_base_for_residual,
            )
            self.latent_list = torch.cat([self.latent_list, outputs["z_joint"].detach().flatten().cpu()])
            if Training:
                outputs["pred_image"] = self._thermal_from_zero_one(outputs["x_hat"])
                results = outputs
            else:
                results = self._thermal_from_zero_one(outputs["x_hat"])
        elif self.model_arch == "phys_factor_vae":
            thermal_01 = self._thermal_to_zero_one(Thermal_padded)
            sample_posterior = bool(self.model_config.get("sample_posterior", Training))
            outputs = self.model(
                thermal_01,
                sample=sample_posterior if Training else False,
                recompose_mode=None if Training else self.psl_recompose_mode,
            )
            self.latent_list = torch.cat([self.latent_list, outputs["z_phys"].detach().flatten().cpu()])
            if Training:
                outputs["pred_image"] = self._thermal_from_zero_one(outputs["y_hat"])
                results = outputs
            else:
                results = self._thermal_from_zero_one(outputs["y_hat"])
        elif self.model_arch == "sit":
            if Training:
                if self.model.repa:
                    with torch.no_grad():
                        zs = []
                        if self.repa_input == "RGB":
                            RGB_norm = self.preprocess_raw_image(RGB_padded, enc_type="dinov2")
                            z = self.repa_encoder.forward_features(RGB_norm)
                        elif self.repa_input == "Thermal":
                            Thermal_norm = self.preprocess_raw_image(Thermal_padded, enc_type="dinov2")
                            z = self.repa_encoder.forward_features(Thermal_norm)
                        z = z['x_norm_patchtokens']
                        zs.append(z)
                else:
                    zs = None
                if 'vae_mixed_precision' in self.model_config and self.model_config['vae_mixed_precision']:
                    with torch.autocast("cuda", torch.float16):
                        input_latent = self.generate_latent_train(Thermal_padded, RGB_padded)
                else:
                    input_latent = self.generate_latent_train(Thermal_padded, RGB_padded)
                x, x_RGB = input_latent[0], input_latent[1]
                x_phys = None
                if Phys_padded is not None:
                    x_phys = self.generate_phys_latent(Phys_padded)
                if self.calculate_stats:
                    self.latent_list = torch.cat([self.latent_list, x.detach().cpu()])
                    self.latent_RGB_list = torch.cat([self.latent_RGB_list, x_RGB.detach().cpu()])
                model_kwargs = dict(y=dataset_idx_model, x_RGB=x_RGB)
                if x_phys is not None:
                    model_kwargs["x_phys"] = x_phys
                loss_dict = self.transport.training_losses(self.model, x, model_kwargs, zs)

                # Base flow matching loss is a per-sample vector.
                results = loss_dict["loss"]

                # Optional latent surrogate loss (P3).
                if getattr(self, "use_latent_surrogate_loss", False):
                    try:
                        results = results + self.compute_latent_surrogate_aux_loss(
                            loss_dict=loss_dict,
                            gt_latent=x,
                        )
                    except Exception as e:
                        print(f"[WARN] Latent surrogate aux loss skipped due to error: {type(e).__name__}: {e}")

                # Optional TeV/physics loss on an estimated x1 (data) reconstructed from velocity.
                if getattr(self, "use_tev_loss", False) and (Thermal_padded is not None):
                    try:
                        results = results + self.compute_tev_aux_loss(
                            loss_dict=loss_dict,
                            gt_thermal=Thermal_padded,
                        )
                    except Exception as e:
                        # Don't crash training if TeV loss is misconfigured; fail loud in logs.
                        print(f"[WARN] TeV aux loss skipped due to error: {type(e).__name__}: {e}")

                if getattr(self, "use_physics_proxy", False) and (Thermal_padded is not None):
                    try:
                        z_clean_hat, _ = self._estimate_x1_from_loss_dict(loss_dict)
                        phys_losses = self.compute_physics_losses(
                            z_clean_hat=z_clean_hat,
                            gt_thermal=Thermal_padded,
                            t=loss_dict["t"],
                        )
                        loss_total = results.mean() + phys_losses["loss_weighted"]
                        if self.kl_training and input_latent[2] is not None:
                            loss_total = loss_total + self.model_config['kl_weight'] * input_latent[2].mean()
                        log_dict = {
                            "loss_flow": results.mean().detach(),
                            **phys_losses["log_dict"],
                        }
                        results = {
                            "loss_total": loss_total,
                            "flow_loss": results,
                            "u_hat": loss_dict["pred"],
                            "xt": loss_dict["xt"],
                            "t": loss_dict["t"],
                            "x_gt": x,
                            "x_rgb": x_RGB,
                            "z_clean_hat": z_clean_hat,
                            "loss_phys_proxy": phys_losses["loss_phys_proxy"].detach(),
                            "loss_phys_proj": phys_losses["loss_phys_proj"].detach(),
                            "loss_phys_img": phys_losses["loss_phys_img"].detach(),
                            "loss_phys_closure": phys_losses["loss_phys_closure"].detach(),
                            "log_dict": log_dict,
                        }
                    except Exception as e:
                        print(f"[WARN] Physics proxy loss skipped due to error: {type(e).__name__}: {e}")
                        results = results.mean()
                        if self.kl_training and input_latent[2] is not None:
                            results = results + self.model_config['kl_weight'] * input_latent[2].mean()
                elif self.kl_training:
                    results += self.model_config['kl_weight'] * input_latent[2] # kl loss
            else:
                if self.model_config['vae_model'] == "klvae":
                    latent_size = RGB_padded.shape[2] // int(getattr(self, "thermal_downsample_factor", 8)), RGB_padded.shape[3] // int(getattr(self, "thermal_downsample_factor", 8))
                    zs = torch.randn(RGB.shape[0], int(getattr(self, "thermal_latent_channels", 4)), latent_size[0], latent_size[1], device=RGB_padded.device)
                elif self.model_config['vae_model'] == "dcae":
                    if self.model_config['divisible'] == 32:
                        latent_size = RGB_padded.shape[2]//32, RGB_padded.shape[3]//32
                        zs = torch.randn(RGB.shape[0], 32, latent_size[0], latent_size[1], device=RGB_padded.device)
                    elif self.model_config['divisible'] == 64:
                        latent_size = RGB_padded.shape[2]//64, RGB_padded.shape[3]//64
                        zs = torch.randn(RGB.shape[0], 128, latent_size[0], latent_size[1], device=RGB_padded.device)
                elif self.model_config['vae_model'] == "phys_vae_r":
                    factor = int(getattr(self, "thermal_downsample_factor", 8))
                    latent_size = RGB_padded.shape[2] // factor, RGB_padded.shape[3] // factor
                    zs = torch.randn(
                        RGB.shape[0],
                        int(getattr(self, "thermal_latent_channels", self.thermal_vae.joint_channels)),
                        latent_size[0],
                        latent_size[1],
                        device=RGB_padded.device,
                    )
                elif self.model_config['vae_model'] == "phys_factor_vae":
                    factor = int(getattr(self, "thermal_downsample_factor", 8))
                    latent_size = RGB_padded.shape[2] // factor, RGB_padded.shape[3] // factor
                    zs = torch.randn(
                        RGB.shape[0],
                        int(getattr(self, "thermal_latent_channels", self.thermal_vae.latent_channels)),
                        latent_size[0],
                        latent_size[1],
                        device=RGB_padded.device,
                    )
                else:
                    raise NotImplementedError()
                ys = dataset_idx_model
                sample_fn = self.sampler.sample_ode()
                with torch.no_grad():
                    x_RGB = self.encode_rgb_latent(RGB_padded)
                    if self.RGB_normalizer is not None:
                        x_RGB = x_RGB.mul_(self.RGB_normalizer)
                    x_phys = None
                    if Phys_padded is not None:
                        x_phys = self.generate_phys_latent(Phys_padded)
                    if self.use_cfg:
                        cfg_condition = str(self.model_config.get('cfg_condition', 'label')).lower()
                        zs = torch.cat([zs, zs], 0)
                        if cfg_condition == "label":
                            y_null = torch.tensor([1000] * len(ys), device=ys.device)
                            ys = torch.cat([ys, y_null], 0)
                            x_RGB = torch.cat([x_RGB, x_RGB], 0)
                        elif cfg_condition == "rgb":
                            ys = torch.cat([ys, ys], 0)
                            x_RGB = torch.cat([x_RGB, torch.zeros_like(x_RGB)], 0)
                        elif cfg_condition == "both":
                            y_null = torch.tensor([1000] * len(ys), device=ys.device)
                            ys = torch.cat([ys, y_null], 0)
                            x_RGB = torch.cat([x_RGB, torch.zeros_like(x_RGB)], 0)
                        else:
                            raise ValueError(f"Unknown cfg_condition: {cfg_condition}. Use one of: label, rgb, both.")
                        if x_phys is not None:
                            x_phys = torch.cat([x_phys, x_phys], 0)
                        sample_model_kwargs = dict(y=ys, x_RGB=x_RGB, cfg_scale=self.model_config['cfg_scale'])
                        if x_phys is not None:
                            sample_model_kwargs["x_phys"] = x_phys
                        if self.validation_type == "ema":
                            model_eval = self.ema.forward_with_cfg
                        elif self.validation_type == 'current':
                            model_eval = self.model.forward_with_cfg
                        else:
                            raise NotImplementedError()
                    else:
                        if 'force_un' in self.model_config and self.model_config['force_un']:
                            y_null = torch.tensor([1000] * len(ys), device=ys.device)
                            sample_model_kwargs = dict(y=y_null, x_RGB=x_RGB)
                        else:
                            sample_model_kwargs = dict(y=ys, x_RGB=x_RGB)
                        if x_phys is not None:
                            sample_model_kwargs["x_phys"] = x_phys
                        if self.validation_type == "ema":
                            model_eval = self.ema.forward
                        elif self.validation_type == 'current':
                            model_eval = self.model.forward
                        else:
                            raise NotImplementedError()
                    samples = sample_fn(zs, model_eval, **sample_model_kwargs)[-1]
                    if self.use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    Pred_Thermal_padded = self.decode_thermal_latent(samples)
                    results = Pred_Thermal_padded
        else:
            raise NotImplementedError()

        # Reverse Pad RGB and Thermal for image-like outputs only.
        if isinstance(results, dict):
            return results
        if type(results) != list and len(results.shape) == 4:
            results = results[:, :, :RGB.shape[2], :RGB.shape[3]]
        else:
            for i, result_padded in enumerate(results):
                if len(result_padded.shape) == 4:
                    results[i] = result_padded[:, :, :RGB.shape[2], :RGB.shape[3]]
        return results
    
    def generate_latent_train(self, Thermal_padded, RGB_padded):
        posterior = None
        with torch.no_grad():
            if self.model_config['vae_model'] == "klvae":
                if self.thermal_vae.in_channels == 3:
                    Thermal_padded = Thermal_padded.repeat(1,3,1,1)
                x = self.thermal_vae.encode(Thermal_padded).latent_dist.sample()
            elif self.model_config['vae_model'] == "dcae":
                x = self.thermal_vae.encode(Thermal_padded).latent
            elif self.model_config['vae_model'] == "phys_vae_r":
                thermal_01 = self._thermal_to_zero_one(Thermal_padded)
                encoded = self.thermal_vae.encode_joint_latent(
                    thermal_01,
                    sample=bool(self.model_config.get("sample_thermal_latent", True)),
                    detach_base_for_residual=bool(self.model_config.get("detach_base_for_residual", True)),
                )
                x = encoded["z_joint"]
                posterior = {
                    "phys": encoded["posterior_phys"],
                    "res": encoded["posterior_res"],
                }
            elif self.model_config['vae_model'] == "phys_factor_vae":
                thermal_01 = self._thermal_to_zero_one(Thermal_padded)
                encoded = self.thermal_vae.encode_from_ir(
                    thermal_01,
                    sample=bool(self.model_config.get("sample_thermal_latent", True)),
                )
                x = encoded["z_phys"]
                posterior = encoded["posterior"]
            if self.thermal_normalizer is not None:
                x = x.mul_(self.thermal_normalizer)
            if hasattr(self, 'latent_cache'):
                for item in x.detach().cpu():
                    self.latent_cache.append(item)
        if self.RGB_encoder_training:
            if self.model_config['vae_model'] == "klvae":
                posterior = self.encode_rgb_posterior(RGB_padded)
                x_RGB = posterior.sample()
            elif self.model_config['vae_model'] == "dcae":
                x_RGB = self.encode_rgb_latent(RGB_padded)
            if self.RGB_normalizer is not None:
                x_RGB = x_RGB.mul_(self.RGB_normalizer)
            if hasattr(self, 'RGB_cache'):
                for item in RGB_padded.detach().cpu():
                    self.RGB_cache.append(item)
        else:
            with torch.no_grad():
                x_RGB = self.encode_rgb_latent(RGB_padded)
                if self.RGB_normalizer is not None:
                    x_RGB = x_RGB.mul_(self.RGB_normalizer)
            if hasattr(self, 'RGB_latent_cache'):
                for item in x_RGB.detach().cpu():
                    self.RGB_latent_cache.append(item)
        if self.kl_training and posterior is not None:
            if isinstance(posterior, dict):
                kl_value = posterior["phys"].kl() + posterior["res"].kl()
            else:
                kl_value = posterior.kl()
        else:
            kl_value = None
        return x, x_RGB, kl_value

    def generate_phys_latent(self, Phys_padded):
        """Encode a physics/TeV map into the same latent space as thermal for spatial conditioning.

        Expect `Phys_padded` to be an image-like tensor in the same range as other inputs.
        """
        with torch.no_grad():
            phys_in = Phys_padded
            if self.model_config['vae_model'] == "klvae":
                if self.thermal_vae.in_channels == 3 and phys_in.shape[1] == 1:
                    phys_in = phys_in.repeat(1, 3, 1, 1)
                x_phys = self.thermal_vae.encode(phys_in).latent_dist.sample()
            elif self.model_config['vae_model'] == "dcae":
                x_phys = self.thermal_vae.encode(phys_in).latent
            elif self.model_config['vae_model'] == "phys_vae_r":
                phys_01 = self._thermal_to_zero_one(phys_in)
                x_phys = self.thermal_vae.encode_joint_latent(
                    phys_01,
                    sample=False,
                    detach_base_for_residual=True,
                )["z_joint"]
            elif self.model_config['vae_model'] == "phys_factor_vae":
                phys_01 = self._thermal_to_zero_one(phys_in)
                x_phys = self.thermal_vae.encode_from_ir(phys_01, sample=False)["z_phys"]
            else:
                raise NotImplementedError()
            if self.thermal_normalizer is not None:
                x_phys = x_phys.mul_(self.thermal_normalizer)
        return x_phys

    def decode_thermal_latent(self, z_latent):
        """Decode a thermal latent (optionally normalized) into image space."""
        z = z_latent
        if self.thermal_normalizer is not None:
            z = z / self.thermal_normalizer
        if self.model_config['vae_model'] == "klvae":
            return self.thermal_vae.decode(z).sample
        elif self.model_config['vae_model'] == "dcae":
            return self.thermal_vae.decode(z, return_dict=False)[0]
        elif self.model_config['vae_model'] == "phys_vae_r":
            return self._thermal_from_zero_one(self.thermal_vae.decode_latents(z_joint=z)["x_hat"])
        elif self.model_config['vae_model'] == "phys_factor_vae":
            return self._thermal_from_zero_one(self.thermal_vae.decode_latents(z, recompose_mode=self.psl_recompose_mode)["y_hat"])
        else:
            raise NotImplementedError()

    def _estimate_x1_from_loss_dict(self, loss_dict):
        v_pred = loss_dict['pred']
        t = loss_dict['t']
        xt = loss_dict['xt']

        batch_size = xt.shape[0]
        t_view = t.view(batch_size, *([1] * (xt.dim() - 1)))
        x1_hat = xt + (1.0 - t_view) * v_pred
        return x1_hat, t

    def compute_latent_surrogate_aux_loss(self, loss_dict, gt_latent):
        """Compute latent-space surrogate loss on a late-time subset only.

        This avoids decode+TeV in the main SiT training graph.
        """
        if (
            (not self.use_latent_surrogate_loss)
            or (self.latent_surrogate_model is None)
            or (self.latent_surrogate_loss_obj is None)
            or (self.latent_surrogate_weight <= 0)
        ):
            return torch.zeros_like(loss_dict['loss'])

        if getattr(self.transport, 'model_type', None) is None:
            return torch.zeros_like(loss_dict['loss'])
        from models.generative_models.sit_networks.transport import ModelType
        if self.transport.model_type != ModelType.VELOCITY:
            return torch.zeros_like(loss_dict['loss'])

        x1_hat, t = self._estimate_x1_from_loss_dict(loss_dict)
        batch_size = x1_hat.shape[0]
        aux = torch.zeros(batch_size, device=x1_hat.device, dtype=loss_dict['loss'].dtype)

        idx = torch.where(t >= float(self.latent_surrogate_t_min))[0]
        if idx.numel() == 0:
            return aux

        x1_subset = x1_hat.index_select(0, idx)
        gt_subset = gt_latent.index_select(0, idx)

        # Student prediction on generated latent.
        student_out = self.latent_surrogate_model(x1_subset)
        # Detached target on GT latent; no decode/TeV graph involved here.
        with torch.no_grad():
            teacher_proxy = self.latent_surrogate_model(gt_subset)
            target_detached = {
                "feat_dec": teacher_proxy["feat_dec"],
                "feat_deep": teacher_proxy["feat_deep"],
                "phys": teacher_proxy["phys_logits"],
            }

        loss_subset = self.latent_surrogate_loss_obj(
            z=x1_subset,
            stu=student_out,
            tea_detached=target_detached,
            tea_phys_for_grad=None,
        )
        aux[idx] = float(self.latent_surrogate_weight) * loss_subset
        return aux

    def compute_tev_aux_loss(self, loss_dict, gt_thermal):
        """Compute TeVNet auxiliary loss as a per-sample vector.

        We estimate x1 (data latent) from the current (xt, t) and predicted velocity:
            xt = x0 + t * v  =>  x1 = xt + (1 - t) * v
        """
        if (not self.use_tev_loss) or (self.tevnet_stage_model is None) or (self.tev_weight <= 0 and self.tev_rec_weight <= 0):
            return torch.zeros_like(loss_dict['loss'])

        # We only support the velocity parameterization here.
        if getattr(self.transport, 'model_type', None) is None:
            return torch.zeros_like(loss_dict['loss'])
        from models.generative_models.sit_networks.transport import ModelType
        if self.transport.model_type != ModelType.VELOCITY:
            return torch.zeros_like(loss_dict['loss'])

        x1_hat, t = self._estimate_x1_from_loss_dict(loss_dict)

        pred_img = self.decode_thermal_latent(x1_hat)
        pred_img01 = torch.clamp(pred_img * 0.5 + 0.5, 0, 1)
        gt01 = torch.clamp(gt_thermal * 0.5 + 0.5, 0, 1)

        if pred_img01.shape[1] == 1:
            pred_img01 = pred_img01.repeat(1, 3, 1, 1)
        if gt01.shape[1] == 1:
            gt01 = gt01.repeat(1, 3, 1, 1)

        # TeV feature loss
        tev_pred = self.tevnet_stage_model(pred_img01)
        with torch.no_grad():
            tev_gt = self.tevnet_stage_model(gt01)
        loss_tev = ((tev_pred - tev_gt) ** 2).mean(dim=(1, 2, 3))

        # TeV reconstruction loss (optional)
        if self.tev_rec_weight > 0 and self.tev_loss_obj is not None:
            rec = self.tev_loss_obj.rec(tev_pred, pred_img01)
            pred_gray = pred_img01.mean(dim=1, keepdim=True)
            loss_rec = ((rec - pred_gray) ** 2).mean(dim=(1, 2, 3))
        else:
            loss_rec = torch.zeros_like(loss_tev)

        # Mask by time if needed (early-times tend to be noisy and less physical)
        mask = (t <= float(self.tev_t_max)).float()

        return mask * (self.tev_weight * loss_tev + self.tev_rec_weight * loss_rec)
    
    def _load_named_module_checkpoint(self, module, ckpt_path, name, strict=True, strip_prefixes=('model.', 'module.')):
        if not ckpt_path or (not os.path.isfile(ckpt_path)):
            raise FileNotFoundError(f"{name} ckpt not found: {ckpt_path}")
        load_info = load_module_checkpoint(
            module,
            ckpt_path,
            strict=bool(strict),
            strip_prefixes=strip_prefixes,
        )
        print(
            f"[INFO] Loaded {name} from {ckpt_path} "
            f"(strict={bool(strict)}, source={load_info.get('state_source', 'unknown')}, state_tensors={load_info['num_state_tensors']}, "
            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})"
        )

    def _collect_trainable_physics_params(self):
        params = []
        if not getattr(self, 'use_physics_proxy', False):
            return params
        for module in (self.phys_teacher, self.p_cpen, self.phys_sur, self.phys_decoder):
            if module is None:
                continue
            params.extend([parameter for parameter in module.parameters() if parameter.requires_grad])
        return params

    def _physics_loss_weights(self):
        cfg = self.physics_train_cfg if isinstance(self.physics_train_cfg, dict) else {}
        return {
            'proxy': ramp_weight(cfg.get('lambda_proxy', 0.20), self.current_epoch, cfg.get('warmup_epochs_proxy', 30)),
            'proj': ramp_weight(cfg.get('lambda_proj', 0.10), self.current_epoch, cfg.get('warmup_epochs_proj', 60)),
            'img': ramp_weight(cfg.get('lambda_img', 0.05), self.current_epoch, cfg.get('warmup_epochs_img', 90)),
            'closure': ramp_weight(cfg.get('lambda_closure', 0.02), self.current_epoch, cfg.get('warmup_epochs_closure', 120)),
        }

    def compute_physics_losses(self, z_clean_hat, gt_thermal, t):
        zero = z_clean_hat.new_tensor(0.0)
        if (
            (not getattr(self, 'use_physics_proxy', False))
            or self.phys_teacher is None
            or self.p_cpen is None
            or self.phys_sur is None
            or self.phys_decoder is None
            or gt_thermal is None
        ):
            return {
                'loss_phys_proxy': zero,
                'loss_phys_proj': zero,
                'loss_phys_img': zero,
                'loss_phys_closure': zero,
                'loss_weighted': zero,
                'log_dict': {},
            }

        gt_01 = torch.clamp(gt_thermal * 0.5 + 0.5, 0.0, 1.0)
        with torch.no_grad():
            teacher_gt = self.phys_teacher(gt_01)
            targets32 = build_lowres_targets(teacher_gt)
            p_gt_map, p_gt_token = self.p_cpen(targets32['Y_phys32'])

        p_hat_map, p_hat_token = self.phys_sur(z_clean_hat)
        pred32 = self.phys_decoder(p_hat_map, p_hat_token)

        mask = (t >= float(self.physics_train_cfg.get('t_min_phys', 0.4))).float()
        mask_denom = mask.sum().clamp_min(1.0)

        def _masked_mean(sample_loss):
            return (sample_loss * mask).sum() / mask_denom

        loss_proxy_sample = (
            l1_per_sample(p_hat_map, p_gt_map)
            + l1_per_sample(p_hat_token, p_gt_token)
            + 0.1 * cosine_per_sample(p_hat_token, p_gt_token)
        )
        loss_proj_sample = (
            l1_per_sample(pred32['e'], targets32['e'])
            + l1_per_sample(pred32['T_rad'], targets32['T_rad'])
            + l1_per_sample(pred32['R_env'], targets32['R_env'])
            + 0.5 * l1_per_sample(pred32['A'], targets32['A'])
            + 0.25 * l1_per_sample(pred32['B_edge'], targets32['B_edge'])
        )
        loss_recomp_sample = (
            l1_per_sample(pred32['S_phys'], targets32['S_phys'])
            + 0.2 * (1.0 - ssim_per_sample(pred32['S_phys'], targets32['S_phys']))
        )
        loss_img_gt_sample = (
            l1_per_sample(pred32['S_phys'], targets32['S_01'])
            + 0.2 * (1.0 - ssim_per_sample(pred32['S_phys'], targets32['S_01']))
        )
        loss_img_sample = loss_recomp_sample + 0.5 * loss_img_gt_sample

        loss_proxy = _masked_mean(loss_proxy_sample)
        loss_proj = _masked_mean(loss_proj_sample)
        loss_recomp = _masked_mean(loss_recomp_sample)
        loss_img_gt = _masked_mean(loss_img_gt_sample)
        loss_img = _masked_mean(loss_img_sample)

        loss_closure = zero
        sparse_every = max(1, int(self.physics_train_cfg.get('sparse_every', 8)))
        if int(self.global_step) % sparse_every == 0:
            pred_img = self.decode_thermal_latent(z_clean_hat)
            pred_01 = torch.clamp(pred_img * 0.5 + 0.5, 0.0, 1.0)
            teacher_pred = self.phys_teacher(pred_01)
            loss_closure_sample = (
                l1_per_sample(teacher_pred['e'], teacher_gt['e'].detach())
                + l1_per_sample(teacher_pred['R_env'], teacher_gt['R_env'].detach())
                + 0.5 * l1_per_sample(teacher_pred['B_edge'], teacher_gt['B_edge'].detach())
            )
            loss_closure = _masked_mean(loss_closure_sample)

        weights = self._physics_loss_weights()
        weighted = (
            weights['proxy'] * loss_proxy
            + weights['proj'] * loss_proj
            + weights['img'] * loss_img
            + weights['closure'] * loss_closure
        )

        log_dict = {
            'loss_phys_proxy': loss_proxy.detach(),
            'loss_phys_proj': loss_proj.detach(),
            'loss_phys_recomp': loss_recomp.detach(),
            'loss_phys_img_gt': loss_img_gt.detach(),
            'loss_phys_img': loss_img.detach(),
            'loss_phys_closure': loss_closure.detach(),
            'w_phys_proxy': zero.new_tensor(weights['proxy']),
            'w_phys_proj': zero.new_tensor(weights['proj']),
            'w_phys_img': zero.new_tensor(weights['img']),
            'w_phys_closure': zero.new_tensor(weights['closure']),
            'phys_lambda_env': teacher_gt['lambda_env'].detach().mean(),
            'phys_edge_mean': normalize_01(sobel_mag(gt_01)).detach().mean(),
            'phys_tv_a': tv_loss(teacher_gt['A'].detach()),
            'phys_tv_r': tv_loss(teacher_gt['R_env'].detach()),
            'phys_tv_e': tv_weighted(teacher_gt['e'].detach(), 1.0 - teacher_gt['B_edge'].detach()),
        }
        return {
            'loss_phys_proxy': loss_proxy,
            'loss_phys_proj': loss_proj,
            'loss_phys_img': loss_img,
            'loss_phys_closure': loss_closure,
            'loss_weighted': weighted,
            'log_dict': log_dict,
        }

    def _contains_nonfinite(self, item):
        if torch.is_tensor(item):
            return (not torch.isfinite(item).all().item())
        if isinstance(item, dict):
            return any(self._contains_nonfinite(value) for value in item.values())
        if isinstance(item, (list, tuple)):
            return any(self._contains_nonfinite(value) for value in item)
        return False

    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            opt = torch.optim.SGD
        elif self.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW
        elif self.optimizer.lower() == "adam":
            opt = torch.optim.Adam
        else:
            raise NotImplementedError("Optimizer name is unavailable")
        
        optimizers = []
        if self.model_arch == "pix2pix" or self.model_arch == "pix2pixHD":
            optimizers.append(opt(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
            optimizers.append(opt(
                self.discriminator.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
        elif self.model_arch == "cyclegan":
            optimizers.append(opt(
                itertools.chain(self.model_A.parameters(), self.model.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
            optimizers.append(opt(
                itertools.chain(self.discriminator_A.parameters(), self.discriminator_B.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            ))
        elif self.model_arch == "vqgan" or self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae":
            if self.training_stage == "full":
                optimizers.append(opt(
                    self.model.parameters(),
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                ))
                if self.model_arch == "vqgan":
                    optimizers.append(opt(
                        self.loss_fn.discriminator.parameters(),
                        lr=self.lr, 
                        weight_decay=self.weight_decay
                    ))
            elif self.training_stage == "mid" and self.model_arch == "dcae":
                optimizers.append(opt(
                    [{'params': self.model.encoder.conv_out.parameters()}, {'params': self.model.decoder.conv_in.parameters()}],
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                ))
            elif self.training_stage == "last" and (self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae"):
                if self.model_arch == "klvae" or self.model_arch == "klvae_RGB":
                    optimizers.append(opt(
                        [{'params': self.model.decoder.up_blocks[3].parameters()}, {'params': self.model.decoder.conv_norm_out.parameters()}, {'params': self.model.decoder.conv_out.parameters()}],
                        lr=self.lr, 
                        weight_decay=self.weight_decay
                    ))
                elif self.model_arch == "dcae":
                    optimizers.append(opt(
                        [{'params': self.model.decoder.up_blocks[0].parameters()}, {'params': self.model.decoder.norm_out.parameters()}, {'params': self.model.decoder.conv_out.parameters()}],
                        lr=self.lr, 
                        weight_decay=self.weight_decay
                    ))
                optimizers.append(opt(
                    self.loss_fn.discriminator.parameters(),
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                ))
            else:
                raise NotImplementedError()
        elif self.model_arch == "phys_vae_r":
            trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if len(trainable_params) == 0:
                raise RuntimeError("No trainable parameters found for PHYS-VAE-R. Check training_stage.")
            optimizers.append(opt(
                trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            ))
        elif self.model_arch == "phys_factor_vae":
            optimizers.append(opt(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            ))
        elif self.model_arch == "sit":
            physics_params = self._collect_trainable_physics_params()
            if self.style_finetuning:
                sit_params = list(self.model.y_embedder.parameters())
            elif self.RGB_encoder_training:
                sit_params = list(self.model.parameters()) + list(self.RGB_vae.parameters())
            else:
                sit_params = list(self.model.parameters())
            if physics_params:
                sit_params = sit_params + physics_params
            optimizers.append(opt(
                sit_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            ))
        else:
            raise NotImplementedError()

        schedulers = []
        for optimizer in optimizers:
            if self.lr_sched.lower() == 'multistep':
                schedulers.append(lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma']))
            elif self.lr_sched.lower() == 'cosine':
                schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max']))
            elif self.lr_sched.lower() == 'linear':
                schedulers.append(lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_sched_args['start_factor'],
                    end_factor=self.lr_sched_args['end_factor'],
                    total_iters=self.lr_sched_args['total_iters']
                ))
            

        return optimizers, schedulers
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, Pred, Thermal, RGB, loss_type=None):
        if loss_type == None:
            loss = self.loss_fn(Pred, Thermal)
        elif loss_type == "G_pix2pix":
            fake_AB = torch.cat((RGB, Pred), 1)
            pred_fake = self.discriminator(fake_AB)
            loss = self.loss_fn_GAN(pred_fake, True)
            loss += self.loss_config["G_loss_lambda"] * self.loss_fn_L1(Pred, Thermal)
        elif loss_type == "D_pix2pix" or loss_type == "D_pix2pixHD":
            fake_AB = torch.cat((RGB, Pred.detach()), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.discriminator(fake_AB)
            loss = self.loss_fn_GAN(pred_fake, False)
            real_AB = torch.cat((RGB, Thermal), 1)
            pred_real = self.discriminator(real_AB)
            loss += self.loss_fn_GAN(pred_real, True)
            loss = 0.5 * loss
        elif loss_type == "G_pix2pixHD":
            fake_AB = torch.cat((RGB, Pred), 1)
            pred_fake = self.discriminator(fake_AB)
            loss = self.loss_fn_GAN(pred_fake, True)
            real_AB = torch.cat((RGB, Thermal), 1)
            pred_real = self.discriminator(real_AB)
            feat_weights = 4.0 / (self.model_config["n_layers_D"] + 1)
            D_weights = 1.0 / self.model_config["num_D"]
            loss_G_GAN_Feat = 0
            for i in range(self.model_config["num_D"]):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.loss_fn_L1(pred_fake[i][j], pred_real[i][j].detach()) * self.loss_config["G_loss_lambda"]
            loss += loss_G_GAN_Feat
        elif loss_type == "G_cyclegan":
            Pred_Thermal, Rec_RGB, Rec_Thermal, Pred_RGB = Pred
            loss = self.loss_fn_GAN(self.discriminator_B(Pred_Thermal), True)
            loss += self.loss_fn_GAN(self.discriminator_A(Pred_RGB), True)
            loss += self.loss_config["G_loss_lambda_Thermal"] * self.loss_fn_L1(Rec_Thermal, Thermal)
            loss += self.loss_config["G_loss_lambda_RGB"] * self.loss_fn_L1(Rec_RGB, RGB)
        elif loss_type == "D_cyclegan":
            Pred_Thermal, Rec_RGB, Rec_Thermal, Pred_RGB = Pred
            loss = self.loss_fn_GAN(self.discriminator_B(Pred_Thermal.detach()), False)
            loss += self.loss_fn_GAN(self.discriminator_B(Thermal), True)
            loss += self.loss_fn_GAN(self.discriminator_A(Pred_RGB.detach()), False)
            loss += self.loss_fn_GAN(self.discriminator_A(RGB), True)
            loss = 0.5 * loss
        elif loss_type == "G_vqgan":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, cond=RGB, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_vqgan":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, cond=RGB, split="train")
            self.log_dict(log_dict)
        elif loss_type == "G_klvae":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_klvae":
            Pred_Thermal, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "G_klvae_RGB":
            Pred_RGB, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, RGB, Pred_RGB, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_klvae_RGB":
            Pred_RGB, Pred_q_loss = Pred
            loss, log_dict = self.loss_fn(Pred_q_loss, RGB, Pred_RGB, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "G_dcae":
            Pred_Thermal = Pred
            if hasattr(self.model.decoder.conv_out, "weight"):
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            else:
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 0, self.global_step, last_layer=self.model.decoder.conv_out.conv.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "D_dcae":
            Pred_Thermal = Pred
            if hasattr(self.model.decoder.conv_out, "weight"):
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.weight, split="train")
            else:
                loss, log_dict = self.loss_fn(Thermal, Pred_Thermal, 1, self.global_step, last_layer=self.model.decoder.conv_out.conv.weight, split="train")
            self.log_dict(log_dict)
        elif loss_type == "Diff_sit":
            if isinstance(Pred, dict):
                loss = Pred['loss_total']
                if isinstance(Pred.get('log_dict', None), dict):
                    for key, value in Pred['log_dict'].items():
                        if torch.is_tensor(value):
                            self.log(key, value, logger=True, prog_bar=False, sync_dist=True)
            else:
                loss = Pred.mean()
        elif loss_type == "G_phys_vae_r":
            if not isinstance(Pred, dict):
                raise RuntimeError("G_phys_vae_r expects forward outputs as a dict.")
            thermal_01 = self._thermal_to_zero_one(Thermal)
            loss_dict = self.phys_vae_r_loss(Pred, thermal_01, stage=self.phys_vae_r_stage)
            log_dict = {f"phys_vae_r/{key}": value for key, value in loss_dict.items() if torch.is_tensor(value)}
            self.log_dict(log_dict, logger=True, prog_bar=False, sync_dist=True)
            loss = loss_dict["loss_total"]
        elif loss_type in {"G_psl_vae", "G_phys_factor_vae"}:
            if not isinstance(Pred, dict):
                raise RuntimeError("G_psl_vae expects forward outputs as a dict.")
            thermal_01 = self._thermal_to_zero_one(Thermal)
            loss_dict = self.phys_factor_vae_loss(Pred, thermal_01)
            log_dict = {f"psl_vae/{key}": value for key, value in loss_dict.items() if torch.is_tensor(value)}
            self.log_dict(log_dict, logger=True, prog_bar=False, sync_dist=True)
            loss = loss_dict["loss_total"]
        else:
            raise NotImplementedError(f"{loss_type} not found")
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx, vis_num=4):
        RGB_list = batch[0]
        Thermal_list = batch[1]
        dataset_idx_list = batch[2]
        Phys_list = batch[3] if len(batch) > 3 else None

        thermal_input = self.model_arch == "cyclegan" or self.model_arch == "klvae" or self.model_arch == "dcae" or self.model_arch == "phys_vae_r" or self.model_arch == "phys_factor_vae" or self.model_arch == "sit"
        Pred_list = self(RGB_list, dataset_idx_list, Thermal=Thermal_list if thermal_input else None, Phys=Phys_list, Training=True)

        find_nan = self._contains_nonfinite(Pred_list)
        if find_nan:
            print('NaNs in Pred_list. Skip it')
            if self.fail_on_nan:
                raise RuntimeError(f"NaNs in model output at batch_idx={batch_idx}.")
            return {}

        # Upload pred and GT for vis (only when image logger is available).
        if not self.log_img_first_iter_train:
            logger_obj = getattr(self, "logger", None)
            has_image_logger = False
            if logger_obj is not None:
                if hasattr(logger_obj, "log_image"):
                    has_image_logger = True
                elif hasattr(logger_obj, "loggers"):
                    has_image_logger = any(hasattr(item, "log_image") for item in logger_obj.loggers)

            if has_image_logger:
                RGB_list_vis = torch.clamp(RGB_list.detach().float().cpu() * 0.5 + 0.5, 0, 1)
                Thermal_list_vis = torch.clamp(Thermal_list.detach().float().cpu() * 0.5 + 0.5, 0, 1)
                if isinstance(Pred_list, dict):
                    pred_for_vis = Pred_list.get("pred_image", None)
                else:
                    pred_for_vis = Pred_list[0] if type(Pred_list) == list else Pred_list
                if isinstance(pred_for_vis, torch.Tensor) and pred_for_vis.ndim == 4:
                    Pred_list_vis = torch.clamp(pred_for_vis.detach().float().cpu() * 0.5 + 0.5, 0, 1)
                    self.vis_eval_image(RGB_list_vis, Thermal_list_vis, Pred_list_vis, vis_num, "mixed", 'train', image_norm="none")

        if self.model_arch == 'pix2pix' or \
           self.model_arch == "cyclegan" or \
           self.model_arch == "pix2pixHD" or \
           self.model_arch == "vqgan" or \
           self.model_arch == "klvae" or \
           self.model_arch == "klvae_RGB" or \
           self.model_arch == "dcae":
            # GAN training
            # Train G
            if not ((self.training_stage == "full" or self.training_stage == "mid") and (self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae")):
                opt_g, opt_d = self.optimizers()
            else:
                opt_g = self.optimizers()
            self.toggle_optimizer(opt_g)
            loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, f"G_{self.model_arch}")
            self.log('loss_G', loss.item(), logger=True, prog_bar=True, sync_dist=True)
            self.manual_backward(loss)
            if batch_idx % self.gradient_accumulation == 0:
                opt_g.step()
                opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)

            # Train D
            # DCAE and KLVAE does not use D for full and mid
            if not ((self.training_stage == "full" or self.training_stage == "mid") and (self.model_arch == "klvae" or self.model_arch == "klvae_RGB" or self.model_arch == "dcae")):
                self.toggle_optimizer(opt_d)
                loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, f"D_{self.model_arch}")
                self.log('loss_D', loss.item(), logger=True, prog_bar=True, sync_dist=True)
                self.manual_backward(loss)
                if batch_idx % self.gradient_accumulation == 0:
                    opt_d.step()
                    opt_d.zero_grad()
                self.untoggle_optimizer(opt_d)
        elif self.model_arch == "phys_vae_r":
            opt_g = self.optimizers()
            self.toggle_optimizer(opt_g)
            loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, "G_phys_vae_r")
            self.log('loss_G', loss.item(), logger=True, prog_bar=True, sync_dist=True)
            self.manual_backward(loss)
            if batch_idx % self.gradient_accumulation == 0:
                opt_g.step()
                opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
        elif self.model_arch == "phys_factor_vae":
            opt_g = self.optimizers()
            self.toggle_optimizer(opt_g)
            loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, "G_psl_vae")
            self.log('loss_G', loss.item(), logger=True, prog_bar=True, sync_dist=True)
            self.manual_backward(loss)
            if batch_idx % self.gradient_accumulation == 0:
                opt_g.step()
                opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
        elif self.model_arch == "sit":
            opt_diff = self.optimizers()
            self.toggle_optimizer(opt_diff)
            loss = self.calc_loss_batch(Pred_list, Thermal_list, RGB_list, f"Diff_{self.model_arch}")
            if not torch.isfinite(loss).item():
                msg = f"Non-finite loss_Diff at batch_idx={batch_idx}: {loss.detach().cpu().item()}"
                if self.fail_on_nan:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}. Skip it")
                opt_diff.zero_grad(set_to_none=True)
                self.untoggle_optimizer(opt_diff)
                return {}
            self.log('loss_Diff', loss.item(), logger=True, prog_bar=True, sync_dist=True)
            self.manual_backward(loss)
            if batch_idx % self.gradient_accumulation == 0:
                opt_diff.step()
                opt_diff.zero_grad()
            self.update_ema(self.ema, self.model)
            self.untoggle_optimizer(opt_diff)
        else:
            raise NotImplementedError()

        self.log_img_first_iter_train = True

        return {'loss': loss}
    
    def calc_loss_batch(self, Pred_list, Thermal_list, RGB_list, loss_type=None):
        loss = self.loss_function(Pred_list, Thermal_list, RGB_list, loss_type)
        return loss

    def on_train_epoch_start(self):
        self.log_img_first_iter_train = False
        if "klvae" in self.model_arch or "dcae" in self.model_arch or self.model_arch == "phys_vae_r" or self.model_arch == "phys_factor_vae" or ("sit" in self.model_arch and self.calculate_stats):
            self.latent_list = torch.empty(0)
            if "sit" in self.model_arch:
                self.latent_RGB_list = torch.empty(0)
        if self.model_arch == "sit" and hasattr(self, 'latent_cache') and self.current_epoch % self.model_config['cache_rate'] == 0:
            self.latent_cache = []
            if not self.RGB_encoder_training:
                self.RGB_latent_cache = []
            else:
                self.RGB_cache = []
            self.latent_idx = 0
        elif self.model_arch == "sit" and hasattr(self, 'latent_cache'):
            perm = torch.randperm(len(self.latent_cache))
            self.latent_cache =  [self.latent_cache[i] for i in perm]
            if not self.RGB_encoder_training:
                self.RGB_latent_cache = [self.RGB_latent_cache[i] for i in perm]
            else:
                self.RGB_cache = [self.RGB_cache[i] for i in perm]
            self.latent_idx = 0

    def on_train_epoch_end(self):
        self.log_img_first_iter_train = True
        if "klvae" in self.model_arch or "dcae" in self.model_arch or self.model_arch == "phys_vae_r" or self.model_arch == "phys_factor_vae" or ("sit" in self.model_arch and self.calculate_stats):
            self.latent_std = self.latent_list.std().item()
            self.latent_mean = self.latent_list.mean().item()
            print(f"Latent Standard Deviation: {self.latent_std}")
            print(f"Latent Mean: {self.latent_mean}")
            self.log("latent_std", self.latent_std, sync_dist=True)
            self.log("latent_mean", self.latent_mean, sync_dist=True)
            self.log("latent_normalizer", 1 / self.latent_std, sync_dist=True)
            if "sit" in self.model_arch:
                self.latent_RGB_std = self.latent_RGB_list.std(dim=[0, 2, 3]).tolist() # Needed for batch norm init
                self.latent_RGB_mean = self.latent_RGB_list.mean(dim=[0, 2, 3]).tolist() # Needed for batch norm init
                print(f"RGB Latent Standard Deviation: {self.latent_RGB_std}")
                print(f"RGB Latent Mean: {self.latent_RGB_mean}")
                self.log_dict({f"RGB_latent_std_{i}": self.latent_RGB_std[i] for i in range(len(self.latent_RGB_std))})
                self.log_dict({f"RGB_latent_mean_{i}": self.latent_RGB_mean[i] for i in range(len(self.latent_RGB_std))})
        try:
            if self.lr_schedulers() is not None:
                for lr_scheduler in self.lr_schedulers():
                    lr_scheduler.step()
        except:
            self.lr_schedulers().step()
        if hasattr(self, 'latent_cache_init') and self.latent_cache_init == False:
            self.latent_cache_init = True
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None, vis_num=None):
        if vis_num is None:
            vis_num = self.eval_vis_num
        Phys = None
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            RGB, Thermal, dataset_idx, Phys = batch
        else:
            RGB, Thermal, dataset_idx = batch
        if self.model_arch == "klvae" or self.model_arch == "dcae" or self.model_arch == "phys_vae_r" or self.model_arch == "phys_factor_vae":
            output = self(RGB, dataset_idx, Thermal=Thermal, Phys=Phys)
        else:
            output = self(RGB, dataset_idx, Phys=Phys)
        if dataloader_idx is None: # Only one val dataset
            dataloader_idx = 0
        if self.current_dataloader_idx != dataloader_idx:
            self.eval_calculate_metrics()
            self.eval_outputs = []
            self.current_dataloader_idx = dataloader_idx
            self.log_img_first_iter_val = False
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        Thermal = Thermal * 0.5 + 0.5
        Thermal = torch.clamp(Thermal, 0, 1)
        if self.model_arch == "klvae_RGB":
            self.eval_outputs.append((output.detach().cpu(), RGB))
            self.eval_outputs_all.append((output.detach().cpu(), RGB))
        else:
            self.eval_outputs.append((output.detach().cpu(), Thermal))
            self.eval_outputs_all.append((output.detach().cpu(), Thermal))

        save_all_eval = self.save_all_eval_samples
        should_save_local = save_all_eval or (not self.log_img_first_iter_val)
        if should_save_local:
            eval_dataset_name = self.trainer.datamodule.val_datasets[dataloader_idx].dataset_name
            local_vis_num = int(output.shape[0]) if save_all_eval else vis_num
            self.vis_eval_image(
                RGB,
                Thermal,
                output,
                local_vis_num,
                eval_dataset_name,
                'val',
                batch_idx=batch_idx,
                save_all_local=save_all_eval,
                log_to_logger=(not self.log_img_first_iter_val),
            )
            if self.model_arch == "phys_factor_vae":
                with torch.no_grad():
                    thermal_01 = self._thermal_to_zero_one(Thermal)
                    vis_outputs = self.model(
                        thermal_01,
                        sample=False,
                        recompose_mode=self.psl_recompose_mode,
                    )
                self._save_phys_factor_panel_locally(
                    Thermal,
                    vis_outputs,
                    local_vis_num,
                    eval_dataset_name,
                    'val',
                    batch_idx=batch_idx,
                    save_all=save_all_eval,
                )

        self.log_img_first_iter_val = True

        return output.detach().cpu(), Thermal
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.eval_outputs = []
        self.eval_outputs_all = []
        self.results_list = []
        self.current_dataloader_idx = 0
        self.log_img_first_iter_val = False
        self._last_eval_image_dir = None
    
    def on_validation_epoch_end(self):
        dm = self.trainer.datamodule
        self.eval_calculate_metrics() # For last dataset
        for i, eval_dataset in enumerate(dm.val_datasets):
            eval_set_name = eval_dataset.dataset_name
            results_dict = self.results_list[i]
            if results_dict == []:
                continue
            self.log(f'{eval_set_name}_{eval_dataset.split}/PSNR', results_dict['PSNR'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/SSIM', results_dict['SSIM'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/FID', results_dict['FID'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/LPIPS', results_dict['LPIPS'], prog_bar=False, logger=True)
        aggregate_results = self._compute_eval_metrics(self.eval_outputs_all)
        if aggregate_results is not None:
            self.log('val_all/PSNR', aggregate_results['PSNR'], prog_bar=False, logger=True)
            self.log('val_all/SSIM', aggregate_results['SSIM'], prog_bar=False, logger=True)
            self.log('val_all/FID', aggregate_results['FID'], prog_bar=False, logger=True)
            self.log('val_all/LPIPS', aggregate_results['LPIPS'], prog_bar=False, logger=True)
        print('\n\n')
        # reset the outputs list
        self.eval_outputs = []
        self.eval_outputs_all = []
        self.results_list = []
        self.log_img_first_iter_val = False
        self._last_eval_image_dir = None

    def test_step(self, batch, batch_idx, dataloader_idx=None, vis_num=None):
        if vis_num is None:
            vis_num = self.eval_vis_num
        Phys = None
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            RGB, Thermal, dataset_idx, Phys = batch
        else:
            RGB, Thermal, dataset_idx = batch
        if self.model_arch == "klvae" or self.model_arch == "dcae" or self.model_arch == "phys_vae_r" or self.model_arch == "phys_factor_vae":
            output = self(RGB, dataset_idx, Thermal=Thermal, Phys=Phys)
        else:
            output = self(RGB, dataset_idx, Phys=Phys)
        if dataloader_idx is None: # Only one val dataset
            dataloader_idx = 0
        if self.current_dataloader_idx != dataloader_idx:
            self.eval_calculate_metrics()
            self.eval_outputs = []
            self.current_dataloader_idx = dataloader_idx
            self.log_img_first_iter_test = False
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        Thermal = Thermal * 0.5 + 0.5
        Thermal = torch.clamp(Thermal, 0, 1)
        if self.model_arch == "klvae_RGB":
            self.eval_outputs.append((output.detach().cpu(), RGB))
            self.eval_outputs_all.append((output.detach().cpu(), RGB))
        else:
            self.eval_outputs.append((output.detach().cpu(), Thermal))
            self.eval_outputs_all.append((output.detach().cpu(), Thermal))

        save_all_eval = self.save_all_eval_samples
        should_save_local = save_all_eval or (not self.log_img_first_iter_test)
        if should_save_local:
            eval_dataset_name = self.trainer.datamodule.test_datasets[dataloader_idx].dataset_name
            local_vis_num = int(output.shape[0]) if save_all_eval else vis_num
            self.vis_eval_image(
                RGB,
                Thermal,
                output,
                local_vis_num,
                eval_dataset_name,
                'test',
                batch_idx=batch_idx,
                save_all_local=save_all_eval,
                log_to_logger=(not self.log_img_first_iter_test),
            )
            if self.model_arch == "phys_factor_vae":
                with torch.no_grad():
                    thermal_01 = self._thermal_to_zero_one(Thermal)
                    vis_outputs = self.model(
                        thermal_01,
                        sample=False,
                        recompose_mode=self.psl_recompose_mode,
                    )
                self._save_phys_factor_panel_locally(
                    Thermal,
                    vis_outputs,
                    local_vis_num,
                    eval_dataset_name,
                    'test',
                    batch_idx=batch_idx,
                    save_all=save_all_eval,
                )

        self.log_img_first_iter_test = True

        return output.detach().cpu(), Thermal
    
    def prediction_step(self, batch, batch_idx):
        # Only for SiT generating map
        RGB, coordinates, dataset_idx = batch
        output = self(RGB, dataset_idx)
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        return output.detach().cpu(), coordinates
    
    def on_test_epoch_start(self):
        # reset the outputs list
        self.eval_outputs = []
        self.eval_outputs_all = []
        self.results_list = []
        self.current_dataloader_idx = 0
        self.log_img_first_iter_test = False
    
    def on_test_epoch_end(self):
        dm = self.trainer.datamodule
        self.eval_calculate_metrics() # For last dataset
        for i, eval_dataset in enumerate(dm.test_datasets):
            eval_set_name = eval_dataset.dataset_name
            results_dict = self.results_list[i]
            if results_dict == []:
                continue
            self.log(f'{eval_set_name}_{eval_dataset.split}/PSNR', results_dict['PSNR'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/SSIM', results_dict['SSIM'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/FID', results_dict['FID'], prog_bar=False, logger=True)
            self.log(f'{eval_set_name}_{eval_dataset.split}/LPIPS', results_dict['LPIPS'], prog_bar=False, logger=True)
        aggregate_results = self._compute_eval_metrics(self.eval_outputs_all)
        if aggregate_results is not None:
            self.log('test_all/PSNR', aggregate_results['PSNR'], prog_bar=False, logger=True)
            self.log('test_all/SSIM', aggregate_results['SSIM'], prog_bar=False, logger=True)
            self.log('test_all/FID', aggregate_results['FID'], prog_bar=False, logger=True)
            self.log('test_all/LPIPS', aggregate_results['LPIPS'], prog_bar=False, logger=True)
        print('\n\n')
        # reset the outputs list
        self.eval_outputs = []
        self.eval_outputs_all = []
        self.results_list = []
        self.log_img_first_iter_test = False

    def _align_metric_tensors(self, pred, target):
        # Make sure prediction and target have compatible shape/channel for metrics.
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        if pred.shape[1] != target.shape[1]:
            if pred.shape[1] == 1 and target.shape[1] == 3:
                pred = pred.repeat(1, 3, 1, 1)
            elif pred.shape[1] == 3 and target.shape[1] == 1:
                pred = pred.mean(dim=1, keepdim=True)
            else:
                raise RuntimeError(
                    f"Metric channel mismatch: pred={pred.shape[1]}, target={target.shape[1]}"
                )
        return pred, target

    def _compute_eval_metrics(self, eval_outputs):
        if not eval_outputs:
            return None
        psnr = None
        ssim = None
        fid = None
        lpips = None
        for Pred, Thermal in tqdm(eval_outputs, total=len(eval_outputs)):
            Pred = Pred.to("cuda") if torch.cuda.is_available() else Pred
            Thermal = Thermal.to("cuda") if torch.cuda.is_available() else Thermal
            Pred, Thermal = self._align_metric_tensors(Pred, Thermal)
            psnr = calculate_psnr(Pred, Thermal, psnr)
            ssim = calculate_ssim(Pred, Thermal, ssim)
            fid = calculate_fid(Pred, Thermal, fid)
            lpips = calculate_lpips(Pred, Thermal, lpips)
        mean_PSNR = psnr.compute()
        mean_SSIM = ssim.compute()
        mean_FID = fid.compute()
        mean_LPIPS = lpips.compute()
        results_dict = {'PSNR': mean_PSNR, 'SSIM': mean_SSIM, "FID": mean_FID, "LPIPS": mean_LPIPS}
        return results_dict

    def eval_calculate_metrics(self):
        # Calculate evaluation metrics and save to results_list
        results_dict = self._compute_eval_metrics(self.eval_outputs)
        if results_dict is None:
            self.results_list.append([])
            return
        self.results_list.append(results_dict)

    def _get_eval_sample_dir(self, split, eval_dataset_name):
        trainer = getattr(self, "trainer", None)
        if trainer is None or not self.save_eval_images_local:
            return None
        if not getattr(trainer, "is_global_zero", False) or getattr(trainer, "sanity_checking", False):
            return None
        epoch_tag = f"epoch_{int(self.current_epoch) + 1:04d}"
        return os.path.join(trainer.default_root_dir, f"{split}_samples", epoch_tag, eval_dataset_name)

    def _prepare_rgb_for_vis(self, rgb, image_norm="normal"):
        rgb = rgb.detach().float().cpu()
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        if image_norm == "normal":
            image_mean_std = NORMAL_MEAN_STD
        elif image_norm == "imagenet":
            image_mean_std = IMAGENET_MEAN_STD
        elif image_norm in {"none", "already_normalized"}:
            image_mean_std = None
        else:
            raise NotImplementedError()

        if image_mean_std is not None:
            mean = torch.tensor(image_mean_std["mean"], dtype=rgb.dtype).view(1, -1, 1, 1)
            std = torch.tensor(image_mean_std["std"], dtype=rgb.dtype).view(1, -1, 1, 1)
            rgb = rgb * std + mean

        return torch.clamp(rgb, 0, 1)

    def _prepare_output_for_vis(self, tensor):
        tensor = tensor.detach().float().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = torch.clamp(tensor, 0, 1)
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] > 3:
            tensor = tensor[:, :3]
        return tensor

    def _normalize_phys_factor_pair_for_vis(self, teacher_tensor, pred_tensor, clamp01=False):
        teacher_tensor = teacher_tensor.detach().float().cpu()
        pred_tensor = pred_tensor.detach().float().cpu()
        if teacher_tensor.ndim == 3:
            teacher_tensor = teacher_tensor.unsqueeze(0)
        if pred_tensor.ndim == 3:
            pred_tensor = pred_tensor.unsqueeze(0)
        if clamp01:
            return torch.clamp(teacher_tensor, 0, 1), torch.clamp(pred_tensor, 0, 1)
        t_min = teacher_tensor.amin(dim=(-2, -1), keepdim=True)
        p_min = pred_tensor.amin(dim=(-2, -1), keepdim=True)
        lo = torch.minimum(t_min, p_min)
        t_max = teacher_tensor.amax(dim=(-2, -1), keepdim=True)
        p_max = pred_tensor.amax(dim=(-2, -1), keepdim=True)
        hi = torch.maximum(t_max, p_max)
        scale = (hi - lo).clamp_min(1e-6)
        teacher_vis = torch.clamp((teacher_tensor - lo) / scale, 0, 1)
        pred_vis = torch.clamp((pred_tensor - lo) / scale, 0, 1)
        return teacher_vis, pred_vis

    def _save_phys_factor_panel_locally(self, Thermal, outputs, vis_num, eval_dataset_name, split, batch_idx=None, save_all=False):
        root_dir = self._get_eval_sample_dir(split, eval_dataset_name)
        if root_dir is None:
            return
        targets = outputs.get("targets", None) if isinstance(outputs, dict) else None
        if targets is None:
            return

        os.makedirs(root_dir, exist_ok=True)
        if save_all:
            panel_path = os.path.join(root_dir, f"psl_factor_panel_batch_{int(batch_idx):05d}.png")
        else:
            panel_path = os.path.join(root_dir, "psl_factor_panel.png")

        input_01 = torch.clamp(Thermal.detach().float().cpu() * 0.5 + 0.5, 0, 1)
        s_teacher = torch.clamp(targets["S_phys"].detach().float().cpu(), 0, 1)
        s_pred = torch.clamp(outputs["S_phys"].detach().float().cpu(), 0, 1)
        y_hat = torch.clamp(outputs["y_hat"].detach().float().cpu(), 0, 1)
        e_teacher, e_pred = self._normalize_phys_factor_pair_for_vis(targets["e"], outputs["e"], clamp01=True)
        t_teacher, t_pred = self._normalize_phys_factor_pair_for_vis(targets["T_rad"], outputs["T_rad"])
        r_teacher, r_pred = self._normalize_phys_factor_pair_for_vis(targets["R_env"], outputs["R_env"])
        a_teacher, a_pred = self._normalize_phys_factor_pair_for_vis(targets["A"], outputs["A"])
        b_teacher, b_pred = self._normalize_phys_factor_pair_for_vis(targets["B_edge"], outputs["B_edge"], clamp01=True)
        d_teacher, d_pred = self._normalize_phys_factor_pair_for_vis(targets["delta_res"], outputs["delta_res"])

        columns = [
            ("Input", input_01),
            ("S_phys_teacher", s_teacher),
            ("S_phys_pred", s_pred),
            ("y_hat", y_hat),
            ("delta_teacher", d_teacher),
            ("delta_pred", d_pred),
            ("e_teacher", e_teacher),
            ("e_pred", e_pred),
            ("T_teacher", t_teacher),
            ("T_pred", t_pred),
            ("R_teacher", r_teacher),
            ("R_pred", r_pred),
            ("A_teacher", a_teacher),
            ("A_pred", a_pred),
            ("B_teacher", b_teacher),
            ("B_pred", b_pred),
        ]

        num_images = input_01.shape[0] if save_all else min(vis_num, input_01.shape[0])
        if num_images <= 0:
            return
        fig, axes = plt.subplots(num_images, len(columns), figsize=(len(columns) * 2.2, num_images * 2.2))
        if num_images == 1:
            axes = axes[None, :]
        for row in range(num_images):
            for col, (title, tensor) in enumerate(columns):
                ax = axes[row][col]
                ax.imshow(tensor[row, 0].numpy(), cmap='inferno', vmin=0.0, vmax=1.0)
                ax.axis('off')
                if row == 0:
                    ax.set_title(title, fontsize=8)
        mode_tag = str(outputs.get("recompose_mode", self.psl_recompose_mode)) if isinstance(outputs, dict) else self.psl_recompose_mode
        fig.suptitle(f"PSL-VAE | mode={mode_tag} | split={split} | dataset={eval_dataset_name} | epoch={int(self.current_epoch) + 1:04d}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(panel_path, dpi=200)
        plt.close(fig)

    def _save_eval_images_locally(self, RGB, Thermal, output, vis_num, eval_dataset_name, split, image_norm="normal", batch_idx=None, save_all=False):
        root_dir = self._get_eval_sample_dir(split, eval_dataset_name)
        if root_dir is None:
            return

        os.makedirs(root_dir, exist_ok=True)
        if self._last_eval_image_dir != root_dir:
            print(f"[INFO] Saving {split} samples to: {root_dir}")
            self._last_eval_image_dir = root_dir
        rgb_vis = self._prepare_rgb_for_vis(RGB, image_norm=image_norm)
        thermal_vis = self._prepare_output_for_vis(Thermal)
        output_vis = self._prepare_output_for_vis(output)
        if save_all:
            num_images = min(rgb_vis.shape[0], thermal_vis.shape[0], output_vis.shape[0])
        else:
            num_images = min(vis_num, rgb_vis.shape[0], thermal_vis.shape[0], output_vis.shape[0])

        for idx in range(num_images):
            if save_all:
                prefix = f"b{int(batch_idx):05d}_s{idx:03d}"
            else:
                prefix = f"{idx:02d}"
            torchvision.utils.save_image(rgb_vis[idx], os.path.join(root_dir, f"input_{prefix}.png"))
            torchvision.utils.save_image(thermal_vis[idx], os.path.join(root_dir, f"gt_{prefix}.png"))
            torchvision.utils.save_image(output_vis[idx], os.path.join(root_dir, f"pred_{prefix}.png"))
            torchvision.utils.save_image(
                torch.cat([rgb_vis[idx], thermal_vis[idx], output_vis[idx]], dim=-1),
                os.path.join(root_dir, f"compare_{prefix}.png"),
            )

    def vis_eval_image(self, RGB, Thermal, output, vis_num, eval_dataset_name, split, image_norm="normal", batch_idx=None, save_all_local=False, log_to_logger=True):
        self._save_eval_images_locally(
            RGB,
            Thermal,
            output,
            vis_num,
            eval_dataset_name,
            split,
            image_norm=image_norm,
            batch_idx=batch_idx,
            save_all=save_all_local,
        )
        if not log_to_logger:
            return
        logger_obj = getattr(self, "logger", None)
        if logger_obj is None:
            return
        if hasattr(logger_obj, "log_image"):
            image_loggers = [logger_obj]
        elif hasattr(logger_obj, "loggers"):
            image_loggers = [item for item in logger_obj.loggers if hasattr(item, "log_image")]
        else:
            image_loggers = []
        if len(image_loggers) == 0:
            return

        denormalized_image = self._prepare_rgb_for_vis(RGB, image_norm=image_norm)
        list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
        list_images = list_images[:vis_num]
        for logger_item in image_loggers:
            logger_item.log_image(f'input_{split}_images_{eval_dataset_name}', list_images)
        denormalized_image = self._prepare_output_for_vis(Thermal)
        list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
        list_images = list_images[:vis_num]
        for logger_item in image_loggers:
            logger_item.log_image(f'gt_{split}_thermal_{eval_dataset_name}', list_images)
        if not (self.model_arch == "sit" and split == "train"):
            denormalized_image = self._prepare_output_for_vis(output)
            list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
            list_images = list_images[:vis_num]
            for logger_item in image_loggers:
                logger_item.log_image(f'pred_{split}_thermal_{eval_dataset_name}', list_images)

    @torch.no_grad()
    def update_ema(self, ema_model, model, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

        ema_buffers = OrderedDict(ema_model.named_buffers())
        model_buffers = OrderedDict(model.named_buffers())

        for name, buffer in model_buffers.items():
            name = name.replace("module.", "")
            if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
                # Apply EMA only to float buffers
                ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
            else:
                # Direct copy for non-float buffers
                ema_buffers[name].copy_(buffer)

    def _resolve_phys_vae_r_stage(self, stage: str) -> str:
        stage = str(stage).lower()
        if stage in {"full", "joint", "all"}:
            return "joint"
        if stage in {"phys", "physics"}:
            return "phys"
        if stage in {"res", "residual"}:
            return "res"
        raise ValueError(f"Unsupported PHYS-VAE-R training_stage={stage}. Expected phys, res, or joint.")

    def _thermal_to_zero_one(self, thermal: torch.Tensor) -> torch.Tensor:
        return torch.clamp(thermal * 0.5 + 0.5, 0.0, 1.0)

    def _thermal_from_zero_one(self, thermal_01: torch.Tensor) -> torch.Tensor:
        return torch.clamp(thermal_01, 0.0, 1.0) * 2.0 - 1.0

    def _build_phys_teacher_q(self, teacher_cfg: dict):
        teacher_cfg = dict(teacher_cfg)
        a_low_range = tuple(teacher_cfg.get('a_low_range', [0.8, 1.2]))
        teacher = TeR_B(
            smp_model=str(teacher_cfg.get('smp_model', 'Unet')),
            smp_encoder=str(teacher_cfg.get('smp_encoder', 'resnet18')),
            smp_encoder_weights=teacher_cfg.get('smp_encoder_weights', None),
            vnums=int(teacher_cfg.get('vnums', 4)),
            erme_kernel=int(teacher_cfg.get('erme_kernel', 5)),
            lambda_env_init=float(teacher_cfg.get('lambda_env_init', 0.1)),
            a_low_range=(float(a_low_range[0]), float(a_low_range[1])),
        )
        ckpt = str(teacher_cfg.get('ckpt', ''))
        if not ckpt or (not os.path.isfile(ckpt)):
            raise FileNotFoundError(f"TeR-B Net ckpt not found: {ckpt}")
        load_info = load_module_checkpoint(
            teacher,
            ckpt,
            strict=bool(teacher_cfg.get('strict_load', False)),
            strip_prefixes=('model.', 'module.'),
        )
        print(
            f"[INFO] Loaded PHYS-VAE-R teacher Q from {ckpt} "
            f"(source={load_info.get('state_source', 'unknown')}, state_tensors={load_info['num_state_tensors']}, "
            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})"
        )
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher

    def load_pretrained_phys_vae_r(self, model, allow_missing: bool = False):
        vae_path = self.model_config.get('vae_path', None)
        if not vae_path or (not os.path.isfile(vae_path)):
            msg = f"PHYS-VAE-R checkpoint not found: {vae_path}"
            if allow_missing:
                print(f"[WARN] {msg}. Continue with randomly initialized PHYS-VAE-R.")
                return model
            raise FileNotFoundError(msg)
        load_info = load_module_checkpoint(
            model,
            vae_path,
            strict=False,
            strip_prefixes=('model.', 'module.'),
        )
        print(
            f"[INFO] Loaded PHYS-VAE-R from {vae_path} "
            f"(source={load_info.get('state_source', 'unknown')}, state_tensors={load_info['num_state_tensors']}, "
            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})"
        )
        return model

    def load_pretrained_phys_factor_vae(self, model, allow_missing: bool = False):
        vae_path = self.model_config.get('vae_path', None)
        if not vae_path or (not os.path.isfile(vae_path)):
            msg = f"PSL-VAE checkpoint not found: {vae_path}"
            if allow_missing:
                print(f"[WARN] {msg}. Continue with randomly initialized PSL-VAE.")
                return model
            raise FileNotFoundError(msg)
        load_info = load_module_checkpoint(
            model,
            vae_path,
            strict=False,
            strip_prefixes=('model.', 'module.'),
        )
        print(
            f"[INFO] Loaded PSL-VAE from {vae_path} "
            f"(source={load_info.get('state_source', 'unknown')}, state_tensors={load_info['num_state_tensors']}, "
            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})"
        )
        return model

    def load_pretrained(self, model, allow_missing: bool = False):
        vae_path = self.model_config.get('vae_path', None)
        if not vae_path or (not os.path.isfile(vae_path)):
            msg = f"VAE checkpoint not found: {vae_path}"
            if allow_missing:
                print(f"[WARN] {msg}. Continue with randomly initialized thermal VAE.")
                return model
            raise FileNotFoundError(msg)

        state_obj = torch.load(vae_path, map_location='cpu')
        if isinstance(state_obj, dict) and 'state_dict' in state_obj:
            state_dict = state_obj['state_dict']
        elif isinstance(state_obj, dict):
            state_dict = state_obj
        else:
            raise ValueError(f"Unsupported checkpoint format in {vae_path}")
        new_state_dict = {}
        for old_key, value in state_dict.items():
            # 1) Skip any keys you definitely don’t need:
            if old_key.startswith("loss_fn."):
                continue
            
            # 2) Strip off "model." if the model was saved that way:
            if old_key.startswith("model."):
                new_key = old_key.replace("model.", "")  # remove the "model." prefix
            else:
                new_key = old_key
            
            # Now add to the new dict
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        return model

    def load_rgb_vae_kl(self):
        rgb_local_path = self.model_config.get("rgb_vae_path", None)
        default_rgb_local_path = os.environ.get("PSL_FLOW_RGB_VAE_PATH", "checkpoints/sd-vae-ft-ema")
        rgb_repo = self.model_config.get("rgb_vae_repo", f"stabilityai/sd-vae-ft-{self.model_config['vae']}")
        local_only = bool(self.model_config.get("rgb_vae_local_files_only", False))
        allow_missing = bool(self.model_config.get("allow_missing_rgb_vae_path", False))

        try:
            local_candidates = []
            if rgb_local_path:
                local_candidates.append(str(rgb_local_path))
            local_candidates.append(default_rgb_local_path)

            checked = set()
            for local_path in local_candidates:
                if local_path in checked:
                    continue
                checked.add(local_path)
                if os.path.exists(local_path):
                    print(f"[INFO] Load RGB VAE from local path: {local_path}")
                    return _load_autoencoder_kl_pretrained(local_path, local_files_only=True)

            return _load_autoencoder_kl_pretrained(str(rgb_repo), local_files_only=local_only)
        except Exception as e:
            if allow_missing:
                print(f"[WARN] RGB VAE load failed ({type(e).__name__}: {e}). Fallback to thermal VAE weights.")
                return copy.deepcopy(self.thermal_vae)
            raise

    def _get_vae_input_channels(self, vae):
        if hasattr(vae, "config") and getattr(vae.config, "in_channels", None) is not None:
            return int(vae.config.in_channels)
        encoder = getattr(vae, "encoder", None)
        conv_in = getattr(encoder, "conv_in", None)
        if conv_in is not None and hasattr(conv_in, "in_channels"):
            return int(conv_in.in_channels)
        return None

    def _adapt_channels_for_vae(self, x, vae, vae_name="VAE"):
        target_channels = self._get_vae_input_channels(vae)
        if target_channels is None or x.shape[1] == target_channels:
            return x
        if target_channels == 1 and x.shape[1] == 3:
            return x.mean(dim=1, keepdim=True)
        if target_channels == 3 and x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        raise RuntimeError(
            f"{vae_name} channel mismatch: expected {target_channels}, got {x.shape[1]}."
        )

    def _get_rgb_vae_model_type(self):
        if self.model_arch != "sit":
            return None
        if 'rgb_vae_model' in self.model_config:
            return str(self.model_config['rgb_vae_model']).lower()
        if self.model_config['vae_model'] == "dcae":
            return "dcae"
        return "klvae"

    def encode_rgb_posterior(self, RGB_padded):
        rgb_model_type = self._get_rgb_vae_model_type()
        if rgb_model_type != "klvae":
            raise RuntimeError("encode_rgb_posterior is only valid when vae_model=klvae.")
        rgb_in = self._adapt_channels_for_vae(RGB_padded, self.RGB_vae, vae_name="RGB VAE")
        return self.RGB_vae.encode(rgb_in).latent_dist

    def encode_rgb_latent(self, RGB_padded):
        rgb_in = self._adapt_channels_for_vae(RGB_padded, self.RGB_vae, vae_name="RGB VAE")
        rgb_model_type = self._get_rgb_vae_model_type()
        if rgb_model_type == "klvae":
            return self.RGB_vae.encode(rgb_in).latent_dist.sample()
        elif rgb_model_type == "dcae":
            return self.RGB_vae.encode(rgb_in).latent
        raise NotImplementedError()
    
    def pad_to_divisble(self, input, multiple=8):
        batch, channels, height, width = input.shape
        padded_height = ((height + multiple - 1) // multiple) * multiple
        padded_width = ((width + multiple - 1) // multiple) * multiple
        
        pad_top = 0
        pad_bottom = padded_height - height
        pad_left = 0
        pad_right = padded_width - width
        
        padded_image = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_image
    
    def preprocess_raw_image(self, x, enc_type):
        resolution = x.shape[-1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # For thermal
        if 'dinov2' in enc_type:
            x = x * 0.5 + 0.5
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        return x
    
    def load_encoder(self, enc_type):
        resolution = 256
        if 'dinov2' in enc_type:
            import timm
            if 'reg' in enc_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder.eval()
        return encoder


# Backward-compatible alias for legacy checkpoints and scripts.








