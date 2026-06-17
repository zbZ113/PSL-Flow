# PSL-Flow: Physics-Structured Latent Flow Matching for Aerial Visible-to-Infrared Image Translation

This repository contains the implementation used for **PSL-Flow**, an aerial visible-to-infrared (V2IR) image translation framework that learns conditional flow matching in a physics-structured latent space.

The code keeps only the routes needed for the paper:

- **PSL-Flow main route:** `TeR-B Net -> PSL-VAE -> conditional SiT`
- **Tokenizer ablation:** `KLVAE -> conditional SiT`

Other historical generation routes are not part of this migrated workspace.

## Method

PSL-Flow reformulates aerial V2IR generation from direct infrared image regression into thermal state modeling. It has three stages:

1. **TeR-B Net** estimates physical factors from real infrared images: temperature proxy `T`, emissivity `e`, environmental radiance `R_env`, attenuation/response `A`, and boundary response `B`.
2. **PSL-VAE** encodes the TeR-B factor stack and local residual compensation into a physics-structured latent state.
3. **Conditional SiT** learns a visible-conditioned continuous transport from Gaussian noise to the PSL-VAE thermal latent state, then reconstructs infrared images through the PSL-VAE decoder.

During inference, TeR-B Net is not on the default generation path. The visible image is encoded as condition, SiT samples the target physical latent state, and PSL-VAE decodes the thermal image.

## Setup

```bash
conda env create -f env.yml
conda activate PSL_Flow
pip install -r requirements-physics.txt
```

Expected external assets are not stored in this repo:

- datasets under `datasets_preprocess/`
- TeR-B checkpoints under `checkpoints/physics/<DATASET>/terb_net/`
- PSL-VAE checkpoints under `checkpoints/psl_vae/<DATASET>/`
- KLVAE checkpoints under `checkpoints/klvae/`
- PSL-Flow SiT checkpoints under `logs/psl_flow/<DATASET>/checkpoints/`

## Train TeR-B Net

```bash
DATASET_NAME=AVIID bash shell/Physics/train_terb_net_dataset.sh
DATASET_NAME=CART bash shell/Physics/train_terb_net_dataset.sh
DATASET_NAME=DroneVehicle_day bash shell/Physics/train_terb_net_dataset.sh
DATASET_NAME=DroneVehicle_night bash shell/Physics/train_terb_net_dataset.sh
```

The dataset script writes logs to `logs/physics/<DATASET>/terb_net` and checkpoints to `checkpoints/physics/<DATASET>/terb_net`.

Dataset shortcut scripts such as `shell/Physics/aviid_terb_net.sh` and `shell/Physics/cart_terb_net.sh` are also provided.

## Train PSL-VAE

```bash
DATASET_NAME=AVIID \
TERB_NET_CKPT=checkpoints/physics/AVIID/terb_net/teacher_best.pth \
bash shell/VAE/train_psl_vae_dataset.sh
```

The base config is `configs/train/ldm/psl_vae_all_256_1st.yml`. The wrapper writes the tokenizer checkpoint to `checkpoints/psl_vae/<DATASET>/checkpoints/last.ckpt`.

Dataset shortcut scripts such as `shell/VAE/aviid_psl_vae.sh` and `shell/VAE/cart_psl_vae.sh` are also provided.

## Train PSL-Flow

```bash
SIT_DATASET=AVIID \
TERB_NET_CKPT=checkpoints/physics/AVIID/terb_net/teacher_best.pth \
SIT_VAE_CKPT=auto \
bash shell/SiT/train_psl_flow_dataset.sh
```

The base config is `configs/train/sit_cond/psl_flow_l2_concat.yml`. `SIT_VAE_CKPT=auto` resolves the PSL-VAE checkpoint from `logs/psl_vae` and `checkpoints/psl_vae`.

Dataset shortcut scripts such as `shell/SiT/aviid_sit_train_psl_flow.sh` and `shell/SiT/cart_sit_train_psl_flow.sh` are also provided.

## KLVAE->SiT Ablation

```bash
SIT_DATASET=AVIID \
SIT_VAE_CKPT=checkpoints/klvae/checkpoints/last.ckpt \
bash shell/SiT/train_generic_sit_dataset.sh
```

The base config is `configs/train/sit_cond/generic_sit_l2_concat.yml`. This route keeps the same conditional SiT backbone but uses a generic KLVAE tokenizer instead of PSL-VAE.

## Evaluation

Evaluate PSL-Flow:

```bash
SIT_DATASET=AVIID \
SIT_CKPT=auto \
TERB_NET_CKPT=checkpoints/physics/AVIID/terb_net/teacher_best.pth \
bash shell/SiT/test_psl_flow_dataset.sh
```

Evaluate the KLVAE->SiT ablation:

```bash
SIT_DATASET=AVIID \
SIT_CKPT=auto \
SIT_VAE_CKPT=checkpoints/klvae/checkpoints/last.ckpt \
bash shell/SiT/test_generic_sit_dataset.sh
```

Run PSL-Flow recomposition ablations:

```bash
DATASETS="AVIID CART DroneVehicle_day DroneVehicle_night" \
PSL_RECOMPOSE_MODES="full delta_only phys_only" \
bash shell/SiT/eval_psl_flow_ablation_all.sh
```

Benchmark SiT routes:

```bash
python scripts/benchmark_sit_routes.py \
  --ckpt logs/psl_flow/AVIID/checkpoints/last.ckpt \
  --image path/to/visible.png \
  --route psl_flow
```

## Important Files

- `psl_flow.py`: paper-facing Lightning entrypoint exporting `PSLFlow`
- `models/physics/terb_net.py`: paper-facing TeR-B Net import path
- `models/psl_vae/`: paper-facing PSL-VAE import path
- `configs/train/ldm/psl_vae_all_256_1st.yml`: PSL-VAE tokenizer training config
- `configs/train/sit_cond/psl_flow_l2_concat.yml`: PSL-Flow training config
- `configs/train/sit_cond/generic_sit_l2_concat.yml`: KLVAE->SiT ablation config
- `scripts/write_grsl_docx.py`: manuscript-generation script for the PSL-Flow paper draft

## Compatibility Notes

Some internal filenames and aliases from the source workspace are intentionally retained so old checkpoints and config values can still load. New experiments should use `PSLFlow`, `PSL_VAE`, `TeR_B`, `psl_vae`, and `psl_flow` names.
