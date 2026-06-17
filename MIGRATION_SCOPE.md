# PSL-Flow Migration Scope

This workspace was extracted for the paper route **PSL-Flow: Physics-Structured Latent Flow Matching for Aerial Visible-to-Infrared Image Translation**.

## Kept Routes

1. `TeR-B Net -> PSL-VAE thermal tokenizer -> conditional SiT latent generation`
2. `KLVAE -> conditional SiT latent generation` as the tokenizer ablation

## Kept Runtime Code

- `psl_flow.py`, `psl_flow_core.py`, `main.py`, `test.py`
- `models/generative_models/sit_networks`
- `models/psl_vae`
- `models/physics/terb_net.py` and shared physics utilities
- `dataloaders` and `utils`
- lightweight legacy model folders still imported by the retained runtime

## Kept Configs And Shell Entrypoints

- `configs/train/sit_cond/psl_flow_l2_concat.yml`
- `configs/test/sit_cond/psl_flow_l2_concat.yml`
- `configs/train/sit_cond/generic_sit_l2_concat.yml`
- `configs/test/sit_cond/generic_sit_l2_concat.yml`
- `configs/train/ldm/psl_vae_all_256_1st.yml`
- `configs/test/ldm/psl_vae_all_256_1st.yml`
- `configs/train/ldm/klvae_all_256_1st.yml`
- `configs/train/ldm/klvae_all_256_3rd.yml`
- `shell/Physics/*terb_net*.sh`
- `shell/VAE/*psl_vae*.sh`
- `shell/SiT/*psl_flow*.sh`
- `shell/SiT/*generic_sit*.sh`
- selected benchmark/evaluation scripts under `scripts/`

## Not Migrated

- dataset images and preprocessed samples
- checkpoint weights
- generated output images
- unrelated training routes such as GAN, VQGAN, PHYS-VAE-R, PGQA, and physproxy

## Expected External Paths

- datasets under `datasets_preprocess/`
- TeR-B Net checkpoints under `checkpoints/physics/<DATASET>/terb_net/`
- PSL-VAE checkpoints under `checkpoints/psl_vae/<DATASET>/`
- KLVAE checkpoints under `checkpoints/klvae/...`
- PSL-Flow checkpoints under `logs/psl_flow/<DATASET>/checkpoints/`

Legacy filenames and aliases are kept only for checkpoint/config compatibility.
