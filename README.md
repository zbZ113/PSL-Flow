# PSL-Flow

Clean implementation of Physics-Structured Latent Flow Matching for visible-to-infrared image translation.

The public code contains four routes only:

- TeR-B teacher
- PSL-VAE
- PSL-Flow / SiT
- KLVAE-SiT ablation

## Install

```bash
conda env create -f environment.yml
conda activate psl-flow
```

or install into an existing CUDA-ready PyTorch environment:

```bash
pip install -r requirements.txt
```

## Data

Training reads WebDataset shards from `datasets_preprocess/`. If you downloaded the original image folders, put them under `datasets_raw/` first and run preprocessing before training.

Expected raw layouts:

```text
datasets_raw/
  AVIID/
    train/vis/...
    train/ir/...
    test/vis/...
    test/ir/...
  CART/
    train/RGB/...
    train/Thermal/...
    val/RGB/...          # optional; test is reused when val is absent
    val/Thermal/...
    test/RGB/...
    test/Thermal/...
  DroneVehicle/
    DayDrone/trainA/...
    DayDrone/trainB/...
    DayDrone/testA/...
    DayDrone/testB/...
    NightDrone/trainA/...
    NightDrone/trainB/...
    NightDrone/testA/...
    NightDrone/testB/...
```

Preprocess one dataset:

```bash
bash scripts/preprocess_dataset.sh --dataset AVIID --raw-root datasets_raw --output-root datasets_preprocess --overwrite
```

Preprocess every supported dataset:

```bash
bash scripts/preprocess_dataset.sh --dataset all --raw-root datasets_raw --output-root datasets_preprocess --overwrite
```

`scripts/run_pipeline.sh` checks these shards before training starts. If you use another output root, set `DATASETS_PREPROCESS_ROOT=/path/to/datasets_preprocess`.

The generated structure is:

```text
datasets_preprocess/
  AVIID/
    train/
      dataset-000000.tar
      metadata.json
    test/
      dataset-000000.tar
      metadata.json
```

Each tar sample must contain:

```text
color.png
thermal.png
```

`metadata.json` must contain:

```json
{"num_samples": 2412}
```

Supported dataset keys are `AVIID`, `CART`, `DroneVehicle_day`, and `DroneVehicle_night`.

## One-Command Workflows

Run the full TeR-B -> PSL-VAE -> PSL-Flow route:

```bash
bash scripts/run_pipeline.sh --dataset AVIID --route psl_flow
```

Run the KLVAE-SiT ablation route:

```bash
bash scripts/run_pipeline.sh --dataset AVIID --route klvae_sit
```

The pipeline defaults to GPU 1, batch size 16, no validation during training, checkpoints at 45K and 75K SiT steps, and one final validation at the end. It records elapsed time and peak GPU memory in `summary.csv`.

Useful environment overrides:

```bash
CUDA_VISIBLE_DEVICES_REQUESTED=1
NVIDIA_SMI_GPU_ID=1
RGB_VAE_PATH=/root/autodl-fs/sd-vae-ft-ema
THERMAL_KLVAE_CKPT=/path/to/thermal_klvae.ckpt
THERMAL_KLVAE_NORMALIZER=1.0
```

For the PSL-Flow route, `thermal_normalizer` is not fixed in the base config. The pipeline estimates it from the trained PSL-VAE latent statistics and writes a patched run config before SiT training starts.

## Configs

Experiment configs live in `psl_flow/configs/experiments/`:

- `terb/<dataset>.yml`
- `psl_vae/<dataset>.yml`
- `psl_flow/<dataset>.yml`
- `klvae_sit/<dataset>.yml`

Dataset split definitions live in `psl_flow/configs/experiments/datasets/`.

## Checkpoints

No checkpoint is bundled with this repository. The pipeline writes new checkpoints under `logs/experiments/<dataset>/<route>/<run_id>/artifacts/`.

For `psl_flow`, the next stage automatically consumes the final checkpoint from the previous stage. For `klvae_sit`, provide a trained thermal KL-VAE checkpoint through `THERMAL_KLVAE_CKPT`.

## Validation

Final metrics are written as JSON under:

```text
logs/experiments/<dataset>/<route>/<run_id>/artifacts/metrics/
```

The same run directory can be resumed. If complete checkpoints already exist, the pipeline reuses them and continues with the missing validation or next stage.

## Citation

```bibtex
@article{pslflow2026,
  title = {Physics-Structured Latent Flow Matching for Aerial Visible-to-Infrared Image Translation},
  author = {Anonymous},
  journal = {Manuscript under review},
  year = {2026}
}
```

## License

This repository is released under the MIT License. The bundled SiT transport/network files retain their upstream license in `psl_flow/models/sit/LICENSE`.
