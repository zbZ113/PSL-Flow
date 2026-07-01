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

The pipeline defaults to GPU 1, batch size 16, dataset-size epochs for TeR-B/PSL-VAE, training-time validation, fixed SiT step checkpoints, and one final validation at the end. It records elapsed time and peak GPU memory in `summary.csv`.

Default training management:

- TeR-B: 200 epochs, validation every 20 epochs, `best.ckpt` by `val/loss_total`, plus `last.ckpt`.
- PSL-VAE: 300 epochs, validation every 50 epochs, `best.ckpt` by `val/LPIPS`, plus `last.ckpt`.
- PSL-Flow / SiT: validation every 5K steps, step checkpoints from the experiment table, plus `last.ckpt` and `best.ckpt` when validation is enabled.
- Qualitative samples are written under `val_samples/`, `test_samples/`, and `best_samples/latest`.
- Local metric traces are written to `local_logs/metrics.csv` and `local_logs/history.jsonl`.

Default PSL-Flow step checkpoints:

| Dataset | PSL-Flow steps | KLVAE-SiT step |
| --- | --- | --- |
| AVIID | 45K, 75K | 45K |
| CART | 35K, 65K | 35K |
| DroneVehicle_day | 70K, 100K | 70K |
| DroneVehicle_night | 90K, 120K | 90K |

Useful environment overrides:

```bash
CUDA_VISIBLE_DEVICES_REQUESTED=1
NVIDIA_SMI_GPU_ID=1
RGB_VAE_PATH=/root/autodl-fs/sd-vae-ft-ema
THERMAL_KLVAE_CKPT=/path/to/thermal_klvae.ckpt
THERMAL_KLVAE_NORMALIZER=1.0
TRAIN_WITH_VALIDATION=1
TERB_NUM_SAMPLES_PER_EPOCH=auto
PSLVAE_NUM_SAMPLES_PER_EPOCH=auto
FLOW_NUM_SAMPLES_PER_EPOCH=auto
PSLVAE_SELECT=best_lpips      # best_lpips, best_psnr, best_ssim, best_fid, epoch, last
PSLVAE_EPOCH=280             # used only when PSLVAE_SELECT=epoch
FLOW_VAL_EVERY_STEPS=5000
FLOW_SELECT=step2            # step1, step2, best, best_lpips, best_psnr, best_ssim, last
EVAL_FID=0                   # optional; requires torchmetrics FID support when enabled
SIT_STEP_1=45000
SIT_STEP_2=75000
```

Set `TRAIN_WITH_VALIDATION=0` only for timing-only benchmark runs. In that mode, the pipeline keeps `last.ckpt` and final validation, but training-time best checkpoint selection is unavailable.

For the PSL-Flow route, `thermal_normalizer` is not fixed in the base config. The pipeline selects the PSL-VAE checkpoint according to `PSLVAE_SELECT`, estimates latent statistics from that selected model, and writes `psl_vae_ckpt`, `thermal_normalizer`, latent mean/std, and the stats JSON path into the patched SiT config before training starts. The normalizer cache is checkpoint-aware: changing the selected PSL-VAE checkpoint, TeR-B checkpoint, latent sampling mode, seed, or requested sample count forces re-estimation.

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
