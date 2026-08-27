# PSL-Flow

Implementation of **PSL-Flow: Physics-Structured Latent Flow Matching for Aerial Visible-to-Infrared Image Translation**.

![Overall architecture of PSL-Flow](Overall%20architecture%20of%20PSL-Flow.svg)

PSL-Flow translates aerial visible images into infrared images by learning conditional flow matching in a physics-structured latent space. The released training pipeline follows three stages:

1. **TeR-B** estimates thermal physical factors from infrared images.
2. **PSL-VAE** encodes the five-channel physical representation `[T, e, R_env, B, Delta]` into a latent space.
3. **PSL-Flow / SiT** learns visible-conditioned flow matching in that latent space and decodes the generated latent state back to infrared images.

## Install

The code was tested with Python 3.10, PyTorch 2.0.1, TorchVision 0.15.2, and CUDA 11.7 on an NVIDIA A100 GPU.

```bash
conda env create -f environment.yml
conda activate psl-flow
```

If your server already has a CUDA-ready PyTorch environment, install only the project-level dependencies:

```bash
pip install -r requirements.txt
```

For FID evaluation, keep `torchmetrics` and `torch-fidelity` installed. If they are unavailable, training still runs, but FID reporting is skipped with a warning.

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

`scripts/run_pipeline.sh` checks the generated shards before training starts. If you use another output root, set `DATASETS_PREPROCESS_ROOT=/path/to/datasets_preprocess`.

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

`metadata.json` must contain the actual number of paired samples in that split, for example:

```json
{"num_samples": 10000}
```

Supported dataset keys are `AVIID`, `CART`, `DroneVehicle_day`, and `DroneVehicle_night`.

## One-Command Training

Run the full TeR-B -> PSL-VAE -> PSL-Flow route:

```bash
bash scripts/run_pipeline.sh --dataset AVIID --route psl_flow
```

The command trains the three stages in order, estimates the PSL-VAE latent normalizer for PSL-Flow, saves fixed-step SiT checkpoints, records elapsed time and peak GPU memory, and runs one final validation after training.

Run from the repository root unless you set absolute paths for the data and checkpoint directories.

## Configs

Experiment configs live in `psl_flow/configs/experiments/`:

- `terb/<dataset>.yml`
- `psl_vae/<dataset>.yml`
- `psl_flow/<dataset>.yml`

Dataset split definitions live in `psl_flow/configs/experiments/datasets/`.

## Checkpoints

No checkpoint is bundled with this repository. The pipeline writes new checkpoints under:

```text
logs/experiments/<dataset>/psl_flow/<run_id>/artifacts/
```

Each stage automatically consumes the selected checkpoint from the previous stage. Existing runs can be resumed from the same run directory; completed checkpoints are reused and missing validation or downstream stages continue.

## Validation

Final metrics are written as JSON under:

```text
logs/experiments/<dataset>/psl_flow/<run_id>/artifacts/metrics/
```

## Citation

```bibtex
@article{lin_pslflow,
  title   = {PSL-Flow: Physics-Structured Latent Flow Matching for Aerial Visible-to-Infrared Image Translation},
  author  = {Lin, Leping and Zheng, Zibin and Ouyang, Ning},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  doi     = {10.1109/LGRS.2026.3726472},
}
```

## Acknowledgements

This project builds on and is inspired by the following open-source projects:

- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [willisma/SiT](https://github.com/willisma/SiT)
- [arplaboratory/ThermalGen](https://github.com/arplaboratory/ThermalGen)
- [fangyuanmao/PID](https://github.com/fangyuanmao/PID)

## License

This repository is released under the MIT License. The bundled SiT transport/network files retain their upstream license in `psl_flow/models/sit/LICENSE`.
