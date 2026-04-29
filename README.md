# CNN vs ViT Robustness Benchmark

This repository runs a modular robustness benchmark comparing a plain CNN, a Vision Transformer, and a modern CNN on CIFAR-100 under controlled input corruptions.

The project is currently **inference-only** (no training loop), with dense corruption sweeps and configurable metrics for clearer degradation trends. **Three models of comparable capacity** are benchmarked to isolate architectural differences from capacity confounds.

---

## Current project scope

- Dataset: CIFAR-100 only
- **Three models of comparable capacity**:
  - **ResNet-101** (44.5M params) — plain CNN baseline
  - **ViT-B/16** (86.6M params) — Vision Transformer
  - **ConvNeXt-Small** (50.2M params) — modern CNN variant
  - Capacity spread: 1.9× (ResNet-101 vs ViT-B/16), avoiding architectural confounds from capacity differences
- Corruptions: rotation, translation (x/y), Gaussian noise
- Metrics: top-k, NLL, ECE, robustness ratio, trend summaries, prediction stability

---

## Repository structure

Core source files currently live directly under [src](src):

- [src/main.py](src/main.py) — experiment orchestration
- [src/config_schema.py](src/config_schema.py) — YAML config dataclasses + loader
- [src/datasets.py](src/datasets.py) — CIFAR-100 loading, class-balanced subset, dataloader
- [src/corruptions.py](src/corruptions.py) — corruption specification and transform builders
- [src/models.py](src/models.py) — pretrained model loading and preprocessing
- [src/runner.py](src/runner.py) — clean/corrupted evaluation loops and per-sample logging
- [src/metrics.py](src/metrics.py) — metric computation and trend utilities
- [src/plots.py](src/plots.py) — corruption-family degradation plots
- [src/io_utils.py](src/io_utils.py) — CSV/JSON persistence
- [src/utils.py](src/utils.py) — seed and device utilities
- [run_experiment.py](run_experiment.py) — thin entry script

---

## Configuration (single source of truth)

The current experiment configurations are [configs/testing_experiment.yaml](configs/testing_experiment.yaml) and [configs/production_experiment.yaml](configs/production_experiment.yaml).

### Dataset block

- `name`: currently intended as `cifar100`
- `root`: data directory
- `subset_per_class`: balanced evaluation subset size
- `batch_size`, `num_workers`

### Models block

- `names`: list of model IDs to run
- Available models:
  - `resnet101` (44.5M params)
  - `vit_b_16` (86.6M params)
  - `convnext_small` (50.2M params)
- Both config files (`testing_experiment.yaml` and `production_experiment.yaml`) use all three models

### Corruptions block

Dense sweeps are enabled by default for better trend visibility:

- `rotation_degrees`
- `translation_pixels`
- `gaussian_sigmas`

### Evaluation block

- `device`: `auto | cpu | cuda | mps`
- `seed`
- `num_repeats`: number of independent seeded repeats to average
- `topk`
- `bootstrap_iters`
- `save_per_sample`

### Metrics block

- `enable_topk`
- `enable_nll`
- `enable_ece`
- `ece_bins`
- `enable_stability`

### Output block

- `output_dir`
- `run_name`

---

## Methodology implemented in code

For each model:

1. Evaluate clean subset performance.
2. Evaluate every corruption/severity condition.
3. Save both aggregate rows and (optionally) per-sample rows.
4. Compute per-family summary trends:
   - `audc_*`
   - `slope_*`
   - `delta_*_max`
5. Optionally compute prediction stability from per-sample predictions.

### Metrics produced

- Accuracy: `top1`, `top5` (+ bootstrap CI)
- Confidence quality: `nll_mean`, `ece`
- Robustness normalization: `robustness_ratio_top1`
- Trend summaries:
  - `audc_top1`, `audc_nll`, `audc_ece`
  - `slope_top1`, `slope_nll`, `slope_ece`
  - `delta_top1_max`, `delta_nll_max`, `delta_ece_max`
- Invariance proxy:
  - `stability_top1` (if enabled)

---

## Outputs

Each run creates: `results/<run_name>_<timestamp>/`

- `config_snapshot.json` — frozen config used for the run
- `results_repeat.csv` — raw per-repeat model × condition metrics
- `results.csv` — metrics averaged across repeats, with repeat std columns
- `summary_repeat.csv` — raw per-repeat trend summaries
- `summary.csv` — trend summaries averaged across repeats
- `per_sample.csv` — per-sample outputs with repeat IDs (if enabled)
- `stability_repeat.csv` — raw per-repeat prediction stability (if enabled)
- `stability.csv` — prediction stability averaged across repeats (if enabled)
- `top1_<family>.png` — Top-1 degradation curves with bootstrap confidence bands
- `top5_<family>.png` — Top-5 degradation curves with bootstrap confidence bands
- `nll_mean_<family>.png` — negative log-likelihood curves
- `ece_<family>.png` — calibration curves
- `robustness_ratio_top1_<family>.png` — corruption/clean accuracy ratio curves

---

## Setup

1) Install dependencies:

- `pip install -r requirements.txt`

2) Dataset:

- CIFAR-100 is downloaded automatically to the configured root.

---

## Run

Two experiment configurations are available:

### Testing Config (Laptop)

Quick validation run with sparse corruption sweep and three-seed averaging:
- **5 rotation angles**, **5 translation magnitudes**, **4 noise levels** = 14 corrupted conditions + clean
- **20 samples per class** and **3 repeats** keep runtime reasonable for an overnight laptop run
- Command: `python run_experiment.py --config configs/testing_experiment.yaml`

### Production Config (Strong Machine)

Dense corruption sweep for publication-grade analysis:
- **37 rotation angles** (−45° to +45° in 2.5° steps)
- **21 translation magnitudes** (−20 to +20 pixels in 2-pixel steps)
- **31 noise levels** (0.0 to 0.3 σ in 0.01 steps)
- **50 samples per class** and **3 repeats** for stable averaged curves
- **~2,600+ evaluations per model per repeat** for clear degradation trends
- Expect runtime: hours to days depending on hardware
- Command: `python run_experiment.py --config configs/production_experiment.yaml`

---

## Important notes about current state

- The codebase currently uses module-relative imports in [src/main.py](src/main.py). Keep your Python path/launch method consistent with your current working setup.
- `compute_prediction_stability()` is implemented in [src/metrics.py](src/metrics.py) (there is no separate stability module file).
- [src/models.py](src/models.py) supports `resnet101`, `vit_b_16`, and `convnext_small`.
- **Model selection justification**: ResNet-101 (44.5M) and ViT-B/16 (86.6M) provide a 1.9× capacity spread, enabling clean architectural comparison without capacity confounding. ConvNeXt-Small (50.2M) serves as an additional reference point for modern CNN design. This addresses the parameter imbalance problem identified in ImageNet-P and similar robustness studies.

