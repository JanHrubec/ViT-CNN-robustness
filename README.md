# CNN vs ViT Robustness Benchmark

This repository runs a modular robustness benchmark comparing a plain CNN, a Vision Transformer, and a modern CNN on **ImageNet-1k validation** under controlled input corruptions.

The project is currently **inference-only**, with dense corruption sweeps and configurable metrics for clearer degradation trends. **Three models of comparable capacity** are benchmarked to isolate architectural differences from capacity confounds.

---

## Current project scope

- Dataset: ImageNet-1k validation subset (streamed from Hugging Face; cached locally)
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
- [src/datasets.py](src/datasets.py) — ImageNet-1k val subset streaming + cache, dataloader
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

The main experiment configuration is [configs/base.yaml](configs/base.yaml).

### Dataset block

- `n_per_class`: balanced per-class evaluation size (e.g. 5 → 5000 images total)
- `pool_per_class`: how many images per class to stream/cache before sampling `n_per_class` (max 50 on val)
- `cache_dir`: where the streamed per-class pools are stored for reuse
- `batch_size`, `num_workers`

Prerequisite: run `huggingface-cli login` once with a token that has accepted the dataset terms at `https://huggingface.co/datasets/ILSVRC/imagenet-1k`.

### Running

```bash
cd /Users/jenda/Desktop/School/IB/EE/Implementation
source .venv/bin/activate

# Default run (uses configs/base.yaml if you omit --config)
python run_experiment.py --config configs/base.yaml

# Quick sanity check (1000 images, clean-only)
python run_experiment.py \
  --config configs/base.yaml \
  --override evaluation.num_repeats=1 \
  --override dataset.n_per_class=1 \
  --override corruptions.rotation_degrees='[]' \
  --override corruptions.translation_pixels='[]' \
  --override corruptions.gaussian_sigmas='[0.0]'
```

### Outputs and incremental writes

Each run writes under `results/<run_name>_<timestamp>/` as it progresses (so a long sweep does not need to hold all rows in RAM):

- `results_repeat.csv`, `per_sample.csv`, `summary_repeat.csv`, and `stability_repeat.csv` are **appended** after each evaluation stage.
- `results_intermediate.csv` is refreshed after each model finishes (aggregate of `results_repeat.csv` so far).
- After the full run: `results.csv`, `summary.csv`, `stability.csv`, `metrics_manifest.json`, and PNG plots.
- `corruption_previews/repeat_<i>_seed_<s>/` contains the **first dataset image** for that repeat (reference class-balanced order) under every corruption, plus `00_clean_reference.png`.

### Models block

- `names`: list of model identifiers to run
  - You can use **aliases** (defaults to ImageNet-1k-only timm checkpoints):
    - `resnet101` / `resnet101_in1k` → `resnet101.a1_in1k`
    - `vit_b_16` / `vit_b16_in1k` → `vit_base_patch16_224.augreg_in1k`
    - `convnext_small` / `convnext_small_in1k` → `convnext_small.fb_in1k`
  - Or you can put an **explicit timm model id** directly in `names` (for alternative checkpoints), e.g. `resnet101.a1_in1k`.

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

- ImageNet-1k validation is streamed from Hugging Face and cached as per-class JPEG pools under `dataset.cache_dir`.

---

## Run

`python run_experiment.py --config configs/base.yaml`