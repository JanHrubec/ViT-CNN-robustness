# CNN vs ViT Robustness Benchmark (Foundation)

This project is a structured baseline for your EE experiment:

- compare CNNs and ViTs on the same images,
- apply controlled corruptions (rotation, translation, Gaussian noise),
- measure performance degradation in a reproducible way.

The code is intentionally split into small files so you can extend each part independently without rewriting the whole pipeline.

## Project map (what each file does)

### Entry points

- [run_experiment.py](run_experiment.py)
	- Tiny launcher script.
	- Delegates everything to `robustness_lab.main`.

- [src/robustness_lab/main.py](src/robustness_lab/main.py)
	- The orchestrator.
	- Loads config, sets random seeds, picks device.
	- Builds the base dataset once.
	- Iterates over models and corruption specs.
	- Writes metrics to CSV/JSON and generates plots.

### Configuration

- [configs/base_experiment.yaml](configs/base_experiment.yaml)
	- Human-editable experiment settings:
		- dataset type/path,
		- model list,
		- corruption severity lists,
		- evaluation options,
		- output path/run label.

- [src/robustness_lab/config_schema.py](src/robustness_lab/config_schema.py)
	- Typed dataclass schema for config sections.
	- Safe loader from YAML to `ExperimentConfig`.
	- Fills missing fields with defaults.

### Data layer

- [src/robustness_lab/datasets.py](src/robustness_lab/datasets.py)
	- Creates the unmodified base dataset (`ImageFolder` or `CIFAR100`).
	- Optional deterministic subset-per-class sampling.
	- Wraps base dataset with on-the-fly transform for clean/corrupted runs.
	- Builds dataloaders with consistent batch settings.

### Corruption layer

- [src/robustness_lab/corruptions.py](src/robustness_lab/corruptions.py)
	- Defines `CorruptionSpec` (`family`, `severity`, `name`).
	- Expands YAML severity lists into a full evaluation sweep.
	- Applies corruption in tensor space $[0,1]$ before model preprocessing.
	- Current families:
		- `rotation`
		- `translation_x`
		- `translation_y`
		- `gaussian_noise`

### Model layer

- [src/robustness_lab/models.py](src/robustness_lab/models.py)
	- Loads pretrained torchvision models.
	- Packages model + its exact preprocessing transform together.
	- This prevents preprocessing mismatch (a common experimental mistake).

### Evaluation layer

- [src/robustness_lab/runner.py](src/robustness_lab/runner.py)
	- Core inference loop.
	- `evaluate_clean(...)`: baseline performance.
	- `evaluate_corruption(...)`: one corruption setting at a time.
	- Returns standardized `EvalResult` rows.

### Metrics and reporting

- [src/robustness_lab/metrics.py](src/robustness_lab/metrics.py)
	- Computes top-$k$ correctness counters and accuracies.
	- `robustness_ratio(clean, corrupt)`.
	- `audc(...)` (Area Under Degradation Curve, normalized by severity range).
	- Helpers for ECE and bootstrap CI (ready for deeper stats phase).

- [src/robustness_lab/io_utils.py](src/robustness_lab/io_utils.py)
	- Writes JSON snapshots and CSV tables.
	- Generates UTC timestamp for run folders.

- [src/robustness_lab/plots.py](src/robustness_lab/plots.py)
	- Reads result CSV and produces one degradation curve per corruption family.

- [src/robustness_lab/utils.py](src/robustness_lab/utils.py)
	- Reproducibility helpers (`set_global_seed`).
	- Device selection (`cuda`/`mps`/`cpu`).
	- Directory creation helper.

## End-to-end run flow

1. Parse CLI args and load YAML config.
2. Set seed and select device.
3. Build base dataset.
4. Build full corruption sweep list.
5. For each model:
	 - run clean evaluation,
	 - run all corruption evaluations,
	 - compute per-family AUDC summary.
6. Save:
	 - raw pointwise metrics,
	 - summary metrics,
	 - exact config snapshot,
	 - degradation plots.

## Setup

1) Install dependencies

- `pip install -r requirements.txt`

2) Prepare data (ImageNet-style folder)

- `./data/imagenet/val/<class_name>/*.jpg`

If ImageNet is not available yet, switch to `cifar100` in [configs/base_experiment.yaml](configs/base_experiment.yaml).

## Run

- `PYTHONPATH=src python run_experiment.py --config configs/base_experiment.yaml`

## Output files (exact meaning)

Each run creates `results/<run_name>_<timestamp>/` with:

- `results.csv`
	- One row per model × condition.
	- Includes `top1`, `top5`, `severity`, `robustness_ratio_top1`, and bootstrap CIs.

- `per_sample.csv`
	- One row per sample × model × condition.
	- Includes `top1_correct` and `top5_correct`.
	- Useful for paired significance tests and deeper failure analysis.

- `summary.csv`
	- One row per model × corruption family.
	- Includes clean top-1 and normalized AUDC top-1.

- `config_snapshot.json`
	- Frozen copy of the config actually used.
	- Important for reproducibility and writing the methodology section.

- `curve_<family>.png`
	- Severity vs top-1 curve for each corruption family.

## Practical extension points

- Add corruption families in [src/robustness_lab/corruptions.py](src/robustness_lab/corruptions.py).
- Add per-sample logging in [src/robustness_lab/runner.py](src/robustness_lab/runner.py) for stronger statistics.
- Add Grad-CAM and attention rollout modules under [src/robustness_lab](src/robustness_lab).
- Add hypothesis-specific tests (e.g., translation slope comparison) from saved CSV outputs.
