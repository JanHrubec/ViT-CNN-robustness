# CNN vs ViT Robustness Benchmark (EE Foundation)

This project implements the full experimental plan for comparing robustness of CNNs and Vision Transformers under controlled corruptions. The code is modular, reproducible, and designed to produce interpretable, paper-ready results.

---

## 1) Research focus

**Primary question**  
To what extent does the architectural transition from convolution to self‑attention affect model robustness under rotation, translation, and noise?

**Hypothesis (initial)**
- CNNs (translation‑friendly inductive bias) will degrade more slowly under translation.
- ViTs (global attention from layer 1) may preserve performance under some global distortions but can be sensitive to positional shifts.
- Noise robustness may be more dependent on frequency bias and training priors than architecture alone.

---

## 2) Scope and fairness constraints

1. **Task**: Image classification only.
2. **Training regime**: Start with frozen, pretrained ImageNet‑1k models (no finetuning).
3. **Input resolution**: Use each model’s official preprocessing pipeline to avoid mismatched normalization or resizing.
4. **Evaluation set**: Identical dataset subset and identical corruption pipeline across all models.
5. **Compute budget**: Inference‑only first, optional finetuning extension if needed.

---

## 3) Model selection strategy

### Core comparison (Phase A)
- **CNN**: ResNet‑50
- **ViT**: ViT‑B/16

Rationale: canonical baselines used widely in robustness literature and fully supported by torchvision.

### Architecture bridge (Phase B)
- **CNN‑modernized**: ConvNeXt‑Tiny (optional)

Rationale: tests whether transformer‑style design choices in CNNs close robustness gaps.

### Model swap criteria
- Public pretrained weights available.
- Similar clean top‑1 baseline on chosen dataset.
- Runtime feasible on available hardware.

---

## 4) Dataset choice and sampling

### Dataset used
- **CIFAR‑100** (fixed for this project)

---

## 5) Corruption/manipulation protocol

**Rotation**  
$$\theta \in \{-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30\}$$

**Translation**  
$$\Delta x, \Delta y \in \{-16, -12, -8, -4, 0, 4, 8, 12, 16\}$$

**Noise (Gaussian)**  
$$\sigma \in \{0.00, 0.02, 0.05, 0.10, 0.15\}$$

All corruptions are applied in tensor space $[0,1]$ before model‑specific preprocessing so every architecture sees identical corrupted content.

---

## 6) Metrics and analysis

### Primary metrics
1. **Top‑1 accuracy** per corruption level.
2. **Robustness ratio**:  
$$R(c) = \frac{Acc_{corrupt}(c)}{Acc_{clean}}$$
3. **AUDC**: normalized area under degradation curve.

### Secondary metrics (configurable)
- **Top‑5 accuracy** (ImageNet context)
- **NLL (cross‑entropy)**
- **ECE** (calibration under corruption)

### Consistency / invariance metrics
For rotation/translation, **prediction stability** across small shifts can directly test invariance claims. This measures agreement between predictions under a neighborhood of minor transforms and is a strong complement to accuracy‑only metrics.

---

## 7) Methodology decisions (with trade‑offs)

### Why top‑$k$ is still the baseline
- **Pros**: standard in ImageNet robustness literature, easy to interpret, comparable across papers.
- **Cons**: hides confidence shifts and probability mass changes.

### Why add NLL and ECE (optional)
- **NLL** captures confidence degradation even when top‑1 remains correct.
- **ECE** measures reliability, which is critical in safety‑sensitive scenarios.

### Why keep metrics configurable
Different sections of the paper emphasize different properties (accuracy vs calibration vs invariance). Configurable metrics prevent overfitting your methodology to one interpretation and keep results transparent.

---

## 8) Interpretation layer

### CNN
- Grad‑CAM on final convolutional block.

### ViT
- Attention rollout or attention distance statistics.

Interpretability should link observed failures to the expected behavior of locality vs global attention.

---

## 9) Reproducibility requirements

- Fixed random seeds.
- Config‑driven experiments (YAML + snapshot).
- Version‑pinned dependencies.
- Consistent preprocessing with official model weights.
- Automatic CSV and plot logging.

---

## 10) Project map (what each file does)

### Entry points
- [run_experiment.py](run_experiment.py): Tiny launcher.
- [src/robustness_lab/main.py](src/robustness_lab/main.py): Experiment orchestrator.

### Configuration
- [configs/base_experiment.yaml](configs/base_experiment.yaml): All experiment settings.
- [src/robustness_lab/config_schema.py](src/robustness_lab/config_schema.py): Typed config loader.

### Data layer
- [src/robustness_lab/datasets.py](src/robustness_lab/datasets.py): Dataset building + subset sampling + index‑safe wrappers.

### Corruption layer
- [src/robustness_lab/corruptions.py](src/robustness_lab/corruptions.py): Rotation/translation/noise sweeps.

### Model layer
- [src/robustness_lab/models.py](src/robustness_lab/models.py): Pretrained model loading + transforms.

### Evaluation and metrics
- [src/robustness_lab/runner.py](src/robustness_lab/runner.py): Clean + corrupted evaluation loops.
- [src/robustness_lab/metrics.py](src/robustness_lab/metrics.py): Top‑$k$, NLL, ECE, AUDC, bootstrap CI.

### Outputs and plots
- [src/robustness_lab/io_utils.py](src/robustness_lab/io_utils.py): CSV/JSON writing.
- [src/robustness_lab/plots.py](src/robustness_lab/plots.py): Degradation curves.

---

## 11) End‑to‑end run flow

1. Load YAML config.
2. Set seed, select device.
3. Build base dataset (optionally subset‑per‑class).
4. Build corruption sweep list.
5. For each model:
	 - clean evaluation,
	 - corrupted evaluations,
	 - AUDC summary per corruption family.
6. Save artifacts and plots.

---

## 12) Setup

1) Install dependencies

- `pip install -r requirements.txt`

2) Prepare data

- CIFAR‑100 will be downloaded automatically to `./data/cifar100`.

---

## 13) Run

- `PYTHONPATH=src python run_experiment.py --config configs/base_experiment.yaml`

---

## 14) Output files (exact meaning)

Each run creates `results/<run_name>_<timestamp>/` with:

- `results.csv`  
	One row per model × condition with enabled metrics.

- `per_sample.csv`  
	One row per sample × model × condition with dataset‑level `sample_index` for paired stats.

- `stability.csv`  
	Only written when `metrics.enable_stability=true`.  
	Reports top‑1 prediction agreement with clean predictions for each corruption setting.

- `summary.csv`  
	Per‑model × per‑family AUDC summary.

- `config_snapshot.json`  
	Frozen config for reproducibility.

- `curve_<family>.png`  
	Degradation curves per corruption family.

---

## 15) Practical extension points

- Add additional corruption families in [src/robustness_lab/corruptions.py](src/robustness_lab/corruptions.py).
- Add Grad‑CAM and attention rollout modules under [src/robustness_lab](src/robustness_lab).
- Add prediction stability metrics for rotation/translation invariance tests.
- Add statistical report generation from per‑sample outputs.
