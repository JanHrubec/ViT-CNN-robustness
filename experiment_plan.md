# Experiment Plan: Robustness of CNNs vs ViTs

## 1) Research focus

**Primary question**  
To what extent does replacing convolution with self-attention affect image-classification robustness under controlled input manipulation (rotation, translation, and noise)?

**Hypothesis (initial)**
- CNNs (with translation-friendly inductive bias) will degrade more slowly under translation.
- ViTs (global attention) may preserve performance better under some global distortions but can be sensitive to positional shifts if positional embeddings are not adapted.
- Noise robustness may depend on frequency profile and training setup more than architecture alone.

## 2) Scope and fairness constraints

To keep the comparison scientifically fair:
1. **Task**: Image classification only.
2. **Training regime**: Start with frozen, pretrained ImageNet-1k models (no finetuning).
3. **Input resolution**: Match each model’s expected resolution; evaluate at a common resolution where feasible.
4. **Evaluation set**: Same image subset and identical corruption pipeline for all models.
5. **Compute budget**: Prefer inference-only first; optional finetuning extension later.

## 3) Model selection strategy

### Phase A (core comparison)
- **CNN**: ResNet-50 (strong, standard baseline)
- **ViT**: ViT-B/16 (standard baseline)

Reason: both are canonical and well-supported in `torchvision`, making results reproducible.

### Phase B (architecture bridge / extension)
- **CNN-modernized**: ConvNeXt-Tiny or ConvNeXt-Base

Reason: tests whether Transformer-like design choices in CNNs close robustness gaps.

### Selection criteria if model swap is needed
- Public pretrained weights available.
- Similar clean top-1 baseline on selected dataset split.
- Runtime feasible on available hardware.

## 4) Dataset choice and sampling

## Preferred route (recommended)
- **ImageNet-1k validation subset** (e.g., 50 images/class = 50,000 is full val; subset could be 10–25 images/class for faster iteration).

Why: direct compatibility with pretrained model priors and external robustness literature.

## Fallback route
- **CIFAR-100** only if compute/storage/access constraints prevent ImageNet subset use.

Important caveat: CIFAR resolution/domain mismatch may confound conclusions for pretrained ImageNet models.

## 5) Corruption/manipulation protocol

Use deterministic, parameterized corruption sweeps.

### 5.1 Rotation
- Angles: \(\theta \in \{0, \pm5, \pm10, \pm15, \pm20, \pm30\}\) degrees
- Interpolation and fill mode fixed across models.

### 5.2 Translation
- Pixel shifts: \((\Delta x, \Delta y)\) with magnitudes \(\{0, 4, 8, 12, 16\}\)
- Test horizontal, vertical, and diagonal variants.

### 5.3 Noise
- Gaussian: \(\sigma \in \{0.00, 0.02, 0.05, 0.10, 0.15\}\) on normalized [0,1] scale.
- Optional extension: salt-and-pepper with density \(p \in \{0.01, 0.03, 0.05\}\).

### 5.4 Frequency-oriented extension (optional)
- Low-pass / high-pass perturbation sets to test frequency sensitivity claims.

## 6) Metrics and analysis

## Primary metrics
1. **Top-1 accuracy** per corruption level.
2. **Relative robustness ratio**:
   $$
   R(c) = \frac{Acc_{corrupt}(c)}{Acc_{clean}}
   $$
3. **Area Under Degradation Curve (AUDC)** over corruption severity.

## Secondary metrics
- Top-5 accuracy (ImageNet context).
- Calibration shift (ECE) under corruption (optional but informative).

## Statistical reporting
- Bootstrap 95% confidence intervals for accuracy and robustness deltas.
- Paired tests over identical samples per model/corruption level.
- Effect size reporting, not only p-values.

## 7) Representation-level interpretation

### CNN
- Grad-CAM on final conv block.

### ViT
- Attention rollout and/or attention distance statistics.

### Comparison goal
- Link failure modes to representation patterns:
  - local texture collapse vs global structure loss,
  - center-bias changes under translation,
  - attention dispersion under noise.

## 8) Experimental workflow

1. Build clean evaluation loader.
2. Add corruption generator with fixed random seeds.
3. Run baseline clean accuracy checks.
4. Run corruption sweeps per model.
5. Save per-sample predictions/logits + metadata.
6. Aggregate metrics and confidence intervals.
7. Generate plots and saliency/attention case studies.
8. Write interpretation tied to theory (equivariance, positional encoding, inductive bias).

## 9) Reproducibility requirements

- Fixed global seed(s).
- Version-pinned dependencies.
- Config-driven experiments (YAML/JSON).
- Cached transformed datasets where possible.
- Automatic logging (CSV + plots + run config snapshot).

## 10) Risks and mitigations

1. **Model unfairness from parameter count differences**  
   Mitigation: report count/compute and include ConvNeXt bridge model.

2. **Dataset-domain mismatch (if CIFAR fallback used)**  
   Mitigation: clearly label as exploratory; avoid over-generalization.

3. **Corruption implementation artifacts**  
   Mitigation: visually inspect random samples at each severity level.

4. **Compute/time limits**  
   Mitigation: pilot on small subset; scale only once pipeline is validated.

## 11) Ethical and academic integrity checklist

- Cite datasets, model sources, and all prior papers used.
- Separate internship deliverables from EE contributions in documentation.
- Use pretrained models where possible to reduce energy cost.
- Report negative/ambiguous results transparently.

## 12) Proposed implementation milestones

### Milestone 1 (Pilot, 2–4 days)
- ResNet-50 vs ViT-B/16, small ImageNet subset, rotation+translation only.
- Deliver: first degradation curves and sanity-check visualizations.

### Milestone 2 (Core, 4–7 days)
- Add Gaussian noise sweep, confidence intervals, better plotting.
- Deliver: robust comparative tables and per-corruption analysis.

### Milestone 3 (Extension, 3–5 days)
- Add ConvNeXt and saliency/attention interpretation section.
- Deliver: architecture-transition discussion with evidence.

## 13) Immediate next decisions needed

1. Confirm dataset route: ImageNet subset vs CIFAR-100 fallback.
2. Confirm model set for core run:
   - minimal: ResNet-50 + ViT-B/16
   - extended: + ConvNeXt
3. Confirm target runtime budget (single GPU/CPU, max hours).
4. Confirm whether calibration metric (ECE) is in scope.

---

If approved, the next step will be to translate this plan into a reproducible project structure (configs, data pipeline, evaluation scripts, plotting, and interpretation utilities).