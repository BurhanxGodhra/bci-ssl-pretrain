# Benchmarks

Every experiment referenced in the README, including the ones that didn't work. Numbers here are the actual output of the scripts in `tests/`, not curated for presentation.

## Classical baseline (Riemannian tangent-space, full labeled data, 5-fold CV)

### BNCI2014_001 (holdout subjects 1, 4)

| Subject | Accuracy |
|---|---|
| 1 | 80.7% ± 3.3% |
| 4 | 63.9% ± 4.2% |
| **Overall** | **72.3% ± 8.4%** |

### PhysionetMI (holdout subjects 6, 7)

| Subject | Accuracy |
|---|---|
| 6 | 49.4% ± 3.1% |
| 7 | 85.7% ± 8.6% |
| **Overall** | **67.6% ± 18.1%** |

PhysionetMI's much larger spread is consistent with its known lower data-collection consistency relative to a curated competition dataset like BNCI2014_001, not an error in the baseline itself.

## Random-weight encoder control (mandatory before trusting any pretrained result)

Same EEGNet architecture, no training, linear probe on top:

| k | Accuracy |
|---|---|
| 1 | 32.0% |
| 5 | 36.8% |
| 10 | 37.5% |
| 20 | 39.1% |

Confirms the architecture's inductive bias carries real signal above chance (25%) independent of pretraining — this is the floor every pretrained result must be compared against, not zero.

## Pretraining scale-up experiments

Three independent interventions were tested to push past the ~45-47% plateau. All three failed to improve results meaningfully — reported here in full because ruling things out is real information, not a wasted effort.

### 1. Dataset combination (BNCI-only vs. combined BNCI+PhysionetMI pretraining)

| k | BNCI-only | Combined | Delta |
|---|---|---|---|
| 1 | 32.7% | 32.6% | +0.1 |
| 5 | 40.7% | 40.1% | +0.6 |
| 10 | 43.9% | 43.6% | +0.2 |
| 20 | 46.0% | 46.9% | -0.9 |

**Verdict: neutral.** Combining datasets neither helped nor hurt meaningfully.

### 2. LR schedule (flat vs. cosine decay, both BNCI-only)

| k | Flat LR | Cosine LR | Delta |
|---|---|---|---|
| 1 | 32.4% | 32.7% | -0.3 |
| 5 | 39.9% | 40.7% | -0.8 |
| 10 | 43.1% | 43.9% | -0.7 |
| 20 | 45.4% | 46.0% | -0.7 |

Flat LR produced a **lower pretraining loss** (3.25 vs. 3.30 at epoch 25 — confirmed via a step-matched diagnostic table) but **worse downstream accuracy** at every k. **Verdict: neutral to slightly negative.** This was the first clear signal that pretraining loss is not a reliable proxy for few-shot accuracy in this setup.

### 3. Augmentation strength (gentle vs. original strong augmentation, both BNCI-only, flat LR)

| k | Gentle | Strong | Delta |
|---|---|---|---|
| 1 | 32.8% | 32.4% | +0.4 |
| 5 | 41.0% | 39.9% | +1.1 |
| 10 | 43.0% | 43.1% | -0.1 |
| 20 | 46.6% | 45.4% | +1.3 |

**Verdict: small real gain**, but nowhere near closing the gap to the classical baseline alone.

## Few-shot method comparison (BNCI2014_001, gentle-aug flat-LR checkpoint)

### Linear probe

| k | Accuracy |
|---|---|
| 1 | 32.6% |
| 5 | 40.1% |
| 10 | 43.6% |
| 20 | 46.9% |

### Full fine-tune

| k | Accuracy |
|---|---|
| 1 | 34.3% |
| 5 | 38.6% |
| 10 | 39.6% |
| 20 | 40.4% |

Linear probe beats fine-tuning at every k≥5, and the gap *widens* with more data (46.9% vs. 40.4% at k=20, a 6.5pt gap) — opposite of what "fine-tuning just needs more data" would predict. The unfrozen encoder appears to settle into a worse representation early and plateau, rather than genuinely improving.

## SSL + Riemannian fusion (the method that worked)

### BNCI2014_001

| k | Accuracy | Δ vs. linear probe alone |
|---|---|---|
| 1 | 38.0% | +5.4 |
| 5 | 47.0% | +6.9 |
| 10 | 52.0% | +8.4 |
| 20 | 58.2% | +11.3 |

The gap over SSL-only linear probing grows with k — a real, structural improvement, not noise.

### Regularization sweep (k=20, confirms 58.2% is a genuine ceiling, not an artifact of C=0.1)

| C | Aggregate accuracy |
|---|---|
| 0.01 | 57.0% |
| 0.05 | 58.2% |
| 0.10 | 58.2% |
| 0.30 | 58.0% |
| 1.00 | 58.0% |

Flat across C=0.05-1.0 (spread of 0.2pt, within noise); only strong under-regularization (C=0.01) hurt. Confirms the result is not regularization-limited.

### Cross-dataset comparison (via Benchmarking Lab)

| k | BNCI2014_001 | PhysionetMI |
|---|---|---|
| 1 | 37.7% | 30.2% |
| 5 | 47.8% | 42.0% |
| 10 | 52.1% | 48.3% |
| 20 | 58.0% | 53.1% |

## Embedding diagnostics

Silhouette scores on frozen encoder embeddings, computed both before and after the amplitude-normalization fix:

| Grouping | Before fix (Euclidean) | After fix (Euclidean) | After fix (Cosine) |
|---|---|---|---|
| By class | -0.076 | -0.066 | -0.107 |
| By subject | -0.124 | -0.062 | -0.083 |
| By dataset | **+0.428** | +0.074 | +0.131 |

The pre-fix `by_dataset` score of +0.43 was the direct evidence of the amplitude-shortcut bug — the encoder was organizing its embedding space around "which dataset" far more than any other signal. Post-fix, no single grouping dominates.

A follow-up UMAP robustness check (n_neighbors = 5, 15, 50) showed two subjects with visually stable cross-scale clustering while most others merged inconsistently across settings — consistent with the near-zero global silhouette score, and suggesting a small number of subjects have genuinely distinctive signal characteristics rather than the whole cohort clustering by identity.

## Calibration-time framing

Trial timing: 4s imagery window + 2s cue + 2s rest = 8s/trial (BNCI2014_001 protocol).

| k | Trials | Time | SSL+Fusion accuracy | % of Riemannian ceiling (72.3%) |
|---|---|---|---|---|
| 1 | 4 | 0.5 min | 38.0% | 52.5% |
| 5 | 20 | 2.7 min | 47.0% | 65.0% |
| 10 | 40 | 5.3 min | 52.0% | 71.9% |
| 20 | 80 | 10.7 min | 58.2% | 80.4% |
| Full (576 trials) | 576 | 76.8 min | 72.3% | 100% |

## ONNX export verification

| Check | Result |
|---|---|
| Structural validation (ONNX checker) | Pass |
| Numerical parity, 10 random inputs, PyTorch vs. ONNX Runtime | Max abs diff: 2.98e-07 (tolerance: 1e-04) |

## Reproducing these numbers

Every table above corresponds to a script in `tests/`. The naming convention follows the experiment: `test_phaseN_*.py` for the phased build, `test_accuracy_sprint_*.py` for the negative-result sequence, `test_fusion_*.py` for the winning method. Random seeds are fixed throughout (`seed=42` base, with derived per-draw seeds for k-shot sampling) — re-running should reproduce these numbers within the small noise band reported alongside each mean.