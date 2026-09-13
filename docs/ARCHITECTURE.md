# Architecture

This document covers system design decisions and *why*, not just *what* — the reasoning behind each major choice, since several of them were revised mid-project after evidence contradicted the initial assumption.

## Pipeline overview

Raw EEG (MOABB) → Channel Alignment → Per-Trial Normalization → Augmentation
→ Contrastive Pretraining (NT-Xent) → Frozen Encoder
→ [Linear Probe | Full Fine-Tune | Riemannian Fusion] → Few-Shot Accuracy


## Data layer

### Subject-level leakage firewall

Every dataset has a committed JSON split manifest (`data/splits/*.json`) dividing subjects into a pretrain pool and a holdout set (seeded, deterministic). `PretrainContrastiveDataset` filters through this manifest in its constructor — not as a caller convention, but structurally, so pretraining on a holdout subject requires actively bypassing the class rather than forgetting a parameter. This firewall was verified with an explicit assertion test (`tests/test_phase3_dataset.py`) confirming holdout trials are excluded before any real training run.

### Channel alignment and normalization

`align_epochs()` performs three operations when combining datasets with different montages:

1. **Channel name intersection** — finds the common electrode set across datasets by normalized name matching, not position index. A latent bug here (channel_names pulled from the raw MNE object included non-EEG channels like EOG/STI that MOABB's paradigm had already excluded from the actual data array) was caught by a shape-mismatch assertion, not silently — see the "bugs found" section below.
2. **Timepoint cropping** — datasets with different native sample rates produce off-by-one-sample epoch lengths even after resampling to a common target rate (BNCI2014_001: 1001 samples; PhysionetMI: 1002 at the same nominal rate/duration). All sources are cropped to the shared minimum length.
3. **Per-trial z-score normalization** — the single most important fix in this project. Without it, BNCI2014_001 (std≈7.9) and PhysionetMI (std≈19.9) differ in raw amplitude by ~2.5x, giving contrastive pretraining a trivial shortcut ("which dataset has bigger numbers") that has nothing to do with motor-imagery-relevant signal. Confirmed via silhouette-score diagnostics before and after (see `BENCHMARKS.md`).

## Model

### EEGNet-style encoder

Three-block architecture, chosen specifically to mirror the classical baseline's inductive bias for a fair comparison:

1. **Temporal convolution** — learnable frequency-selective filters applied per-channel (like a learned bandpass filter bank).
2. **Depthwise spatial convolution** — learns spatial patterns across the full channel montage (analogous to what the Riemannian baseline's covariance step captures explicitly).
3. **Separable convolution** — further temporal refinement, then pointwise mixing across spatial filters.

Chosen over a Transformer for parameter efficiency (~thousands, not millions, of parameters) appropriate to the pretraining set size (a few thousand trials, not millions) — a large Transformer would be expected to overfit or require substantially more data than was available.

A **random-weight control** (same architecture, no training) was run as a mandatory baseline before trusting any pretrained-encoder result: it reached 39.1% at k=20 on its own, confirming the architecture's inductive bias carries real signal independent of pretraining — pretraining adds a further, growing +7.8pt on top of that floor, not the entire result.

### Contrastive pretraining (NT-Xent)

Two augmented views of each trial are pulled together in embedding space (positive pair); all other trials in the batch are pushed apart (negatives). Temperature=0.5, embedding dim=128, projection dim=64. The projection head is discarded after pretraining; only the encoder is reused downstream, per standard SimCLR convention.

**What was tuned, and what turned out not to matter**: dataset combination (neutral), LR schedule shape (flat LR slightly outperformed cosine decay, but not meaningfully for downstream accuracy), and augmentation strength (small real gain, +1.2pt). None of these closed the gap to the classical baseline alone. See `BENCHMARKS.md` for the full sequence and the reasoning for why each was ruled out.

### Few-shot adaptation: three methods, one clear winner

- **Linear probe** (frozen encoder + linear head): the standard SSL evaluation protocol, isolates representation quality from adaptation dynamics.
- **Full fine-tune** (unfrozen encoder + linear head, low LR, augmentation-regularized): tested to see whether adaptation beats a frozen representation at this data scale. It does not — linear probing wins at every k≥5, and the gap *widens* with more data (opposite of what an overfitting-that-resolves-with-more-data hypothesis would predict), suggesting the unfrozen encoder settles into a worse local representation early and plateaus rather than genuinely improving with more labels.
- **SSL + Riemannian fusion** (frozen encoder embedding concatenated with tangent-space features, both fit per k-shot support set, standardized, classified with regularized Logistic Regression): the method that actually closed most of the gap to the classical ceiling. Tangent-space features must be fit on the support set only (never the query set) to avoid leakage — the same discipline as any proper train/test split, just applied manually since k-shot isn't standard cross-validation.

## Classical baseline

Covariance estimation (Ledoit-Wolf shrinkage, since raw sample covariance is noisy/near-singular with few timepoints relative to channels) → Riemannian tangent-space projection → Logistic Regression. This is the field's actual gold-standard classical approach, not a strawman — chosen specifically so a negative comparison (SSL failing to beat it) would be a real, publishable finding rather than an artifact of a weak baseline.

## Applications

### Device Setup Wizard

Built on one principle: the same LSL-reading code path must handle both real hardware and our "try it now" simulated mode — the app never has a separate untested branch for the real-hardware case. "Try it now" works by publishing our own holdout-subject data through a genuine `pylsl.StreamOutlet`, which the reading code then discovers via the same `pylsl.resolve_byprop` call it would use for real hardware. A hard channel-count compatibility gate prevents the wizard from silently producing meaningless output if a connected device's montage doesn't match the encoder's requirements.

Calibration cannot self-report accuracy honestly (the only "label" available is what the user was told to imagine, not verified ground truth) — so calibration ends with "classifier trained," and a separate Live Test step provides the honest accuracy signal via fresh, independently-cued trials.

### Benchmarking Lab

Deliberately scoped to evaluate multiple datasets **independently**, never combined at the label level — BNCI2014_001 and PhysionetMI have different, non-comparable class taxonomies, so a single classifier can't meaningfully predict across both. Dataset combination is only valid at the (unsupervised) pretraining level, which this project already supports separately via `scripts/pretrain.py`'s multi-dataset mode.

Custom dataset upload enforces exact channel **identity** matching (not just channel count) against the encoder's required 22-channel montage, since the encoder's spatial convolution weights are tied to specific electrode positions learned during pretraining — a dataset with the right number of channels in the wrong positions would silently produce garbage without this check.

## Notable bugs found and fixed during development

Documented here because they materially shaped the results, not swept into commit messages:

1. **Amplitude-scale shortcut** — combining unnormalized datasets let contrastive pretraining solve the task via raw signal magnitude instead of learning real structure. Diagnosed via a UMAP + silhouette-score investigation, confirmed via direct std comparison, fixed via per-trial normalization.
2. **Channel-name mismatch** — `load_dataset()` pulled channel names from the raw MNE recording (which includes non-EEG channels like EOG/STI) while the actual data array had already been filtered to EEG-only by MOABB's paradigm — a silent off-by-N bug that happened to self-correct when combining two datasets (intersection filtered out the spurious names) but surfaced immediately on single-dataset use. Fixed with an explicit shape-match assertion at the source, not just downstream.
3. **LSL replay outlet running dry mid-session** — the simulated device stream played its pre-loaded data once, at real wall-clock rate, and went silent once exhausted — causing the Device Setup Wizard to stall mid-calibration once the session ran longer than the outlet's data length. Fixed by looping the replay indefinitely.
4. **MNE_DATA config drift** — a stale environment/config value from an unrelated local project caused dataset downloads to fail with `FileNotFoundError` partway through a benchmarking session; diagnosed by inspecting the full MNE config dictionary for any value pointing at the broken path, rather than patching the single symptom that surfaced first.