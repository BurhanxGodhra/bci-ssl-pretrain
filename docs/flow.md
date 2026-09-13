# System & Data Flow

Step-by-step flows for each major pipeline in this project. Complements `ARCHITECTURE.md` (component design) with process-level detail.

## 1. Pretraining data flow

```mermaid
flowchart TD
    A[MOABB: BNCI2014_001, PhysionetMI] --> B[load_dataset<br/>per-dataset epoching]
    B --> C[align_epochs<br/>channel intersection + timepoint crop]
    C --> D[normalize_trials<br/>per-trial z-score]
    D --> E[get_or_create_split<br/>subject-level pretrain/holdout manifest]
    E -->|pretrain subjects only| F[PretrainContrastiveDataset]
    F -->|per trial| G[2x augmentation pipeline<br/>independent RNG streams]
    G --> H[view1, view2]
    H --> I[EEGNetEncoder + ProjectionHead]
    I --> J[NT-Xent Loss]
    J -.backprop.-> I
    I -->|after training| K[Frozen Encoder Checkpoint .pt]
```

## 2. Few-shot evaluation flow

```mermaid
flowchart TD
    A[Frozen Encoder Checkpoint] --> B{Method}
    B -->|Linear Probe| C[Freeze encoder<br/>train linear head only]
    B -->|Full Fine-Tune| D[Unfreeze encoder<br/>low-LR joint training + augmentation]
    B -->|SSL+Riemannian Fusion| E[Frozen embedding<br/>+ tangent-space features]

    F[Holdout subject trials] --> G[sample_k_shot_split<br/>support set, query set]
    G -->|support| C
    G -->|support| D
    G -->|support: fit tangent-space + classifier| E
    G -->|query: evaluate only| H[Held-out accuracy]

    C --> H
    D --> H
    E --> H
```

## 3. Device Setup Wizard user flow

```mermaid
sequenceDiagram
    participant User
    participant Wizard as Setup Wizard
    participant LSL
    participant Encoder as Frozen Encoder

    User->>Wizard: Select real device or "Try it now"
    alt Real device
        Wizard->>LSL: resolve_byprop("type", "EEG")
        LSL-->>Wizard: Stream found / not found
    else Try it now
        Wizard->>LSL: start_replay_outlet (holdout subject data, looped)
        Wizard->>LSL: resolve_byprop("type", "EEG")
        LSL-->>Wizard: Stream found (same code path)
    end
    Wizard->>LSL: pull_chunk (2s sample)
    Wizard->>Wizard: check_signal_quality (flat/noisy channel check)
    alt Signal OK
        Wizard->>User: Proceed to calibration
    else Signal issue
        Wizard->>User: Report specific channels, offer re-check or override
    end
    loop k trials per class, randomized order
        Wizard->>User: Countdown + cue (e.g. "LEFT HAND")
        Wizard->>LSL: pull_chunk (4s imagery window)
        Wizard->>Wizard: Store (trial, label)
    end
    Wizard->>Encoder: embed_all(support set)
    Wizard->>Wizard: fit tangent-space + LogisticRegression on support
    Wizard->>User: "Classifier trained" (no accuracy claimed yet)
    loop Live test trials
        Wizard->>User: Countdown + cue (fresh, random)
        Wizard->>LSL: pull_chunk (4s)
        Wizard->>Encoder: embed + tangent-space transform
        Wizard->>Wizard: predict
        Wizard->>User: Show cued vs. predicted, running accuracy
    end
```

## 4. Benchmarking Lab flow

```mermaid
flowchart TD
    A[Select encoder checkpoint] --> B[Select 1+ built-in datasets<br/>and/or upload custom]
    B -->|Upload path| C[validate_and_adapt_upload]
    C -->|channel identity check| D{All 22 required<br/>channels present?}
    D -->|No| E[Report missing channels<br/>stop]
    D -->|Yes| F[Reorder to required order<br/>normalize]

    B -->|Built-in path| G[get_or_create_split<br/>load_dataset holdout subjects]
    F --> H[Per-source independent evaluation]
    G --> H
    H --> I[kshot_fusion_eval<br/>across selected k values]
    I --> J[Per-source results table]
    J --> K[Minimum-trial finder<br/>vs. target accuracy]

    style H fill:#12151C,stroke:#1F232D,color:#E9EBF0
```

Note: sources in step H are evaluated **independently** — never combined at the label/classifier level (see `docs/decisions.md`, D-014).