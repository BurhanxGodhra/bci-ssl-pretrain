import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder, kshot_linear_probe_eval
from src.utils.device import get_device

device = get_device()
encoder = load_pretrained_encoder("checkpoints/encoder_multi_bnci_physionet.pt", device)

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs = load_dataset("bnci2014_001", subjects=holdout_subjects)

K_VALUES = [1, 5, 10, 20]
N_DRAWS = 10  # more draws for smaller k, since variance is higher there

all_results = {}

for subj in holdout_subjects:
    mask = epochs.subject_ids == subj
    X_subj, y_subj = epochs.X[mask], epochs.y[mask]
    print(f"\n=== Subject {subj} ({len(y_subj)} trials) ===")

    subj_results = {}
    for k in K_VALUES:
        result = kshot_linear_probe_eval(
            encoder, X_subj, y_subj, k=k, n_draws=N_DRAWS, device=device
        )
        subj_results[k] = result
        print(f"  k={k:2d}: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")

    all_results[subj] = subj_results

# Aggregate across subjects, per k
print("\n=== Aggregate across holdout subjects ===")
aggregate = {}
for k in K_VALUES:
    means = [all_results[s][k]["mean_accuracy"] for s in holdout_subjects]
    aggregate[k] = {"mean": float(np.mean(means)), "std": float(np.std(means))}
    print(f"  k={k:2d}: {aggregate[k]['mean']:.4f} (+/- {aggregate[k]['std']:.4f})")

# Persist for Phase 6 plotting
out_path = Path("results/phase5_fewshot_linear_probe_bnci2014_001.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "dataset": "bnci2014_001",
        "checkpoint": "encoder_multi_bnci_physionet.pt",
        "holdout_subjects": holdout_subjects,
        "k_values": K_VALUES,
        "n_draws": N_DRAWS,
        "per_subject": {
            str(s): {str(k): v for k, v in subj_results.items()}
            for s, subj_results in all_results.items()
        },
        "aggregate": {str(k): v for k, v in aggregate.items()},
        "riemannian_baseline_reference": {"overall_mean_accuracy": 0.7232},
    }, f, indent=2)
print(f"\nSaved -> {out_path}")