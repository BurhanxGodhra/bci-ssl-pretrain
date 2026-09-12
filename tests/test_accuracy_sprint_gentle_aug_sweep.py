# tests/test_accuracy_sprint_gentle_aug_sweep.py
import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder, kshot_linear_probe_eval
from src.utils.device import get_device

device = get_device()
CHECKPOINT = "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt"
encoder = load_pretrained_encoder(CHECKPOINT, device)

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs_data = load_dataset("bnci2014_001", subjects=holdout_subjects)

K_VALUES = [1, 5, 10, 20]
N_DRAWS = 10
all_results = {}

for subj in holdout_subjects:
    mask = epochs_data.subject_ids == subj
    X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]
    print(f"\n=== Subject {subj} ===")
    subj_results = {}
    for k in K_VALUES:
        result = kshot_linear_probe_eval(encoder, X_subj, y_subj, k=k, n_draws=N_DRAWS, device=device)
        subj_results[k] = result
        print(f"  k={k:2d}: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
    all_results[subj] = subj_results

print("\n=== Gentle-Aug vs Strong-Aug (both BNCI-only, flat LR) ===")
strong_aug_ref = {1: 0.3241, 5: 0.3990, 10: 0.4313, 20: 0.4535}
aggregate = {}
for k in K_VALUES:
    means = [all_results[s][k]["mean_accuracy"] for s in holdout_subjects]
    agg = float(np.mean(means))
    aggregate[k] = {"mean": agg, "std": float(np.std(means))}
    delta = agg - strong_aug_ref[k]
    print(f"  k={k:2d}: gentle-aug={agg:.4f}  strong-aug={strong_aug_ref[k]:.4f}  delta={delta:+.4f}")

out_path = Path("results/phase5_fewshot_linear_probe_bnci_only_gentle_aug.json")
with open(out_path, "w") as f:
    json.dump({
        "dataset": "bnci2014_001_only_flat_gentleaug", "checkpoint": CHECKPOINT,
        "holdout_subjects": holdout_subjects, "k_values": K_VALUES, "n_draws": N_DRAWS,
        "per_subject": {str(s): {str(k): v for k, v in sr.items()} for s, sr in all_results.items()},
        "aggregate": {str(k): v for k, v in aggregate.items()},
        "strong_aug_reference": strong_aug_ref,
    }, f, indent=2)
print(f"\nSaved -> {out_path}")