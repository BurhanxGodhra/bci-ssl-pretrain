# tests/test_accuracy_sprint_bnci_only_sweep.py
import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder, kshot_linear_probe_eval
from src.utils.device import get_device

device = get_device()
CHECKPOINT = "checkpoints/encoder_bnci2014_001_only_e25.pt"
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

print("\n=== BNCI-Only vs Combined-Dataset Comparison ===")
combined_ref = {1: 0.3261, 5: 0.4013, 10: 0.4364, 20: 0.4689}
aggregate = {}
for k in K_VALUES:
    means = [all_results[s][k]["mean_accuracy"] for s in holdout_subjects]
    agg = float(np.mean(means))
    aggregate[k] = {"mean": agg, "std": float(np.std(means))}
    delta = agg - combined_ref[k]
    print(f"  k={k:2d}: bnci-only={agg:.4f}  combined={combined_ref[k]:.4f}  "
          f"delta={delta:+.4f}  ({'bnci-only wins' if delta>0 else 'combined wins'})")

out_path = Path("results/phase5_fewshot_linear_probe_bnci_only_e25.json")
with open(out_path, "w") as f:
    json.dump({
        "dataset": "bnci2014_001_only", "checkpoint": CHECKPOINT,
        "holdout_subjects": holdout_subjects, "k_values": K_VALUES, "n_draws": N_DRAWS,
        "per_subject": {str(s): {str(k): v for k, v in sr.items()} for s, sr in all_results.items()},
        "aggregate": {str(k): v for k, v in aggregate.items()},
        "combined_dataset_reference": combined_ref,
    }, f, indent=2)
print(f"\nSaved -> {out_path}")