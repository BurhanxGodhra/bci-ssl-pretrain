import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.full_finetune import kshot_full_finetune_eval
from src.utils.device import get_device

device = get_device()
CHECKPOINT = "checkpoints/encoder_multi_full_e25.pt"

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs_data = load_dataset("bnci2014_001", subjects=holdout_subjects)

K_VALUES = [1, 5, 10, 20]
N_DRAWS = 10

all_results = {}
for subj in holdout_subjects:
    mask = epochs_data.subject_ids == subj
    X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]
    print(f"\n=== Subject {subj} ({len(y_subj)} trials) ===")

    subj_results = {}
    for k in K_VALUES:
        result = kshot_full_finetune_eval(
            CHECKPOINT, X_subj, y_subj, sfreq=epochs_data.sfreq,
            k=k, n_draws=N_DRAWS, device=device,
        )
        subj_results[k] = result
        print(f"  k={k:2d}: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
    all_results[subj] = subj_results

print("\n=== Aggregate: Full Fine-Tune vs Linear Probe ===")
# Linear probe reference from the post-fix sweep
linear_probe_ref = {1: 0.3261, 5: 0.4013, 10: 0.4364, 20: 0.4689}

aggregate = {}
for k in K_VALUES:
    means = [all_results[s][k]["mean_accuracy"] for s in holdout_subjects]
    agg_mean = float(np.mean(means))
    aggregate[k] = {"mean": agg_mean, "std": float(np.std(means))}
    delta = agg_mean - linear_probe_ref[k]
    winner = "fine-tune" if delta > 0 else "linear-probe"
    print(f"  k={k:2d}: fine-tune={agg_mean:.4f}  linear-probe={linear_probe_ref[k]:.4f}  "
          f"delta={delta:+.4f}  ({winner} wins)")

out_path = Path("results/phase5_fewshot_full_finetune_bnci2014_001.json")
with open(out_path, "w") as f:
    json.dump({
        "dataset": "bnci2014_001", "checkpoint": CHECKPOINT,
        "holdout_subjects": holdout_subjects, "k_values": K_VALUES, "n_draws": N_DRAWS,
        "per_subject": {str(s): {str(k): v for k, v in sr.items()} for s, sr in all_results.items()},
        "aggregate": {str(k): v for k, v in aggregate.items()},
        "linear_probe_reference": linear_probe_ref,
        "riemannian_baseline_reference": 0.7232,
    }, f, indent=2)
print(f"\nSaved -> {out_path}")