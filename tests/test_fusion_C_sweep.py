# tests/test_fusion_C_sweep.py
from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.finetune.fusion_eval import kshot_fusion_eval
from src.utils.device import get_device
import numpy as np

device = get_device()
CHECKPOINT = "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt"
encoder = load_pretrained_encoder(CHECKPOINT, device)

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs_data = load_dataset("bnci2014_001", subjects=holdout_subjects)

C_VALUES = [0.01, 0.05, 0.1, 0.3, 1.0]
K = 20
N_DRAWS = 10

print(f"{'C':>8} {'Subj1':>10} {'Subj4':>10} {'Aggregate':>12}")
print("-" * 44)

results_by_C = {}
for C in C_VALUES:
    subj_means = []
    for subj in holdout_subjects:
        mask = epochs_data.subject_ids == subj
        X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]
        result = kshot_fusion_eval(encoder, X_subj, y_subj, k=K, n_draws=N_DRAWS, device=device, C=C)
        subj_means.append(result["mean_accuracy"])

    agg = float(np.mean(subj_means))
    results_by_C[C] = {"subj1": subj_means[0], "subj4": subj_means[1], "aggregate": agg}
    print(f"{C:>8.2f} {subj_means[0]:>10.4f} {subj_means[1]:>10.4f} {agg:>12.4f}")

best_C = max(results_by_C, key=lambda c: results_by_C[c]["aggregate"])
print(f"\nBest C: {best_C} -> {results_by_C[best_C]['aggregate']:.4f}")
print(f"Reference (C=0.1 from previous run): 0.5816")