# tests/test_fusion_sweep.py
import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.finetune.fusion_eval import kshot_fusion_eval
from src.utils.device import get_device

device = get_device()
# Use our best-so-far checkpoint (gentle aug, flat LR)
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
        result = kshot_fusion_eval(encoder, X_subj, y_subj, k=k, n_draws=N_DRAWS, device=device)
        subj_results[k] = result
        print(f"  k={k:2d}: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
    all_results[subj] = subj_results

print("\n=== Fusion vs SSL-Only (Gentle-Aug Checkpoint) ===")
ssl_only_ref = {1: 0.3276, 5: 0.4099, 10: 0.4302, 20: 0.4663}
riemannian_full_data_ref = 0.7232
for k in K_VALUES:
    means = [all_results[s][k]["mean_accuracy"] for s in holdout_subjects]
    agg = float(np.mean(means))
    delta = agg - ssl_only_ref[k]
    print(f"  k={k:2d}: fusion={agg:.4f}  ssl-only={ssl_only_ref[k]:.4f}  delta={delta:+.4f}")

out_path = Path("results/phase5_fewshot_fusion.json")
with open(out_path, "w") as f:
    json.dump({
        "holdout_subjects": holdout_subjects, "k_values": K_VALUES, "n_draws": N_DRAWS,
        "per_subject": {str(s): {str(k): v for k, v in sr.items()} for s, sr in all_results.items()},
        "ssl_only_reference": ssl_only_ref,
        "riemannian_full_data_reference": riemannian_full_data_ref,
    }, f, indent=2)
print(f"\nSaved -> {out_path}")