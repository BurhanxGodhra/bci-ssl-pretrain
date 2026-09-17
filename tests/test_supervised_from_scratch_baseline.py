import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.supervised_baseline import kshot_supervised_from_scratch_eval
from src.utils.device import get_device

device = get_device()
split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs_data = load_dataset("bnci2014_001", subjects=holdout_subjects)

K_VALUES = [1, 5, 10, 20]
N_DRAWS = 10
aggregate = {}

for k in K_VALUES:
    means = []
    for subj in holdout_subjects:
        mask = epochs_data.subject_ids == subj
        X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]
        r = kshot_supervised_from_scratch_eval(
            n_channels=22, n_timepoints=1001, embed_dim=128,
            X_subject=X_subj, y_subject=y_subj, sfreq=epochs_data.sfreq,
            k=k, n_draws=N_DRAWS, device=device,
        )
        means.append(r["mean_accuracy"])
    agg = float(np.mean(means))
    aggregate[k] = agg
    print(f"k={k:2d}: random-init-full-finetune={agg:.4f}")

ssl_init_finetune_ref = {1: 0.3429, 5: 0.3864, 10: 0.3960, 20: 0.4037}
print("\n=== Random-Init vs SSL-Init (both full fine-tune) ===")
for k in K_VALUES:
    delta = ssl_init_finetune_ref[k] - aggregate[k]
    print(f"  k={k:2d}: ssl-init={ssl_init_finetune_ref[k]:.4f} random-init={aggregate[k]:.4f} delta={delta:+.4f}")

out_path = Path("results/phase_audit_supervised_from_scratch.json")
with open(out_path, "w") as f:
    json.dump({"k_values": K_VALUES, "aggregate": aggregate, "ssl_init_reference": ssl_init_finetune_ref}, f, indent=2)
print(f"\nSaved -> {out_path}")