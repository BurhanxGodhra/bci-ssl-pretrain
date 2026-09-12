# tests/test_fusion_full_sweep.py
import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.finetune.fusion_eval import kshot_fusion_eval
from src.utils.device import get_device

device = get_device()
CHECKPOINT = "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt"
encoder = load_pretrained_encoder(CHECKPOINT, device)

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
        result = kshot_fusion_eval(encoder, X_subj, y_subj, k=k, n_draws=N_DRAWS, device=device, C=0.1)
        means.append(result["mean_accuracy"])
    agg = float(np.mean(means))
    aggregate[k] = agg
    print(f"k={k:2d}: fusion={agg:.4f}")

out_path = Path("results/phase5_fewshot_fusion_full_sweep.json")
with open(out_path, "w") as f:
    json.dump({"k_values": K_VALUES, "aggregate": aggregate}, f, indent=2)
print(f"\nSaved -> {out_path}")