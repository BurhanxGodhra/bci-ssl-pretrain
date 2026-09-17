import json
from pathlib import Path
import numpy as np

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.finetune.fusion_eval import kshot_fusion_eval
from src.utils.device import get_device

device = get_device()
K_VALUES = [1, 5, 10, 20]
N_DRAWS = 10


def evaluate_transfer(checkpoint_path, dataset_key, label):
    encoder = load_pretrained_encoder(checkpoint_path, device)
    split = load_split(dataset_key)
    holdout_subjects = split["holdout_subjects"]
    epochs = load_dataset(dataset_key, subjects=holdout_subjects)

    print(f"\n=== {label} ===")
    aggregate = {}
    for k in K_VALUES:
        means = []
        for subj in holdout_subjects:
            mask = epochs.subject_ids == subj
            X_subj, y_subj = epochs.X[mask], epochs.y[mask]
            r = kshot_fusion_eval(encoder, X_subj, y_subj, k=k, n_draws=N_DRAWS, device=device)
            means.append(r["mean_accuracy"])
        agg = float(np.mean(means))
        aggregate[k] = agg
        print(f"  k={k:2d}: {agg:.4f}")
    return aggregate


bnci_to_physionet = evaluate_transfer(
    "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt",
    "physionet_mi",
    "Pretrain=BNCI2014_001 -> Eval=PhysionetMI (cross-dataset transfer)",
)

physionet_to_bnci = evaluate_transfer(
    "checkpoints/encoder_physionet_mi_only_e25_flat_gentleaug.pt",
    "bnci2014_001",
    "Pretrain=PhysionetMI -> Eval=BNCI2014_001 (cross-dataset transfer)",
)

within_dataset_ref = {
    "bnci2014_001_fusion_within": {1: 0.377, 5: 0.478, 10: 0.521, 20: 0.580},
    "physionet_mi_fusion_within": {1: 0.302, 5: 0.420, 10: 0.483, 20: 0.531},
}

out_path = Path("results/phase_audit_crossdataset_transfer.json")
with open(out_path, "w") as f:
    json.dump({
        "bnci_to_physionet": bnci_to_physionet,
        "physionet_to_bnci": physionet_to_bnci,
        "within_dataset_reference": within_dataset_ref,
    }, f, indent=2)
print(f"\nSaved -> {out_path}")