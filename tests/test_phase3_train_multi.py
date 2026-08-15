# tests/test_phase3_train_multi.py
from scripts.pretrain import train_multi

history = train_multi(
    dataset_keys=["bnci2014_001", "physionet_mi"],
    subjects_per_dataset={
        "bnci2014_001": [2, 3, 5],
        "physionet_mi": [2, 3, 5],
    },
    batch_size=64,
    epochs=8,
)
print(f"\nLoss trajectory: {[round(h,4) for h in history]}")