# tests/test_phase3_train_multi_full.py
from scripts.pretrain import train_multi

history = train_multi(
    dataset_keys=["bnci2014_001", "physionet_mi"],
    subjects_per_dataset={
        "bnci2014_001": None,   # loads all 9; leakage filter keeps only the 7 pretrain subjects
        "physionet_mi": None,   # loads all 10; leakage filter keeps only the 8 pretrain subjects
    },
    batch_size=128,   # larger batch given ~5x more data
    epochs=25,
)
print(f"\nFinal loss trajectory (last 5 epochs): {[round(h,4) for h in history[-5:]]}")