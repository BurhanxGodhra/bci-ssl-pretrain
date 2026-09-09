# tests/test_accuracy_sprint_bnci_only.py
from scripts.pretrain import train

history = train(
    dataset_key="bnci2014_001",
    subjects=None,       # all 9; leakage filter keeps only the 7 pretrain-pool subjects
    batch_size=64,        # matches our original (better-performing) small-scale run, not 128
    epochs=25,
    lr=3e-4,
)
print(f"\nFinal loss trajectory (last 5): {[round(h,4) for h in history[-5:]]}")