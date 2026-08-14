# tests/test_phase2_pipeline.py
import numpy as np
import matplotlib.pyplot as plt

from src.data.loaders import load_dataset
from src.data.splits import create_split, load_split
from src.augmentations.eeg_augment import build_default_pipeline

# 1. Load a small MOABB dataset (start with BNCI2014_001, 9 subjects, fast download)
epochs = load_dataset("bnci2014_001", subjects=[1, 2, 3])
print(f"\nX shape: {epochs.X.shape}, y shape: {epochs.y.shape}")
print(f"Classes: {epochs.label_map}")
print(f"Subjects present: {sorted(set(epochs.subject_ids))}")

# 2. Create + reload a subject split (leakage firewall check)
create_split("bnci2014_001", all_subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9], n_holdout=2, seed=42)
split = load_split("bnci2014_001")
assert set(split["pretrain_subjects"]).isdisjoint(set(split["holdout_subjects"])), \
    "LEAKAGE: pretrain and holdout subjects overlap!"
print("\n[PASS] No subject overlap between pretrain pool and holdout.")

# 3. Generate two augmented views of one trial and plot for visual sanity check
aug_pipeline = build_default_pipeline(sfreq=epochs.sfreq, seed=123)
trial = epochs.X[0]  # (n_channels, n_timepoints)
view1 = aug_pipeline(trial)
view2 = aug_pipeline(trial)

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
ch = 0  # plot first channel
axes[0].plot(trial[ch]); axes[0].set_title("Original (channel 0)")
axes[1].plot(view1[ch]); axes[1].set_title("Augmented View 1")
axes[2].plot(view2[ch]); axes[2].set_title("Augmented View 2")
plt.tight_layout()
plt.savefig("results/phase2_augmentation_check.png", dpi=100)
print("\n[SAVED] results/phase2_augmentation_check.png")
print("Views should look plausibly similar to original but visibly perturbed.")
