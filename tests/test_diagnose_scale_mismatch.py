# tests/test_diagnose_scale_mismatch.py
from src.data.loaders import load_dataset
from src.data.channel_utils import align_epochs
import numpy as np

epochs_list = [load_dataset("bnci2014_001"), load_dataset("physionet_mi")]
aligned = align_epochs(epochs_list)

for e in aligned:
    print(f"{e.dataset_name}: mean={e.X.mean():.4f}, std={e.X.std():.4f}, "
          f"abs_max={np.abs(e.X).max():.4f}")