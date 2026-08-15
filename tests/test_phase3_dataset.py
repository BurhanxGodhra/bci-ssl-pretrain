from torch.utils.data import DataLoader
from src.data.loaders import load_dataset
from src.data.contrastive_dataset import PretrainContrastiveDataset

# Reuse the same 3-subject load from Phase 2 -- subject 1 is holdout per our split,
# so we should see it explicitly filtered out here.
epochs = load_dataset("bnci2014_001", subjects=[1, 2, 3])

ds = PretrainContrastiveDataset(epochs, seed=42)
print(f"\nDataset length: {len(ds)}")
assert 1 not in ds.subject_ids, "LEAKAGE: holdout subject 1 present in pretrain dataset!"
print("[PASS] Holdout subject 1 correctly excluded.")

view1, view2 = ds[0]
print(f"View1 shape: {view1.shape}, View2 shape: {view2.shape}, dtype: {view1.dtype}")

loader = DataLoader(ds, batch_size=8, shuffle=True)
batch_v1, batch_v2 = next(iter(loader))
print(f"Batch shapes: v1={batch_v1.shape}, v2={batch_v2.shape}")
print("\n[PASS] DataLoader batching works end-to-end.")
