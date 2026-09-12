# tests/test_accuracy_sprint_gentle_aug.py
from scripts.pretrain import train
from src.augmentations.eeg_augment import build_gentle_pipeline

history = train(
    dataset_key="bnci2014_001",
    subjects=None,
    batch_size=64,
    epochs=25,
    lr=3e-4,
    use_cosine_schedule=False,  # keep the flat-LR win from last experiment
    pipeline_builder=build_gentle_pipeline,
    seed=42,
)
print(f"\nFinal loss (last 5): {[round(h,4) for h in history[-5:]]}")