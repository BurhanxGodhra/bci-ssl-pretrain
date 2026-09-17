from scripts.pretrain import train
from src.augmentations.eeg_augment import build_gentle_pipeline

history = train(
    dataset_key="physionet_mi",
    subjects=None,
    batch_size=64,
    epochs=25,
    lr=3e-4,
    use_cosine_schedule=False,
    pipeline_builder=build_gentle_pipeline,
    seed=42,
)
print(f"Final loss (last 5): {[round(h,4) for h in history[-5:]]}")
