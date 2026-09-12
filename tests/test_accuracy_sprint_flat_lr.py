# tests/test_accuracy_sprint_flat_lr.py
from scripts.pretrain import train

history_flat = train(
    dataset_key="bnci2014_001",
    subjects=None,
    batch_size=64,
    epochs=25,
    lr=3e-4,
    use_cosine_schedule=False,
    seed=42,
)

# Reference from the cosine-schedule run we already have
cosine_history = [4.7829, 4.2020, 3.7203, 3.5851, 3.5135, 3.4509, 3.4172, 3.3928,
                   3.3668, 3.3435, 3.3419, 3.3349, 3.3232, 3.3162, 3.3120, 3.3058,
                   3.3017, 3.2994, 3.2979, 3.2963, 3.2938, 3.2952, 3.2939, 3.2922, 3.2978]

print(f"\n{'Epoch':>6} {'Flat LR':>10} {'Cosine':>10} {'Delta':>10}")
for i, (flat, cos) in enumerate(zip(history_flat, cosine_history), 1):
    marker = " <-- flat still improving" if i > 10 and flat < cos - 0.02 else ""
    print(f"{i:>6} {flat:>10.4f} {cos:>10.4f} {flat-cos:>+10.4f}{marker}")