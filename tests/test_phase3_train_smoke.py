# tests/test_phase3_train_smoke.py
from scripts.pretrain import train

history = train(
    dataset_key="bnci2014_001",
    subjects=[2, 3],       # small, fast — pretrain-pool subjects only
    batch_size=32,
    epochs=3,
    lr=3e-4,
)

print(f"\nLoss trajectory: {[round(h, 4) for h in history]}")
assert history[-1] < history[0], "Loss did not decrease -- something's wrong"
print("[PASS] Loss decreased over 3 epochs.")
