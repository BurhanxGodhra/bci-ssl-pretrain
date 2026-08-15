import json
from pathlib import Path
from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.baselines.riemannian import evaluate_holdout_subjects

split = load_split("physionet_mi")
holdout_subjects = split["holdout_subjects"]
print(f"Holdout subjects (never seen during pretraining): {holdout_subjects}")

epochs = load_dataset("physionet_mi", subjects=holdout_subjects)
print(f"Loaded holdout data: X={epochs.X.shape}, classes={epochs.label_map}")

print("\nRunning per-subject Riemannian baseline (5-fold CV, full labeled data)...")
results = evaluate_holdout_subjects(epochs, holdout_subjects, n_splits=5)

n_classes = len(epochs.label_map)
chance = 1.0 / n_classes
print(f"\nOverall mean accuracy: {results['overall_mean_accuracy']:.4f} "
      f"(+/- {results['overall_std_accuracy']:.4f})")
print(f"Chance level ({n_classes}-class): {chance:.4f}")

assert results["overall_mean_accuracy"] > chance, \
    "Baseline barely above chance with full labeled data -- investigate pipeline."
print("[PASS] Baseline meaningfully exceeds chance level.")

out_path = Path("results/phase4_riemannian_baseline_physionet_mi.json")
with open(out_path, "w") as f:
    json.dump({
        "dataset": "physionet_mi",
        "holdout_subjects": holdout_subjects,
        "n_classes": n_classes,
        "chance_level": chance,
        "per_subject": {str(k): v for k, v in results["per_subject"].items()},
        "overall_mean_accuracy": results["overall_mean_accuracy"],
        "overall_std_accuracy": results["overall_std_accuracy"],
    }, f, indent=2)
print(f"\nSaved -> {out_path}")