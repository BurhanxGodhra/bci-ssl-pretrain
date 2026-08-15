from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.baselines.riemannian import evaluate_holdout_subjects

# Evaluate on the SAME holdout subjects reserved for Phase 5 few-shot eval --
# this is the critical apples-to-apples constraint: baseline and SSL
# fine-tuning must be scored on identical unseen subjects.
split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
print(f"Holdout subjects (never seen during pretraining): {holdout_subjects}")

epochs = load_dataset("bnci2014_001", subjects=holdout_subjects)
print(f"Loaded holdout data: X={epochs.X.shape}, classes={epochs.label_map}")

print("\nRunning per-subject Riemannian baseline (5-fold CV, full labeled data)...")
results = evaluate_holdout_subjects(epochs, holdout_subjects, n_splits=5)

print(f"\nOverall mean accuracy across holdout subjects: "
      f"{results['overall_mean_accuracy']:.4f} (+/- {results['overall_std_accuracy']:.4f})")

# Sanity bound: 4-class MI, random chance = 0.25. Riemannian baseline with
# FULL labeled data should comfortably beat this -- if not, something's wrong
# upstream (bad epoching, label misalignment, etc.), not just "hard problem."
assert results["overall_mean_accuracy"] > 0.25, \
    "Baseline barely above chance with full labeled data -- investigate pipeline, not model."
print("[PASS] Baseline meaningfully exceeds chance level with full labeled data.")

import json
from pathlib import Path

results_out = Path(__file__).resolve().parents[1] / "results" / "phase4_riemannian_baseline_bnci2014_001.json"
results_out.parent.mkdir(exist_ok=True)
with open(results_out, "w") as f:
    json.dump({
        "dataset": "bnci2014_001",
        "holdout_subjects": holdout_subjects,
        "n_splits": 5,
        "per_subject": {str(k): v for k, v in results["per_subject"].items()},
        "overall_mean_accuracy": results["overall_mean_accuracy"],
        "overall_std_accuracy": results["overall_std_accuracy"],
    }, f, indent=2)
print(f"\nSaved -> {results_out}")