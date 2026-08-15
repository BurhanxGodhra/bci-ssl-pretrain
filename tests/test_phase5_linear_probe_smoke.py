from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import load_pretrained_encoder, kshot_linear_probe_eval
from src.utils.device import get_device

device = get_device()
print(f"Device: {device}")

encoder = load_pretrained_encoder(
    "checkpoints/encoder_multi_bnci_physionet.pt", device
)
print("Loaded frozen pretrained encoder.")

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs = load_dataset("bnci2014_001", subjects=holdout_subjects)

# Test on ONE holdout subject, k=5, before running the full sweep
subj = holdout_subjects[0]
mask = epochs.subject_ids == subj
X_subj, y_subj = epochs.X[mask], epochs.y[mask]
print(f"Subject {subj}: {len(y_subj)} trials, {len(set(y_subj))} classes")

result = kshot_linear_probe_eval(
    encoder, X_subj, y_subj, k=5, n_draws=5, device=device
)
print(f"\nk={result['k']} linear probe accuracy: "
      f"{result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
print(f"Per-draw accuracies: {[round(a,3) for a in result['accuracies']]}")

chance = 1.0 / len(set(y_subj))
print(f"\nChance level (4-class): {chance:.4f}")
assert result["mean_accuracy"] > chance, "Linear probe not beating chance -- check encoder/embedding pipeline."
print("[PASS] Linear probe beats chance level.")