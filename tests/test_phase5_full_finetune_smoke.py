from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.full_finetune import kshot_full_finetune_eval
from src.utils.device import get_device

device = get_device()
CHECKPOINT = "checkpoints/encoder_multi_full_e25.pt"

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs_data = load_dataset("bnci2014_001", subjects=holdout_subjects)

subj = holdout_subjects[0]  # subject 1
mask = epochs_data.subject_ids == subj
X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]

print(f"Subject {subj}, k=5, full fine-tuning (5 draws, 30 epochs each)...")
result = kshot_full_finetune_eval(
    CHECKPOINT, X_subj, y_subj, sfreq=epochs_data.sfreq,
    k=5, n_draws=5, device=device,
)
print(f"\nFull fine-tune  k=5: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
print(f"Per-draw: {[round(a,3) for a in result['accuracies']]}")

# Reference: linear probe on the SAME subject/k/draws from the post-fix sweep was 0.4932
print(f"\nLinear probe   k=5 (same subject, from prior sweep): 0.4932")