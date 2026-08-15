"""
Critical control: does a randomly-initialized, NEVER-TRAINED encoder
achieve similar linear-probe accuracy to our pretrained one? If yes,
pretraining isn't earning its keep. If no, pretraining is doing real work.
"""
import torch
from src.models.encoder import EEGNetEncoder
from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import kshot_linear_probe_eval
from src.utils.device import get_device

device = get_device()

# Build a fresh, UNTRAINED encoder -- same architecture, random init weights
torch.manual_seed(42)
random_encoder = EEGNetEncoder(n_channels=22, n_timepoints=1001, embed_dim=128).to(device)
random_encoder.eval()
for p in random_encoder.parameters():
    p.requires_grad = False
print("Built random (untrained) encoder -- same architecture as pretrained checkpoint.")

split = load_split("bnci2014_001")
holdout_subjects = split["holdout_subjects"]
epochs_data = load_dataset("bnci2014_001", subjects=holdout_subjects)

K_VALUES = [1, 5, 10, 20]
N_DRAWS = 10

print("\n=== Random Encoder Control (linear probe on top of UNTRAINED encoder) ===")
aggregate = {}
for k in K_VALUES:
    accs = []
    for subj in holdout_subjects:
        mask = epochs_data.subject_ids == subj
        X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]
        result = kshot_linear_probe_eval(random_encoder, X_subj, y_subj, k=k, n_draws=N_DRAWS, device=device)
        accs.append(result["mean_accuracy"])
    agg = sum(accs) / len(accs)
    aggregate[k] = agg
    print(f"  k={k:2d}: random-encoder={agg:.4f}")

print("\n=== Comparison: Pretrained vs Random Encoder ===")
pretrained_ref = {1: 0.3261, 5: 0.4013, 10: 0.4364, 20: 0.4689}
for k in K_VALUES:
    delta = pretrained_ref[k] - aggregate[k]
    print(f"  k={k:2d}: pretrained={pretrained_ref[k]:.4f}  random={aggregate[k]:.4f}  "
          f"delta={delta:+.4f}  {'[pretraining helps]' if delta > 0.02 else '[MARGINAL/NO BENEFIT]'}")