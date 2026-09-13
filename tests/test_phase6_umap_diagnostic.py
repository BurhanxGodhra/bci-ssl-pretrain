from pathlib import Path
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.visualization.embedding_diagnostics import (
    gather_pretrain_embeddings, compute_separation_scores, plot_umap_diagnostic,
)
from src.utils.device import get_device

device = get_device()
encoder = load_pretrained_encoder("checkpoints/encoder_multi_full_e25.pt", device)

print("Embedding all pretrain-pool trials...")
embeddings, classes, subjects, datasets = gather_pretrain_embeddings(
    encoder, ["bnci2014_001", "physionet_mi"], device
)
print(f"\nTotal embeddings: {embeddings.shape}")

print("\nComputing silhouette scores (higher = encoder organizes more strongly around this label)...")
scores = compute_separation_scores(embeddings, classes, subjects, datasets)
for k, v in scores.items():
    print(f"  {k:12s}: {v:+.4f}")

print("\nGenerating UMAP visualization...")
out_path = Path("results/phase6_umap_diagnostic.png")
out_path.parent.mkdir(exist_ok=True)
plot_umap_diagnostic(embeddings, classes, subjects, datasets, out_path)