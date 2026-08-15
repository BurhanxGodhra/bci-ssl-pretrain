# tests/test_phase6_cosine_silhouette_check.py
from sklearn.metrics import silhouette_score
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.visualization.embedding_diagnostics import gather_pretrain_embeddings
from src.utils.device import get_device

device = get_device()
encoder = load_pretrained_encoder("checkpoints/encoder_multi_full_e25.pt", device)
embeddings, classes, subjects, datasets = gather_pretrain_embeddings(
    encoder, ["bnci2014_001", "physionet_mi"], device
)

print("\nSilhouette scores -- COSINE metric (matches training objective's geometry):")
for name, labels in [("by_class", classes), ("by_subject", subjects), ("by_dataset", datasets)]:
    score = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  {name:12s}: {score:+.4f}")

# Also report raw embedding norm variance -- if this is large, it directly
# supports the "Euclidean silhouette was misleading" hypothesis
import numpy as np
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nEmbedding norm stats: mean={norms.mean():.3f}, std={norms.std():.3f}, "
      f"cv={norms.std()/norms.mean():.3f}  (high cv = norm varies a lot -> Euclidean distance misleading)")