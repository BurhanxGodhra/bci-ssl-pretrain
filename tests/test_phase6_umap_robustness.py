# tests/test_phase6_umap_robustness.py
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.visualization.embedding_diagnostics import gather_pretrain_embeddings
from src.utils.device import get_device
import umap
import matplotlib.pyplot as plt
import seaborn as sns

device = get_device()
encoder = load_pretrained_encoder("checkpoints/encoder_multi_full_e25.pt", device)
embeddings, classes, subjects, datasets = gather_pretrain_embeddings(
    encoder, ["bnci2014_001", "physionet_mi"], device
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, n_neighbors in zip(axes, [5, 15, 50]):
    proj = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(embeddings)
    sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=subjects, s=8, alpha=0.6, ax=ax, legend=False)
    ax.set_title(f"n_neighbors={n_neighbors}")
plt.tight_layout()
plt.savefig("results/phase6_umap_robustness_check.png", dpi=120)
print("Saved -> results/phase6_umap_robustness_check.png")