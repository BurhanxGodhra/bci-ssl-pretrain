"""
Diagnostic: what did the encoder actually organize its embedding space
around -- MI class (good), subject identity, or dataset-of-origin (bad,
means pretraining learned nuisance shortcuts instead of task signal)?
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import umap

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.data.channel_utils import align_epochs
from src.finetune.linear_probe_eval import embed_all


def gather_pretrain_embeddings(encoder, dataset_keys, device):
    """Embeds all PRETRAIN-POOL trials (holdout excluded) across datasets,
    tagged with class/subject/dataset labels for diagnostic coloring."""
    epochs_list = [load_dataset(dk) for dk in dataset_keys]
    aligned = align_epochs(epochs_list)

    all_emb, all_class, all_subject, all_dataset = [], [], [], []

    for e in aligned:
        split = load_split(e.dataset_name)
        pretrain_subjects = set(split["pretrain_subjects"])
        mask = np.isin(e.subject_ids, list(pretrain_subjects))

        X, y, subj = e.X[mask], e.y[mask], e.subject_ids[mask]
        class_names = np.array([e.label_map[i] for i in y])

        emb = embed_all(encoder, X, device).numpy()
        all_emb.append(emb)
        all_class.append(class_names)
        all_subject.append(subj.astype(str))
        all_dataset.append(np.array([e.dataset_name] * len(y)))

        print(f"  {e.dataset_name}: embedded {len(y)} pretrain-pool trials")

    return (
        np.concatenate(all_emb, axis=0),
        np.concatenate(all_class),
        np.concatenate(all_subject),
        np.concatenate(all_dataset),
    )


def compute_separation_scores(embeddings, classes, subjects, datasets) -> dict:
    """Silhouette score per grouping -- higher = encoder organizes more
    strongly around that label. Computed in the ORIGINAL embedding space,
    not the 2D UMAP projection."""
    return {
        "by_class": float(silhouette_score(embeddings, classes)),
        "by_subject": float(silhouette_score(embeddings, subjects)),
        "by_dataset": float(silhouette_score(embeddings, datasets)),
    }


def plot_umap_diagnostic(embeddings, classes, subjects, datasets, out_path, seed=42):
    reducer = umap.UMAP(n_components=2, random_state=seed)
    proj = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, labels, title in zip(
        axes, [classes, subjects, datasets],
        ["Colored by MI Class", "Colored by Subject", "Colored by Source Dataset"],
    ):
        sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, s=8, alpha=0.6, ax=ax, legend="brief")
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(loc="upper right", fontsize=7, markerscale=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved -> {out_path}")