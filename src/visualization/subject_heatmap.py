"""
Subject x k-shot accuracy heatmap -- makes per-subject difficulty
differences visible rather than buried in aggregate means.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_subject_transfer_heatmap(result_jsons: dict, out_path: str):
    """
    result_jsons: {"dataset_label": path_to_linear_probe_sweep_json, ...}
    Builds one combined heatmap across all datasets/subjects provided.
    """
    rows = []
    row_labels = []

    for dataset_label, path in result_jsons.items():
        with open(path) as f:
            data = json.load(f)
        k_values = data["k_values"]
        for subj, subj_results in data["per_subject"].items():
            row = [subj_results[str(k)]["mean_accuracy"] for k in k_values]
            rows.append(row)
            row_labels.append(f"{dataset_label} - Subj {subj}")

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(row_labels) + 2))
    sns.heatmap(
        matrix, annot=True, fmt=".2%", cmap="RdYlGn",
        xticklabels=[f"k={k}" for k in k_values],
        yticklabels=row_labels,
        cbar_kws={"label": "Linear Probe Accuracy"},
        vmin=0.2, vmax=0.8, ax=ax,
    )
    ax.set_title("Subject-Transfer Heatmap: Few-Shot Linear Probe Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved -> {out_path}")