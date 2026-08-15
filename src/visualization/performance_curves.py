"""
Combined few-shot accuracy-vs-k curve: random baseline, pretrained linear
probe, pretrained full fine-tune, plus Riemannian ceiling and chance-level
reference lines. This is the primary evidence plot for the project.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_fewshot_comparison(
    linear_probe_json: str,
    finetune_json: str,
    random_control: dict,      # {k: accuracy} -- from the random-encoder control run
    riemannian_accuracy: float,
    chance_level: float,
    out_path: str,
    title: str = "Few-Shot MI Classification: Method Comparison (BNCI2014_001)",
):
    with open(linear_probe_json) as f:
        lp = json.load(f)
    with open(finetune_json) as f:
        ft = json.load(f)

    k_values = lp["k_values"]
    lp_means = [lp["aggregate"][str(k)]["mean"] for k in k_values]
    lp_stds = [lp["aggregate"][str(k)]["std"] for k in k_values]
    ft_means = [ft["aggregate"][str(k)]["mean"] for k in k_values]
    ft_stds = [ft["aggregate"][str(k)]["std"] for k in k_values]
    rand_means = [random_control[k] for k in k_values]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(k_values, lp_means, yerr=lp_stds, marker="o", capsize=4,
                label="Pretrained + Linear Probe", linewidth=2, color="#2563eb")
    ax.errorbar(k_values, ft_means, yerr=ft_stds, marker="s", capsize=4,
                label="Pretrained + Full Fine-Tune", linewidth=2, color="#dc2626")
    ax.plot(k_values, rand_means, marker="^", linestyle="--",
            label="Random (Untrained) Encoder + Linear Probe", color="#6b7280")

    ax.axhline(riemannian_accuracy, color="#16a34a", linestyle=":", linewidth=2,
               label=f"Riemannian Baseline (full data): {riemannian_accuracy:.1%}")
    ax.axhline(chance_level, color="#9ca3af", linestyle=":", linewidth=1,
               label=f"Chance Level: {chance_level:.1%}")

    ax.set_xlabel("k (labeled trials per class)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(k_values)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved -> {out_path}")