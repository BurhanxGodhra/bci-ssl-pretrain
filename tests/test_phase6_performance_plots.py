from pathlib import Path
from src.visualization.performance_curves import plot_fewshot_comparison
from src.visualization.subject_heatmap import plot_subject_transfer_heatmap

Path("results").mkdir(exist_ok=True)

# From the random-encoder control run (paste your actual printed values if they differ)
random_control = {1: 0.3200, 5: 0.3679, 10: 0.3751, 20: 0.3911}

plot_fewshot_comparison(
    linear_probe_json="results/phase5_fewshot_linear_probe_bnci2014_001_v2_full25.json",
    finetune_json="results/phase5_fewshot_full_finetune_bnci2014_001.json",
    random_control=random_control,
    riemannian_accuracy=0.7232,
    chance_level=0.25,
    out_path="results/phase6_fewshot_comparison_bnci2014_001.png",
)

plot_subject_transfer_heatmap(
    result_jsons={
        "BNCI2014_001": "results/phase5_fewshot_linear_probe_bnci2014_001_v2_full25.json",
    },
    out_path="results/phase6_subject_transfer_heatmap.png",
)