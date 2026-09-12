# tests/test_phase6_update_with_fusion.py
import json
from src.visualization.performance_curves import plot_fewshot_comparison
from src.analysis.calibration_time import build_calibration_time_table

with open("results/phase5_fewshot_fusion_full_sweep.json") as f:
    fusion = json.load(f)
fusion_dict = {int(k): v for k, v in fusion["aggregate"].items()}

random_control = {1: 0.3200, 5: 0.3679, 10: 0.3751, 20: 0.3911}

plot_fewshot_comparison(
    linear_probe_json="results/phase5_fewshot_linear_probe_bnci2014_001_v2_full25.json",
    finetune_json="results/phase5_fewshot_full_finetune_bnci2014_001.json",
    random_control=random_control,
    riemannian_accuracy=0.7232,
    chance_level=0.25,
    out_path="results/phase6_fewshot_comparison_bnci2014_001_v2_with_fusion.png",
    fusion_results=fusion_dict,
)

print("\n=== Calibration-time framing, updated with fusion numbers ===")
# Manually adapt: swap accuracy source to fusion for the headline table
for k, acc in fusion_dict.items():
    pct = acc / 0.7232
    print(f"k={k:2d}: {acc:.1%} accuracy = {pct:.1%} of Riemannian ceiling")