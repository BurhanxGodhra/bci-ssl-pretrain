# tests/test_calibration_time_framing.py
from src.analysis.calibration_time import build_calibration_time_table

# Subject 1's full trial count (576) is what the Riemannian baseline's
# 5-fold CV effectively used as its labeled pool
rows, full_time = build_calibration_time_table(
    linear_probe_json="results/phase5_fewshot_linear_probe_bnci2014_001_v2_full25.json",
    riemannian_accuracy=0.7232,
    full_data_trials=576,
)