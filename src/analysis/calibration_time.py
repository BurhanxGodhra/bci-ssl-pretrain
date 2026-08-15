"""
Converts k-shot accuracy results into real-world calibration time framing.
All time constants are stated explicitly and are adjustable -- this is
meant to be transparent, not a black-box marketing number.
"""
import json

# BNCI2014_001 protocol constants (from Phase 2 epoching params + standard
# MI paradigm timing -- cue + imagery + rest per trial)
IMAGERY_DURATION_SEC = 4.0      # tmax - tmin, our actual epoch window
CUE_OVERHEAD_SEC = 2.0          # visual/audio cue before imagery starts
REST_OVERHEAD_SEC = 2.0         # inter-trial rest, standard MI protocol practice
SECONDS_PER_TRIAL = IMAGERY_DURATION_SEC + CUE_OVERHEAD_SEC + REST_OVERHEAD_SEC  # 8.0s
N_CLASSES = 4


def k_to_time(k: int, n_classes: int = N_CLASSES) -> dict:
    total_trials = k * n_classes
    total_seconds = total_trials * SECONDS_PER_TRIAL
    return {
        "k": k,
        "total_trials": total_trials,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
    }


def build_calibration_time_table(linear_probe_json: str, riemannian_accuracy: float, full_data_trials: int):
    with open(linear_probe_json) as f:
        lp = json.load(f)

    print(f"{'k':>4} {'trials':>8} {'time':>10} {'accuracy':>10} {'% of ceiling':>14}")
    print("-" * 52)

    rows = []
    for k in lp["k_values"]:
        t = k_to_time(k)
        acc = lp["aggregate"][str(k)]["mean"]
        pct_of_ceiling = acc / riemannian_accuracy
        rows.append({**t, "accuracy": acc, "pct_of_riemannian_ceiling": pct_of_ceiling})
        mins = t["total_minutes"]
        print(f"{k:>4} {t['total_trials']:>8} {mins:>8.1f}min {acc:>9.1%} {pct_of_ceiling:>13.1%}")

    full_time = k_to_time(full_data_trials // N_CLASSES)
    print(f"\nFull calibration (Riemannian baseline): {full_data_trials} trials, "
          f"~{full_time['total_minutes']:.1f} min, {riemannian_accuracy:.1%} accuracy")

    # Headline trade-off point: smallest k reaching >=80% of ceiling accuracy
    best_80pct = next((r for r in rows if r["pct_of_riemannian_ceiling"] >= 0.80), None)
    if best_80pct:
        savings_pct = 1 - (best_80pct["total_minutes"] / full_time["total_minutes"])
        print(f"\nHeadline: k={best_80pct['k']} reaches {best_80pct['pct_of_riemannian_ceiling']:.0%} "
              f"of full-calibration accuracy using {best_80pct['total_minutes']:.1f} min "
              f"({savings_pct:.0%} less calibration time)")
    else:
        print(f"\nNo tested k reached 80% of Riemannian ceiling -- "
              f"honest gap remains at largest k tested.")

    return rows, full_time