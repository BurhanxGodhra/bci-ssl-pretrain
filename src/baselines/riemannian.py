"""
Classical Riemannian geometry baseline: Covariance estimation ->
Tangent Space projection -> Logistic Regression.
Standard, rigorous BCI benchmark used to contextualize SSL results.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def build_riemannian_pipeline(estimator: str = "lwf", C: float = 1.0) -> Pipeline:
    """
    estimator: covariance estimator type. 'lwf' = Ledoit-Wolf shrinkage,
               which regularizes the covariance estimate -- important
               for EEG since trials often have few timepoints relative
               to channels, making raw sample covariance noisy/singular.
    """
    return Pipeline([
        ("covariances", Covariances(estimator=estimator)),
        ("tangent_space", TangentSpace(metric="riemann")),
        ("classifier", LogisticRegression(max_iter=1000, C=C)),
    ])


def evaluate_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Standard within-subject stratified k-fold cross-validation --
    the conventional way Riemannian baselines are reported in BCI
    literature (e.g. MOABB benchmark tables).

    X: (n_trials, n_channels, n_timepoints) for ONE subject
    y: (n_trials,) integer labels
    """
    pipeline = build_riemannian_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "fold_scores": scores.tolist(),
        "n_trials": len(y),
        "n_classes": len(set(y)),
    }


def evaluate_holdout_subjects(
    epochs,
    holdout_subjects: list,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Runs evaluate_subject_cv independently per holdout subject, then
    aggregates -- gives both per-subject numbers (useful for the
    subject-transfer heatmap in Phase 6) and an overall mean.
    """
    results = {}
    for subj in holdout_subjects:
        mask = epochs.subject_ids == subj
        X_subj = epochs.X[mask]
        y_subj = epochs.y[mask]

        if len(X_subj) == 0:
            print(f"[WARNING] No trials found for subject {subj}, skipping.")
            continue

        result = evaluate_subject_cv(X_subj, y_subj, n_splits=n_splits, seed=seed)
        results[subj] = result
        print(
            f"  Subject {subj}: {result['mean_accuracy']:.4f} "
            f"(+/- {result['std_accuracy']:.4f}), n={result['n_trials']}"
        )

    overall_mean = float(np.mean([r["mean_accuracy"] for r in results.values()]))
    overall_std = float(np.std([r["mean_accuracy"] for r in results.values()]))

    return {
        "per_subject": results,
        "overall_mean_accuracy": overall_mean,
        "overall_std_accuracy": overall_std,
    }