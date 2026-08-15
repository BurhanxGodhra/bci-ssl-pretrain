"""
Stratified k-shot sampling: draws exactly k trials per class from a
subject's data as a "support set" for few-shot adaptation, with the
remainder as the "query set" for evaluation.
"""
import numpy as np


def sample_k_shot_split(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int,
) -> tuple:
    """
    X: (n_trials, n_channels, n_timepoints) for ONE subject
    y: (n_trials,) integer labels
    k: number of support trials PER CLASS

    Returns: X_support, y_support, X_query, y_query
    """
    rng = np.random.default_rng(seed)
    support_idx = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < k:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} trials, "
                f"cannot sample k={k}."
            )
        chosen = rng.choice(cls_idx, size=k, replace=False)
        support_idx.extend(chosen.tolist())

    support_idx = np.array(support_idx)
    query_idx = np.setdiff1d(np.arange(len(y)), support_idx)

    return X[support_idx], y[support_idx], X[query_idx], y[query_idx]