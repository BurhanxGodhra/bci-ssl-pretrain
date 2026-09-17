"""
Subject-ID probe: tests whether frozen SSL embeddings carry exploitable
subject-identity information -- directly answers whether the representation
encodes "who" instead of "what class of motor imagery."
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


def run_subject_id_probe(embeddings: np.ndarray, subject_ids: np.ndarray, n_folds: int = 5) -> dict:
    n_subjects = len(np.unique(subject_ids))
    chance = 1.0 / n_subjects

    clf = LogisticRegression(max_iter=1000, C=0.1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, subject_ids, cv=cv, scoring="accuracy")

    return {
        "n_subjects": n_subjects,
        "chance_level": chance,
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "fold_scores": scores.tolist(),
    }