"""
Fusion approach: concatenates frozen SSL embeddings with Riemannian
tangent-space features, evaluated via k-shot linear classification.
Tests whether the two feature types are complementary.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.finetune.linear_probe_eval import embed_all
from src.finetune.fewshot_sampler import sample_k_shot_split


def compute_tangent_features(X_support: np.ndarray, X_query: np.ndarray):
    """Fits covariance + tangent-space projection on SUPPORT ONLY,
    transforms both -- prevents leakage from query set into the
    Riemannian mean reference point."""
    cov = Covariances(estimator="lwf")
    ts = TangentSpace(metric="riemann")

    cov_sup = cov.fit_transform(X_support)
    ts_sup = ts.fit_transform(cov_sup)

    cov_qry = cov.transform(X_query)
    ts_qry = ts.transform(cov_qry)

    return ts_sup, ts_qry


def kshot_fusion_eval(
    encoder,
    X_subject: np.ndarray,
    y_subject: np.ndarray,
    k: int,
    n_draws: int,
    device,
    C: float = 0.1,   # stronger regularization than default -- high-dim, few samples
    base_seed: int = 42,
) -> dict:
    accuracies = []

    for draw in range(n_draws):
        seed = base_seed + draw
        X_sup, y_sup, X_qry, y_qry = sample_k_shot_split(X_subject, y_subject, k=k, seed=seed)

        # SSL embeddings (frozen encoder)
        emb_sup = embed_all(encoder, X_sup, device).numpy()
        emb_qry = embed_all(encoder, X_qry, device).numpy()

        # Riemannian tangent-space features (fit on support only)
        ts_sup, ts_qry = compute_tangent_features(X_sup, X_qry)

        # Concatenate, then standardize using support-set statistics only
        combined_sup = np.concatenate([emb_sup, ts_sup], axis=1)
        combined_qry = np.concatenate([emb_qry, ts_qry], axis=1)

        scaler = StandardScaler()
        combined_sup_scaled = scaler.fit_transform(combined_sup)
        combined_qry_scaled = scaler.transform(combined_qry)

        clf = LogisticRegression(max_iter=1000, C=C)
        clf.fit(combined_sup_scaled, y_sup)
        acc = clf.score(combined_qry_scaled, y_qry)
        accuracies.append(acc)

    return {
        "k": k, "n_draws": n_draws, "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }