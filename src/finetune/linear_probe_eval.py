"""
Loads a pretrained encoder checkpoint, freezes it, and evaluates
k-shot linear probe performance on a holdout subject.
"""
import numpy as np
import torch
import torch.nn as nn

from src.models.encoder import EEGNetEncoder
from src.models.classifier_head import LinearProbe
from src.finetune.fewshot_sampler import sample_k_shot_split
from src.utils.device import get_device


def load_pretrained_encoder(checkpoint_path: str, device) -> EEGNetEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = EEGNetEncoder(
        n_channels=ckpt["n_channels"],
        n_timepoints=ckpt["n_timepoints"],
        embed_dim=ckpt["embed_dim"],
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False  # freeze -- linear probing only
    return encoder


@torch.no_grad()
def embed_all(encoder: EEGNetEncoder, X: np.ndarray, device, batch_size: int = 64) -> torch.Tensor:
    """Compute frozen encoder embeddings for a full array of trials."""
    embeddings = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        embeddings.append(encoder(batch).cpu())
    return torch.cat(embeddings, dim=0)


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    n_classes: int,
    device,
    epochs: int = 100,
    lr: float = 1e-2,
) -> LinearProbe:
    probe = LinearProbe(embed_dim=train_embeddings.shape[1], n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    X = train_embeddings.to(device)
    y = torch.from_numpy(train_labels).long().to(device)

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    return probe


@torch.no_grad()
def evaluate_probe(probe: LinearProbe, embeddings: torch.Tensor, labels: np.ndarray, device) -> float:
    probe.eval()
    logits = probe(embeddings.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()
    return float((preds == labels).mean())


def kshot_linear_probe_eval(
    encoder: EEGNetEncoder,
    X_subject: np.ndarray,
    y_subject: np.ndarray,
    k: int,
    n_draws: int,
    device,
    base_seed: int = 42,
) -> dict:
    """
    Runs n_draws independent k-shot trials for one subject, one k value.
    Returns per-draw accuracies plus mean/std.
    """
    n_classes = len(np.unique(y_subject))
    accuracies = []

    for draw in range(n_draws):
        seed = base_seed + draw
        X_sup, y_sup, X_qry, y_qry = sample_k_shot_split(X_subject, y_subject, k=k, seed=seed)

        emb_sup = embed_all(encoder, X_sup, device)
        emb_qry = embed_all(encoder, X_qry, device)

        probe = train_linear_probe(emb_sup, y_sup, n_classes=n_classes, device=device)
        acc = evaluate_probe(probe, emb_qry, y_qry, device)
        accuracies.append(acc)

    return {
        "k": k,
        "n_draws": n_draws,
        "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }