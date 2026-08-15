"""
Full fine-tuning: unfreezes the pretrained encoder and updates it jointly
with a classifier head on the k-shot support set. Compared against
linear_probe_eval.py's frozen-encoder results using identical splits.
"""
import numpy as np
import torch
import torch.nn as nn

from src.models.encoder import EEGNetEncoder
from src.models.classifier_head import LinearProbe
from src.augmentations.eeg_augment import build_default_pipeline
from src.finetune.fewshot_sampler import sample_k_shot_split


def load_encoder_for_finetune(checkpoint_path: str, device) -> EEGNetEncoder:
    """Loads a fresh, UNFROZEN copy of the pretrained encoder. Must be
    called fresh per fine-tuning run -- never reuse a fine-tuned instance
    across draws, or later draws leak information from earlier ones."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = EEGNetEncoder(
        n_channels=ckpt["n_channels"],
        n_timepoints=ckpt["n_timepoints"],
        embed_dim=ckpt["embed_dim"],
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(device)
    return encoder


def finetune_on_support(
    checkpoint_path: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    n_classes: int,
    sfreq: float,
    device,
    epochs: int = 30,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    encoder = load_encoder_for_finetune(checkpoint_path, device)
    head = LinearProbe(embed_dim=encoder.embed_proj.out_features, n_classes=n_classes).to(device)

    optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": lr_encoder},
        {"params": head.parameters(), "lr": lr_head},
    ], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Fresh augmentation stream reused each epoch -- regularizes against
    # memorizing the tiny literal support set.
    aug_pipeline = build_default_pipeline(sfreq=sfreq, seed=seed)
    y_t = torch.from_numpy(y_support).long().to(device)

    encoder.train()
    head.train()
    for _ in range(epochs):
        X_aug = np.stack([aug_pipeline(x) for x in X_support])
        X_t = torch.from_numpy(X_aug).float().to(device)

        optimizer.zero_grad()
        logits = head(encoder(X_t))
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    encoder.eval()
    head.eval()
    return encoder, head


@torch.no_grad()
def evaluate_finetuned(encoder, head, X_query: np.ndarray, y_query: np.ndarray, device, batch_size: int = 64) -> float:
    encoder.eval()
    head.eval()
    correct = 0
    for i in range(0, len(X_query), batch_size):
        batch = torch.from_numpy(X_query[i:i + batch_size]).float().to(device)
        preds = head(encoder(batch)).argmax(dim=1).cpu().numpy()
        correct += int((preds == y_query[i:i + batch_size]).sum())
    return correct / len(y_query)


def kshot_full_finetune_eval(
    checkpoint_path: str,
    X_subject: np.ndarray,
    y_subject: np.ndarray,
    sfreq: float,
    k: int,
    n_draws: int,
    device,
    epochs: int = 30,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-3,
    base_seed: int = 42,
) -> dict:
    """Same base_seed + draw scheme as kshot_linear_probe_eval -- guarantees
    identical support/query splits for paired, apples-to-apples comparison."""
    n_classes = len(np.unique(y_subject))
    accuracies = []

    for draw in range(n_draws):
        seed = base_seed + draw
        X_sup, y_sup, X_qry, y_qry = sample_k_shot_split(X_subject, y_subject, k=k, seed=seed)

        encoder, head = finetune_on_support(
            checkpoint_path, X_sup, y_sup, n_classes, sfreq, device,
            epochs=epochs, lr_encoder=lr_encoder, lr_head=lr_head, seed=seed,
        )
        acc = evaluate_finetuned(encoder, head, X_qry, y_qry, device)
        accuracies.append(acc)

    return {
        "k": k, "n_draws": n_draws, "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }