"""
Isolates the value of SSL pretraining as weight initialization: same
architecture, same subjects, same fine-tuning procedure as
kshot_full_finetune_eval, but starting from RANDOM weights instead of
an SSL-pretrained checkpoint.
"""
import numpy as np
import torch
import torch.nn as nn

from src.augmentations.eeg_augment import build_default_pipeline
from src.finetune.fewshot_sampler import sample_k_shot_split
from src.finetune.full_finetune import evaluate_finetuned
from src.models.classifier_head import LinearProbe
from src.models.encoder import EEGNetEncoder


def kshot_supervised_from_scratch_eval(
    n_channels, n_timepoints, embed_dim,
    X_subject, y_subject, sfreq, k, n_draws, device,
    epochs=30, lr_encoder=1e-4, lr_head=1e-3, base_seed=42,
):
    n_classes = len(np.unique(y_subject))
    accuracies = []

    for draw in range(n_draws):
        seed = base_seed + draw
        X_sup, y_sup, X_qry, y_qry = sample_k_shot_split(X_subject, y_subject, k=k, seed=seed)

        torch.manual_seed(seed)
        encoder = EEGNetEncoder(n_channels=n_channels, n_timepoints=n_timepoints, embed_dim=embed_dim).to(device)
        head = LinearProbe(embed_dim=embed_dim, n_classes=n_classes).to(device)

        optimizer = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": lr_encoder},
            {"params": head.parameters(), "lr": lr_head},
        ], weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        aug_pipeline = build_default_pipeline(sfreq=sfreq, seed=seed)
        y_t = torch.from_numpy(y_sup).long().to(device)

        encoder.train()
        head.train()
        for _ in range(epochs):
            X_aug = np.stack([aug_pipeline(x) for x in X_sup])
            X_t = torch.from_numpy(X_aug).float().to(device)
            optimizer.zero_grad()
            logits = head(encoder(X_t))
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

        acc = evaluate_finetuned(encoder, head, X_qry, y_qry, device)
        accuracies.append(acc)

    return {
        "k": k, "n_draws": n_draws, "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }