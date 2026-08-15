"""
Lightweight linear classification head for few-shot linear probing.
Sits on top of a FROZEN pretrained encoder.
"""
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)