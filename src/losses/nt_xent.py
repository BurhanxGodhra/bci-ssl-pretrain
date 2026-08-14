"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss -- SimCLR's
contrastive objective. Operates on a batch of 2N projected embeddings
(N trials x 2 augmented views each).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (N, proj_dim) -- embeddings of view1 and view2 for the same
                 N trials, already index-aligned (z1[i] and z2[i] are a
                 positive pair).
        """
        N = z1.shape[0]
        device = z1.device

        z = torch.cat([z1, z2], dim=0)          # (2N, proj_dim)
        z = F.normalize(z, dim=1)               # cosine similarity prep

        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity (diagonal) -- a sample is never its own negative
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(self_mask, float("-inf"))

        # Positive pair indices: for row i in [0, N), positive is at i+N; and vice versa
        positive_idx = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])

        loss = F.cross_entropy(sim_matrix, positive_idx)
        return loss
