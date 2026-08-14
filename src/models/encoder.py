"""
EEGNet-style convolutional encoder for self-supervised pretraining.
Input:  (batch, 1, n_channels, n_timepoints)
Output: (batch, embed_dim)  -- the representation reused in Phase 5 fine-tuning
"""
import torch
import torch.nn as nn


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        embed_dim: int = 128,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Block 1: temporal conv -- learnable frequency-selective filters
        # applied independently per EEG channel (kernel spans time only)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 64), padding="same", bias=False),
            nn.BatchNorm2d(temporal_filters),
        )

        # Block 2: depthwise spatial conv -- learns per-filter spatial
        # patterns across all EEG channels (kernel spans full channel dim)
        spatial_filters = temporal_filters * depth_multiplier
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                temporal_filters, spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,  # depthwise
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3: separable temporal conv -- refines temporal patterns
        # per spatial filter, then mixes across filters (pointwise)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                spatial_filters, spatial_filters,
                kernel_size=(1, 16), padding="same",
                groups=spatial_filters, bias=False,  # depthwise
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_proj = nn.Linear(spatial_filters, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_timepoints) -> add channel dim for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, n_channels, n_timepoints)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.global_pool(x)          # (batch, spatial_filters, 1, 1)
        x = x.flatten(start_dim=1)       # (batch, spatial_filters)
        x = self.embed_proj(x)           # (batch, embed_dim)
        return x


class ProjectionHead(nn.Module):
    """
    SimCLR-style MLP head. Contrastive loss is applied HERE, not on the raw
    encoder output -- the projection head is discarded after pretraining;
    only EEGNetEncoder is reused for Phase 5 fine-tuning.
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLPretrainModel(nn.Module):
    """Wraps encoder + projection head for pretraining forward passes."""
    def __init__(self, n_channels: int, n_timepoints: int, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = EEGNetEncoder(n_channels, n_timepoints, embed_dim=embed_dim)
        self.projection_head = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
