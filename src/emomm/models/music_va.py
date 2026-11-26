"""Point regression for music valence and arousal."""

from __future__ import annotations

import torch
from torch import nn


class MusicVARegressor(nn.Module):
    """Return two VA coordinates in [-1, 1]."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.network(embeddings)) - 1.0
