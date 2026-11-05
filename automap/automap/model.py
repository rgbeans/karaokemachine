"""The note-proposal policy network."""

from __future__ import annotations

import torch
from torch import nn


class NotePolicy(nn.Module):
    """Scores one note-onset proposal at a time.

    Input shape is ``[batch, features, candidates]``. The output is one keep
    score and three correction means per proposal. A per-onset MLP is small
    enough to learn from a single approved map; sequence context is already
    represented by the onset, pitch, gap and lyric features.
    """

    def __init__(self, feature_count: int, hidden: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        # [batch, features, candidates] -> [batch, candidates, features]
        value = features.transpose(1, 2)
        output = self.network(value)
        return {
            "select_logit": output[:, :, 0],
            "pitch_offset": 3.0 * torch.tanh(output[:, :, 1]),
            "onset_offset": 0.08 * torch.tanh(output[:, :, 2]),
            "log_scale": 1.50 * torch.tanh(output[:, :, 3]),
        }
