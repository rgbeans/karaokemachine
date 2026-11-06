"""Policy sampling, likelihoods and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import (
    DURATION_SIGMA,
    ONSET_SIGMA,
    PITCH_SIGMA,
    SELECT_TEMPERATURE,
)
from .model import NotePolicy


@dataclass
class Checkpoint:
    model: NotePolicy
    feature_mean: np.ndarray
    feature_std: np.ndarray
    optimizer_state: dict[str, Any] | None
    step: int
    reward_baseline: float
    reward_count: int
    hidden: int
    path: Path
    feature_names: list[str] | None = None


def as_tensor(features: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(features.T[None, :, :])).float()


def calibrate_selection_bias(
    logits: torch.Tensor,
    target_count: int,
    temperature: float = SELECT_TEMPERATURE,
) -> float:
    """Find a logit offset whose expected selection count matches target_count."""
    values = logits.detach() / max(1e-3, temperature)
    low, high = -40.0, 40.0
    for _ in range(60):
        middle = (low + high) / 2.0
        expected = float(torch.sigmoid(values + middle).sum().item())
        if expected < target_count:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def sample_actions(
    model: NotePolicy,
    normalized_features: np.ndarray,
    temperature: float = SELECT_TEMPERATURE,
    seed: int | None = None,
    target_notes: int | None = None,
) -> dict[str, np.ndarray]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    tensor = as_tensor(normalized_features)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        raw_logits = output["select_logit"][0]
        selection_bias = (
            calibrate_selection_bias(raw_logits, target_notes, temperature)
            if target_notes is not None
            else 0.0
        )
        logits = raw_logits / max(1e-3, temperature) + selection_bias
        probabilities = torch.sigmoid(logits)
        selected = torch.bernoulli(probabilities, generator=generator)
        pitch_mu = output["pitch_offset"][0]
        onset_mu = output["onset_offset"][0]
        scale_mu = output["log_scale"][0]
        pitch = pitch_mu + PITCH_SIGMA * torch.randn(pitch_mu.shape, generator=generator)
        onset = onset_mu + ONSET_SIGMA * torch.randn(onset_mu.shape, generator=generator)
        scale = scale_mu + DURATION_SIGMA * torch.randn(scale_mu.shape, generator=generator)
    return {
        "select_logit": logits.cpu().numpy().astype(np.float32),
        "select_prob": probabilities.cpu().numpy().astype(np.float32),
        "selection_bias": np.array([selection_bias], dtype=np.float32),
        "selected": selected.cpu().numpy().astype(np.int8),
        "pitch_offset": pitch.cpu().numpy().astype(np.float32),
        "onset_offset": onset.cpu().numpy().astype(np.float32),
        "log_scale": scale.cpu().numpy().astype(np.float32),
        "pitch_mu": pitch_mu.cpu().numpy().astype(np.float32),
        "onset_mu": onset_mu.cpu().numpy().astype(np.float32),
        "scale_mu": scale_mu.cpu().numpy().astype(np.float32),
    }


def action_log_prob(
    model: NotePolicy,
    normalized_features: np.ndarray,
    actions: dict[str, np.ndarray],
    temperature: float = SELECT_TEMPERATURE,
    selection_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-candidate log probability and selection entropy."""
    tensor = as_tensor(normalized_features)
    output = model(tensor)
    logits = output["select_logit"][0] / max(1e-3, temperature) + selection_bias
    selected = torch.from_numpy(actions["selected"].astype(np.float32)).float()
    select_log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(selected)
    entropy = torch.distributions.Bernoulli(logits=logits).entropy()

    pitch = torch.from_numpy(actions["pitch_offset"]).float()
    onset = torch.from_numpy(actions["onset_offset"]).float()
    scale = torch.from_numpy(actions["log_scale"]).float()
    pitch_log = torch.distributions.Normal(output["pitch_offset"][0], PITCH_SIGMA).log_prob(pitch)
    onset_log = torch.distributions.Normal(output["onset_offset"][0], ONSET_SIGMA).log_prob(onset)
    scale_log = torch.distributions.Normal(output["log_scale"][0], DURATION_SIGMA).log_prob(scale)
    correction_log = (pitch_log + onset_log + scale_log) * selected
    return select_log_prob + correction_log, entropy


def save_checkpoint(checkpoint: Checkpoint) -> None:
    checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model": checkpoint.model.state_dict(),
            "feature_mean": checkpoint.feature_mean,
            "feature_std": checkpoint.feature_std,
            "optimizer": checkpoint.optimizer_state,
            "step": checkpoint.step,
            "reward_baseline": checkpoint.reward_baseline,
            "reward_count": checkpoint.reward_count,
            "hidden": checkpoint.hidden,
            "feature_names": checkpoint.feature_names,
        },
        checkpoint.path,
    )


def load_checkpoint(path: Path) -> Checkpoint:
    if not path.exists():
        raise FileNotFoundError(
            f"no model checkpoint at {path}; run `python -m automap train` first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(payload.get("hidden", 64))
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    model = NotePolicy(feature_count=int(feature_mean.size), hidden=hidden)
    model.load_state_dict(payload["model"])
    model.eval()
    return Checkpoint(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        optimizer_state=payload.get("optimizer"),
        step=int(payload.get("step", 0)),
        reward_baseline=float(payload.get("reward_baseline", 0.0)),
        reward_count=int(payload.get("reward_count", 0)),
        hidden=hidden,
        path=path,
        feature_names=payload.get("feature_names"),
    )


def make_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
