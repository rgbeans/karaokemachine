"""Interactive grading loop and policy-gradient update."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    FEEDBACK_PATH,
    MODEL_PATH,
    TEACHER_MODEL_PATH,
    POLICY_GRADIENT_SCALE,
    REWARD_BY_GRADE,
    ensure_dirs,
)
from .dataset import TrackData, load_track, normalize
from .generate import pending_path
from .maps import load_approved_tracks
from .policy import (
    action_log_prob,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
)


def load_pending(song_id: str) -> dict[str, np.ndarray] | None:
    path = pending_path(song_id)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _anchor_loss(
    model: torch.nn.Module,
    anchor_data: list[TrackData],
    rng: np.random.Generator,
    chunk_size: int = 4096,
    pos_weight: float = 20.0,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for data in anchor_data:
        indices = np.where(data.train_mask)[0]
        if indices.size == 0:
            continue
        start = int(rng.integers(0, max(1, indices.size - chunk_size + 1)))
        chunk = indices[start : start + chunk_size]
        if chunk.size == 0:
            continue
        normalized = normalize(
            data.candidates.features[chunk], data.feature_mean, data.feature_std
        )
        tensor = torch.from_numpy(normalized.T[None, :, :]).float()
        labels = torch.from_numpy(data.candidates.labels[chunk]).float()[None, :]
        output = model(tensor)
        loss = F.binary_cross_entropy_with_logits(
            output["select_logit"], labels, pos_weight=torch.tensor(pos_weight)
        )
        positive = labels > 0.5
        if bool(positive.any()):
            loss = (
                loss
                + F.smooth_l1_loss(
                    output["pitch_offset"][positive],
                    torch.from_numpy(data.candidates.pitch_offsets[chunk]).float()[None, :][positive],
                )
                + 2.0
                * F.smooth_l1_loss(
                    output["onset_offset"][positive],
                    torch.from_numpy(data.candidates.onset_offsets[chunk]).float()[None, :][positive],
                )
                + 0.5
                * F.smooth_l1_loss(
                    output["log_scale"][positive],
                    torch.from_numpy(data.candidates.log_scales[chunk]).float()[None, :][positive],
                )
            )
        losses.append(loss)
    if not losses:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def update_from_grade(
    song_id: str,
    grade: int,
    anchor_data: list[TrackData] | None = None,
    learning_rate: float = 2e-4,
    entropy_beta: float = 0.01,
    anchor_weight: float = 0.5,
    model_path: Path = MODEL_PATH,
    seed: int = 0,
) -> dict[str, object]:
    ensure_dirs()
    pending = load_pending(song_id)
    if pending is None:
        raise FileNotFoundError(f"no pending candidate for {song_id}; run generate first")
    pending_model = pending.get("model_path")
    if pending_model is not None and str(pending_model[0]).strip():
        model_path = Path(str(pending_model[0]))
    if model_path == TEACHER_MODEL_PATH:
        # The UI session keeps the old one-song anchor warm; teacher candidates
        # use a different feature layout and need their own replay anchor.
        anchor_data = None
    checkpoint = load_checkpoint(model_path)
    if model_path == TEACHER_MODEL_PATH and checkpoint.feature_names is None:
        from .teacher import TEACHER_FEATURE_NAMES

        checkpoint.feature_names = list(TEACHER_FEATURE_NAMES)
    reward = REWARD_BY_GRADE[grade]
    baseline = checkpoint.reward_baseline if checkpoint.reward_count > 0 else 0.0
    advantage = float(np.clip(reward - baseline, -2.0, 2.0))
    rng = np.random.default_rng(seed + checkpoint.step)
    normalized = normalize(
        pending["features"], checkpoint.feature_mean, checkpoint.feature_std
    )
    actions = {
        "selected": pending["selected"],
        "pitch_offset": pending["pitch_offset"],
        "onset_offset": pending["onset_offset"],
        "log_scale": pending["log_scale"],
    }
    selection_bias = (
        float(pending["selection_bias"][0]) if "selection_bias" in pending else 0.0
    )

    policy_value = 0.0
    anchor_value = 0.0
    total_value = 0.0
    if reward != 0.0:
        if anchor_data is None:
            if model_path == TEACHER_MODEL_PATH:
                from .teacher import load_teacher_anchor_data

                anchor_data = load_teacher_anchor_data()
            else:
                anchor_data = [load_track(track) for track in load_approved_tracks()]
        model = checkpoint.model
        model.train()
        optimizer = make_optimizer(model, learning_rate)
        log_prob, entropy = action_log_prob(
            model, normalized, actions, selection_bias=selection_bias
        )
        policy_loss = (
            -advantage * log_prob.mean() * POLICY_GRADIENT_SCALE
            - entropy_beta * entropy.mean()
        )
        anchor = _anchor_loss(model, anchor_data, rng)
        loss = policy_loss + anchor_weight * anchor
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        checkpoint.step += 1
        policy_value = float(policy_loss.item())
        anchor_value = float(anchor.item())
        total_value = float(loss.item())

    if reward != 0.0:
        checkpoint.reward_count += 1
        if checkpoint.reward_count == 1:
            checkpoint.reward_baseline = reward
        else:
            checkpoint.reward_baseline = (
                0.9 * checkpoint.reward_baseline + 0.1 * reward
            )
    checkpoint.optimizer_state = None
    save_checkpoint(checkpoint)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "song_id": song_id,
        "archive": str(pending["archive"][0]),
        "installed": str(pending["installed"][0]),
        "grade": grade,
        "reward": reward,
        "advantage": advantage,
        "policy_loss": policy_value,
        "anchor_loss": anchor_value,
        "loss": total_value,
        "baseline": checkpoint.reward_baseline,
        "step": checkpoint.step,
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pending_path(song_id).unlink(missing_ok=True)
    return record


def parse_grade(text: str) -> int | None:
    value = text.strip().lower()
    if value in {"+", "good", "g", "5"}:
        return 5
    if value in {"-", "bad", "b", "1"}:
        return 1
    if value in {"0", "s", "skip"}:
        return 3
    if value in {"1", "2", "3", "4", "5"}:
        return int(value)
    return None


def print_pending(song_id: str) -> bool:
    pending = load_pending(song_id)
    if pending is None:
        return False
    title = str(pending["title"][0])
    notes = int(pending["note_count"][0])
    print(f"Candidate ready: {title} — {notes} generated notes")
    print(f"  archive: {str(pending['archive'][0])}")
    if str(pending["installed"][0]):
        print(f"  installed: {str(pending['installed'][0])}")
        print("  Restart KaraokeMachine, choose AI-Candidate for this song, and listen.")
    return True
