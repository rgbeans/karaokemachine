"""Supervised training on the explicitly approved map(s)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import FEATURE_COUNT, MODEL_PATH
from .dataset import TrackData, augment_features, load_track, normalize
from .maps import load_approved_tracks
from .model import NotePolicy
from .notes import build_notes, match_note_sets
from .policy import Checkpoint, make_optimizer, save_checkpoint


@dataclass
class Validation:
    note_precision: float
    note_recall: float
    note_f1: float
    candidate_f1: float
    loss: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _validation_actions(
    logits: np.ndarray,
    pitch: np.ndarray,
    onset: np.ndarray,
    scale: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    probabilities = _sigmoid(logits)
    return {
        "select_prob": probabilities.astype(np.float32),
        "selected": (probabilities >= threshold).astype(np.int8),
        "pitch_offset": pitch.astype(np.float32),
        "onset_offset": onset.astype(np.float32),
        "log_scale": scale.astype(np.float32),
    }


def _forward_all(
    model: NotePolicy,
    data: TrackData,
    indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    indices = np.arange(len(data.candidates)) if indices is None else indices
    features = normalize(
        data.candidates.features[indices], data.feature_mean, data.feature_std
    )
    chunks: dict[str, list[np.ndarray]] = {
        "select_logit": [],
        "pitch_offset": [],
        "onset_offset": [],
        "log_scale": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], 4096):
            chunk = features[start : start + 4096]
            tensor = torch.from_numpy(chunk.T[None, :, :]).float()
            output = model(tensor)
            for key in chunks:
                chunks[key].append(output[key][0].cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def _evaluate_model(
    model: NotePolicy,
    data: TrackData,
    threshold: float = 0.50,
) -> Validation:
    val_indices = np.where(~data.train_mask)[0]
    if val_indices.size == 0:
        val_indices = np.arange(len(data.candidates))
    output = _forward_all(model, data, val_indices)
    probabilities = _sigmoid(output["select_logit"])
    selected = probabilities >= threshold
    labels = data.candidates.labels[val_indices]
    true_positive = int(np.logical_and(selected, labels > 0.5).sum())
    precision = true_positive / int(selected.sum()) if selected.any() else 0.0
    recall = true_positive / int((labels > 0.5).sum()) if (labels > 0.5).any() else 0.0
    candidate_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    actions = _validation_actions(
        output["select_logit"],
        output["pitch_offset"],
        output["onset_offset"],
        output["log_scale"],
        threshold,
    )
    predicted = build_notes(data.candidates, actions)
    note_precision, note_recall, note_f1 = match_note_sets(predicted, data.level.notes)
    with torch.no_grad():
        labels_tensor = torch.from_numpy(labels).float()
        logits = torch.from_numpy(output["select_logit"]).float()
        loss = float(F.binary_cross_entropy_with_logits(logits, labels_tensor).item())
    return Validation(
        note_precision=note_precision,
        note_recall=note_recall,
        note_f1=note_f1,
        candidate_f1=candidate_f1,
        loss=loss,
    )


def _full_song_note_f1(model: NotePolicy, data: TrackData, threshold: float = 0.50) -> float:
    output = _forward_all(model, data)
    actions = _validation_actions(
        output["select_logit"],
        output["pitch_offset"],
        output["onset_offset"],
        output["log_scale"],
        threshold,
    )
    predicted = build_notes(data.candidates, actions, song_duration=data.song.duration)
    return match_note_sets(predicted, data.level.notes)[2]


def train(
    epochs: int = 60,
    seed: int = 0,
    learning_rate: float = 1e-3,
    steps_per_epoch: int = 20,
    positive_batch: int = 128,
    negative_batch: int = 192,
    patience: int = 30,
    force_audio: bool = False,
) -> Checkpoint:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tracks = load_approved_tracks()
    data_sets = [load_track(track, force_audio=force_audio) for track in tracks]

    all_train_features = np.concatenate(
        [data.candidates.features[data.train_mask] for data in data_sets], axis=0
    )
    mean = all_train_features.mean(axis=0).astype(np.float32)
    std = all_train_features.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    for data in data_sets:
        data.feature_mean = mean
        data.feature_std = std

    total_candidates = sum(len(data.candidates) for data in data_sets)
    total_notes = sum(data.candidates.total_notes for data in data_sets)
    matched = sum(data.candidates.matched_notes for data in data_sets)
    print(f"Approved tracks: {len(data_sets)}")
    for data in data_sets:
        print(
            f"  {data.song.title}: {len(data.candidates):,} onset proposals, "
            f"{data.candidates.matched_notes}/{data.candidates.total_notes} notes matched"
        )
    print(
        f"Total: {total_candidates:,} proposals, {matched}/{total_notes} notes matched "
        f"({100.0 * matched / max(1, total_notes):.1f}%)"
    )
    if matched < total_notes:
        print("  warning: unmatched notes are excluded from selection labels")

    positives = sum(int(data.candidates.labels[data.train_mask].sum()) for data in data_sets)
    negatives = sum(
        int((data.candidates.labels[data.train_mask] <= 0.5).sum()) for data in data_sets
    )
    print(f"Training labels: {positives} positive, {negatives} negative (balanced batches)")

    model = NotePolicy(FEATURE_COUNT)
    optimizer = make_optimizer(model, learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps_per_epoch)
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    best_note_f1 = 0.0
    best_candidate_f1 = 0.0
    stale = 0
    step = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for _ in range(steps_per_epoch):
            for data in data_sets:
                positive = np.where(
                    (data.candidates.labels > 0.5) & data.train_mask
                )[0]
                negative = np.where(
                    (data.candidates.labels <= 0.5) & data.train_mask
                )[0]
                if positive.size == 0 or negative.size == 0:
                    continue
                chosen = np.concatenate(
                    [
                        rng.choice(positive, size=min(positive_batch, positive.size), replace=True),
                        rng.choice(negative, size=min(negative_batch, negative.size), replace=False),
                    ]
                )
                labels = torch.from_numpy(data.candidates.labels[chosen]).float()
                time_scale = float(rng.uniform(0.90, 1.10))
                pitch_shift = float(rng.uniform(-2.0, 2.0))
                raw = augment_features(
                    data.candidates.features[chosen], time_scale, pitch_shift
                )
                normalized = normalize(raw, mean, std)
                normalized = normalized + rng.normal(0.0, 0.02, normalized.shape).astype(np.float32)
                tensor = torch.from_numpy(normalized.T[None, :, :]).float()
                output = model(tensor)
                loss = F.binary_cross_entropy_with_logits(
                    output["select_logit"], labels[None, :]
                )
                mask = labels > 0.5
                if bool(mask.any()):
                    mask_np = mask.numpy()
                    loss = (
                        loss
                        + F.smooth_l1_loss(
                            output["pitch_offset"][0][mask],
                            torch.from_numpy(data.candidates.pitch_offsets[chosen][mask_np]).float(),
                        )
                        + 2.0
                        * F.smooth_l1_loss(
                            output["onset_offset"][0][mask],
                            torch.from_numpy(
                                data.candidates.onset_offsets[chosen][mask_np] * time_scale
                            ).float(),
                        )
                        + 0.5
                        * F.smooth_l1_loss(
                            output["log_scale"][0][mask],
                            torch.from_numpy(data.candidates.log_scales[chosen][mask_np]).float(),
                        )
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                epoch_losses.append(float(loss.item()))

        validations = [_evaluate_model(model, data) for data in data_sets]
        candidate_f1 = float(np.mean([item.candidate_f1 for item in validations]))
        note_f1 = float(np.mean([_full_song_note_f1(model, data) for data in data_sets]))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        marker = ""
        if note_f1 > best_score:
            best_score = note_f1
            best_note_f1 = note_f1
            best_candidate_f1 = candidate_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
            marker = " *"
        else:
            stale += 1
        print(
            f"epoch {epoch:03d}  loss={mean_loss:.3f}  "
            f"full_note_f1={note_f1:.3f}  val_candidate_f1={candidate_f1:.3f}"
            f"{marker}"
        )
        if stale >= patience:
            print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    checkpoint = Checkpoint(
        model=model,
        feature_mean=mean,
        feature_std=std,
        optimizer_state=optimizer.state_dict(),
        step=step,
        reward_baseline=0.0,
        reward_count=0,
        hidden=128,
        path=MODEL_PATH,
    )
    save_checkpoint(checkpoint)
    elapsed = time.time() - started
    print(
        f"best epoch {best_epoch}: full-song note F1={best_note_f1:.3f}, "
        f"held-out candidate F1={best_candidate_f1:.3f}"
    )
    print(f"saved {MODEL_PATH} after {step} updates in {elapsed:.1f}s")
    return checkpoint
