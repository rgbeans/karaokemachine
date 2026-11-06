"""Learned ribbon-link policy for adjacent vocal notes.

Ribbons in KaraokeMachine are explicit note-to-note connectors. The old builder
only linked a narrow gap range; this model learns the link decision from the
hand-mapped Man in the Box chart using gaps, pitch movement, duration and lyric
line context. It is intentionally a separate small model so the note denoiser
does not need another output head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import RIBBON_MODEL_PATH
from .maps import Level, Song, load_approved_tracks, load_level, load_song

RIBBON_METRICS_PATH = RIBBON_MODEL_PATH.parent.parent / "ribbon_metrics.json"
FEATURE_NAMES = (
    "gap",
    "abs_gap",
    "overlap",
    "pitch_diff",
    "prev_duration",
    "next_duration",
    "prev_midi_norm",
    "next_midi_norm",
    "time_frac",
    "index_frac",
    "same_line",
    "prev_line_pos",
    "next_line_pos",
    "gap_under_20ms",
    "pitch_under_3",
)


@dataclass
class RibbonRuntime:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    metrics: dict[str, Any]


def _line_for_time(song: Song | None, time: float) -> tuple[int, float, float]:
    if song is None or not song.lines:
        return -1, 0.0, 0.0
    starts = [float(line.get("start", 0.0)) for line in song.lines]
    position = int(np.searchsorted(starts, time, side="right") - 1)
    if position < 0:
        position = 0
    line = song.lines[position]
    start = float(line.get("start", 0.0))
    end = float(line.get("end", start))
    return position, start, max(0.1, end - start)


def _pair_features(
    song: Song | None,
    notes: list[dict[str, Any]],
    index: int,
) -> list[float]:
    previous = notes[index]
    following = notes[index + 1]
    prev_start = float(previous["start"])
    next_start = float(following["start"])
    prev_dur = float(previous["dur"])
    next_dur = float(following["dur"])
    gap = next_start - (prev_start + prev_dur)
    overlap = max(0.0, -gap)
    pitch_diff = abs(int(following["midi"]) - int(previous["midi"]))
    prev_line, prev_line_start, prev_line_duration = _line_for_time(song, prev_start)
    next_line, next_line_start, next_line_duration = _line_for_time(song, next_start)
    same_line = 1.0 if prev_line == next_line and prev_line >= 0 else 0.0
    return [
        float(np.clip(gap, -2.0, 2.0)),
        float(min(2.0, abs(gap))),
        float(min(1.0, overlap)),
        pitch_diff / 12.0,
        min(4.0, prev_dur),
        min(4.0, next_dur),
        (float(previous["midi"]) - 60.0) / 24.0,
        (float(following["midi"]) - 60.0) / 24.0,
        prev_start / max(1.0, song.duration if song else 1.0),
        index / max(1, len(notes) - 1),
        same_line,
        float(np.clip((prev_start - prev_line_start) / prev_line_duration, 0.0, 1.0)),
        float(np.clip((next_start - next_line_start) / next_line_duration, 0.0, 1.0)),
        1.0 if gap <= 0.02 else 0.0,
        1.0 if pitch_diff <= 3 else 0.0,
    ]


def _pairs(
    song: Song | None,
    notes: list[dict[str, Any]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    features = [
        _pair_features(song, notes, index) for index in range(len(notes) - 1)
    ]
    return np.asarray(features, dtype=np.float32), [
        (index, index + 1) for index in range(len(notes) - 1)
    ]


def _link_labels(links: list[dict[str, Any]], count: int) -> np.ndarray:
    labels = np.zeros(count, dtype=np.float32)
    for link in links:
        left = int(link.get("from", -1))
        right = int(link.get("to", -1))
        if left < 0 or right < 0 or abs(left - right) != 1:
            continue
        index = min(left, right)
        if index < count:
            labels[index] = 1.0
    return labels


def _training_pair(song: Song, level: Level) -> tuple[np.ndarray, np.ndarray]:
    features, _ = _pairs(song, level.notes)
    return features, _link_labels(level.ribbon_links, len(features))


def train_ribbon_model(epochs: int = 600, seed: int = 5) -> dict[str, Any]:
    torch.manual_seed(seed)
    training: list[tuple[np.ndarray, np.ndarray]] = []
    for track in load_approved_tracks():
        song = load_song(track.song_id)
        level = load_level(track.song_id, track.map_relative)
        features, labels = _training_pair(song, level)
        if features.size:
            training.append((features, labels))
    if not training:
        raise RuntimeError("no ribbon training map was found")
    features = np.concatenate([item[0] for item in training], axis=0)
    labels = np.concatenate([item[1] for item in training], axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-5] = 1.0
    normalized = (features - mean) / std
    model = nn.Sequential(
        nn.Linear(features.shape[1], 32),
        nn.GELU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels)
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    pos_weight = torch.tensor(negative / max(1.0, positive), dtype=torch.float32)
    for _ in range(epochs):
        output = model(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, y, pos_weight=pos_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x).squeeze(-1)).numpy()
    best = None
    for threshold in np.arange(0.05, 0.96, 0.01):
        predicted = probabilities >= threshold
        truth = labels > 0.5
        tp = int((predicted & truth).sum())
        precision = tp / max(1, int(predicted.sum()))
        recall = tp / max(1, int(truth.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if best is None or f1 > best[0]:
            best = (f1, threshold, precision, recall, int(predicted.sum()))
    assert best is not None
    metrics = {
        "model": "learned adjacent-note ribbon policy",
        "features": list(FEATURE_NAMES),
        "threshold": float(best[1]),
        "f1": float(best[0]),
        "precision": float(best[2]),
        "recall": float(best[3]),
        "links": best[4],
        "pairs": len(labels),
        "positive": int(positive),
    }
    RIBBON_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "threshold": metrics["threshold"],
            "feature_names": list(FEATURE_NAMES),
            "metrics": metrics,
        },
        RIBBON_MODEL_PATH,
    )
    RIBBON_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved {RIBBON_MODEL_PATH} · ribbon F1 {metrics['f1']:.3f} "
        f"at threshold {metrics['threshold']:.2f}"
    )
    return metrics


def load_ribbon_model() -> RibbonRuntime | None:
    if not RIBBON_MODEL_PATH.exists():
        return None
    try:
        payload = torch.load(RIBBON_MODEL_PATH, map_location="cpu", weights_only=False)
        if payload.get("feature_names") != list(FEATURE_NAMES):
            return None
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        model = nn.Sequential(
            nn.Linear(mean.size, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        model.load_state_dict(payload["model"])
        model.eval()
        return RibbonRuntime(
            model=model,
            mean=mean,
            std=std,
            threshold=float(payload.get("threshold", 0.5)),
            metrics=payload.get("metrics") or {},
        )
    except Exception as error:
        print(f"warning: could not load ribbon model: {error}")
        return None


def predict_ribbon_links(
    song: Song | None,
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runtime = load_ribbon_model()
    if runtime is None or len(notes) < 2:
        return []
    features, pairs = _pairs(song, notes)
    normalized = (features - runtime.mean) / runtime.std
    with torch.no_grad():
        probabilities = torch.sigmoid(
            runtime.model(torch.from_numpy(normalized)).squeeze(-1)
        ).numpy()
    links: list[dict[str, Any]] = []
    for (left, right), probability in zip(pairs, probabilities):
        if probability >= runtime.threshold:
            links.append(
                {
                    "from": left,
                    "to": right,
                    "fromSide": "right",
                    "toSide": "left",
                }
            )
    return links
