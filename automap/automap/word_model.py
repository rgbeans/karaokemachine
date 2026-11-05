"""Word-placement calibration network trained on public karaoke timings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from . import karaoke
from .audio import load_or_extract
from .config import (
    WORD_METRICS_PATH,
    WORD_MODEL_PATH,
    WORD_OFFSETS_PATH,
)
from .maps import Song, load_song

FEATURE_NAMES = (
    "local_start_frac",
    "local_duration",
    "word_length",
    "word_index_frac",
    "line_position",
    "line_frac",
    "line_start_frac",
    "line_duration",
    "previous_gap",
    "next_gap",
    "source_offset",
    "midi_norm",
    "rms",
    "onset",
    "voiced_ratio",
    "song_duration",
) + tuple(f"source_{source.title}" for source in karaoke.SOURCES)


@dataclass
class WordRuntime:
    model: nn.Module
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metrics: dict[str, Any]


def _audio_stats(audio: dict[str, Any], time: float) -> tuple[float, float, float, float]:
    fps = float(audio["fps"])
    midi = np.asarray(audio["midi"], dtype=np.float32)
    rms = np.asarray(audio["rms"], dtype=np.float32)
    flux = np.asarray(audio["flux"], dtype=np.float32)
    voiced = np.asarray(audio["voiced"], dtype=bool)
    index = int(round(time * fps))
    low = max(0, index - int(0.2 * fps))
    high = min(midi.size, index + int(0.2 * fps) + 1)
    if high <= low:
        return 60.0, 0.0, 0.0, 0.0
    pitches = midi[low:high]
    finite = pitches[np.isfinite(pitches)]
    median = float(np.median(finite)) if finite.size else 60.0
    return (
        median,
        float(np.mean(rms[low:high])),
        float(np.max(flux[low:high])),
        float(np.mean(voiced[low:high])),
    )


def word_features(
    song: Song,
    audio: dict[str, Any],
    line: dict[str, Any],
    line_index: int,
    word: dict[str, Any],
    word_index: int,
    word_count: int,
    source: karaoke.KaraokeSource,
) -> list[float]:
    local = float(word.get("start", 0.0))
    line_start = float(line.get("start", 0.0))
    line_end = float(line.get("end", line_start))
    line_duration = max(0.1, line_end - line_start)
    line_words = list(line.get("words") or [])
    previous = (
        float(line_words[word_index - 1].get("start", local))
        if 0 < word_index <= len(line_words)
        else local - 1.0
    )
    following = (
        float(line_words[word_index + 1].get("start", local))
        if word_index + 1 < len(line_words)
        else local + 1.0
    )
    midi, rms, onset, voiced = _audio_stats(audio, local)
    text = karaoke.normalize(str(word.get("text") or ""))
    source_index = karaoke.SOURCES.index(source)
    values = [
        local / max(1.0, song.duration),
        max(0.05, float(word.get("end", local)) - local),
        len(text) / 12.0,
        word_index / max(1, word_count),
        (local - line_start) / line_duration,
        line_index / max(1, len(song.lines)),
        line_start / max(1.0, song.duration),
        line_duration,
        min(2.0, abs(local - previous)) / 2.0,
        min(2.0, abs(following - local)) / 2.0,
        source.offset,
        midi / 12.0,
        rms,
        min(4.0, onset),
        voiced,
        song.duration / 400.0,
    ]
    values.extend(1.0 if index == source_index else 0.0 for index in range(len(karaoke.SOURCES)))
    return values


def _rows_for_song(song: Song) -> tuple[list[list[float]], list[float], list[float], karaoke.KaraokeSource]:
    targets, source = karaoke.load_targets(song)
    if source is None:
        raise RuntimeError(f"{song.title} has no public karaoke timing source")
    audio = load_or_extract(song)
    features: list[list[float]] = []
    residuals: list[float] = []
    local_times: list[float] = []
    for target in targets:
        line = song.lines[target.line_index]
        features.append(
            word_features(
                song,
                audio,
                line,
                target.line_index,
                target.word,
                target.word_index,
                target.word_count,
                source,
            )
        )
        residuals.append(target.target_time - target.local_time)
        local_times.append(target.local_time)
    return features, residuals, local_times, source


def _target_mae(
    model: nn.Module,
    features: list[list[float]],
    residuals: list[float],
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    if not features:
        return 0.0
    normalized = (np.asarray(features, dtype=np.float32) - mean) / std
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized)).squeeze(-1).numpy()
    expected = np.asarray(residuals, dtype=np.float32)
    return float(np.mean(np.abs(predictions - expected)))


def train_word_model(epochs: int = 2000, seed: int = 2) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_features: list[list[float]] = []
    train_residuals: list[float] = []
    per_song: dict[str, dict[str, Any]] = {}
    for source in karaoke.SOURCES:
        song = load_song(source.song_id)
        features, residuals, _, source = _rows_for_song(song)
        train_features.extend(features)
        train_residuals.extend(residuals)
        per_song[song.id] = {
            "title": song.title,
            "usdb_id": source.usdb_id,
            "source_url": source.source_url,
            "notes_url": source.notes_url,
            "offset": source.offset,
            "words": len(residuals),
            "baseline_mae": float(np.mean(np.abs(residuals))) if residuals else 0.0,
            "_features": features,
            "_residuals": residuals,
        }
        print(f"  {song.title}: {len(residuals)} word targets")

    if not train_features:
        raise RuntimeError("no public karaoke word targets were found")

    mean = np.asarray(train_features, dtype=np.float32).mean(axis=0)
    std = np.asarray(train_features, dtype=np.float32).std(axis=0)
    std[std < 1e-5] = 1.0
    normalized = (np.asarray(train_features, dtype=np.float32) - mean) / std
    expected = np.asarray(train_residuals, dtype=np.float32)
    model = nn.Sequential(
        nn.Linear(normalized.shape[1], 128),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(normalized))
        losses: list[float] = []
        for start in range(0, len(indices), 64):
            batch = indices[start : start + 64]
            features = torch.from_numpy(normalized[batch])
            target = torch.from_numpy(expected[batch])
            output = model(features).squeeze(-1)
            loss = torch.nn.functional.smooth_l1_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch % 250 == 0:
            print(f"epoch {epoch:04d} loss={np.mean(losses):.4f}")

    metrics: dict[str, Any] = {
        "source": "USDB public UltraStar note timings",
        "feature_names": list(FEATURE_NAMES),
        "overall_baseline_mae": float(np.mean(np.abs(expected))),
        "overall_mae": 0.0,
        "songs": {},
    }
    for song_id, item in per_song.items():
        mae = _target_mae(model, item.pop("_features"), item.pop("_residuals"), mean, std)
        item["mae"] = mae
        metrics["songs"][song_id] = item
        print(
            f"  {item['title']}: baseline {item['baseline_mae'] * 1000:.0f} ms, "
            f"model {mae * 1000:.0f} ms"
        )
    # Recompute overall from a final pass over all songs.
    features = np.asarray(train_features, dtype=np.float32)
    normalized_all = (features - mean) / std
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized_all)).squeeze(-1).numpy()
    metrics["overall_mae"] = float(np.mean(np.abs(predictions - expected)))
    metrics["overall_baseline_mae"] = float(np.mean(np.abs(expected)))
    metrics["sources"] = [
        {
            "song_id": source.song_id,
            "usdb_id": source.usdb_id,
            "title": source.title,
            "offset": source.offset,
            "source_url": source.source_url,
            "notes_url": source.notes_url,
        }
        for source in karaoke.SOURCES
    ]

    WORD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "hidden": 128,
            "metrics": metrics,
        },
        WORD_MODEL_PATH,
    )
    WORD_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    WORD_OFFSETS_PATH.write_text(
        json.dumps(
            {
                source.song_id: {
                    "usdb_id": source.usdb_id,
                    "title": source.title,
                    "offset": source.offset,
                    "source_url": source.source_url,
                    "notes_url": source.notes_url,
                }
                for source in karaoke.SOURCES
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"saved {WORD_MODEL_PATH} · overall word MAE "
        f"{metrics['overall_mae'] * 1000:.0f} ms "
        f"(baseline {metrics['overall_baseline_mae'] * 1000:.0f} ms)"
    )
    return metrics


def load_word_model() -> WordRuntime | None:
    if not WORD_MODEL_PATH.exists():
        return None
    try:
        payload = torch.load(WORD_MODEL_PATH, map_location="cpu", weights_only=False)
        mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        std = np.asarray(payload["feature_std"], dtype=np.float32)
        model = nn.Sequential(
            nn.Linear(int(mean.size), 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        model.load_state_dict(payload["model"])
        model.eval()
        return WordRuntime(
            model=model,
            feature_mean=mean,
            feature_std=std,
            metrics=payload.get("metrics") or {},
        )
    except Exception as error:
        print(f"warning: could not load word model: {error}")
        return None


def predict_word_time(
    runtime: WordRuntime,
    song: Song,
    audio: dict[str, Any],
    source: karaoke.KaraokeSource,
    line: dict[str, Any],
    line_index: int,
    word: dict[str, Any],
    word_index: int,
    word_count: int,
) -> float:
    features = word_features(
        song, audio, line, line_index, word, word_index, word_count, source
    )
    normalized = (np.asarray(features, dtype=np.float32) - runtime.feature_mean) / runtime.feature_std
    with torch.no_grad():
        residual = float(runtime.model(torch.from_numpy(normalized)[None, :]).item())
    local = float(word.get("start", 0.0))
    line_start = float(line.get("start", local))
    line_end = float(line.get("end", local))
    return float(np.clip(local + residual, line_start - 0.5, line_end + 0.5))


def alignment_for_song(song_id: str) -> dict[str, Any] | None:
    if not WORD_METRICS_PATH.exists():
        return None
    try:
        payload = json.loads(WORD_METRICS_PATH.read_text(encoding="utf-8"))
        item = payload.get("songs", {}).get(song_id)
        if item is None:
            return None
        return {
            "source": payload.get("source", "USDB"),
            "usdb_id": item.get("usdb_id"),
            "mae": item.get("mae"),
            "baseline_mae": item.get("baseline_mae"),
            "words": item.get("words"),
            "source_url": item.get("source_url"),
        }
    except Exception:
        return None
