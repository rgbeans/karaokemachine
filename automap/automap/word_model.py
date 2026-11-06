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
) + tuple(f"source_{source.title}" for source in karaoke.SOURCES) + tuple(
    f"line_source_{source.title}" for source in karaoke.LINE_SOURCES
)


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
    source: karaoke.TimingSource,
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
    source_index = karaoke.source_index(source)
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
    values.extend(
        1.0 if index == source_index else 0.0
        for index in range(len(karaoke.SOURCES) + len(karaoke.LINE_SOURCES))
    )
    return values


def _rows_for_song(
    song: Song,
) -> tuple[
    list[list[float]],
    list[float],
    list[float],
    karaoke.TimingSource,
    list[karaoke.WordTarget],
]:
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
    return features, residuals, local_times, source, targets


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


def _line_mae(
    model: nn.Module,
    features: list[list[float]],
    targets: list[karaoke.WordTarget],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[float, float, int]:
    """Measure line placement error, not interpolated word error.

    A line-level source gives one timestamp per lyric line. Predictions for all
    words in a line are averaged into a single line shift, then compared with
    the public line-start shift.
    """
    if not features:
        return 0.0, 0.0, 0
    normalized = (np.asarray(features, dtype=np.float32) - mean) / std
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized)).squeeze(-1).numpy()
    grouped: dict[int, list[tuple[float, float]]] = {}
    for prediction, target in zip(predictions, targets):
        if target.level != "line":
            continue
        grouped.setdefault(target.line_index, []).append(
            (float(prediction), float(target.target_time - target.local_time))
        )
    if not grouped:
        return 0.0, 0.0, 0
    errors: list[float] = []
    baseline: list[float] = []
    for pairs in grouped.values():
        predicted = float(np.mean([pair[0] for pair in pairs]))
        expected = float(np.mean([pair[1] for pair in pairs]))
        errors.append(abs(predicted - expected))
        baseline.append(abs(expected))
    baseline_mae = float(np.mean(baseline))
    return (
        float(np.mean(errors)),
        0.0 if baseline_mae < 1e-6 else baseline_mae,
        len(grouped),
    )


def train_word_model(epochs: int = 2000, seed: int = 2) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_features: list[list[float]] = []
    train_residuals: list[float] = []
    train_targets: list[karaoke.WordTarget] = []
    per_song: dict[str, dict[str, Any]] = {}
    all_sources: list[karaoke.TimingSource] = [
        *karaoke.SOURCES,
        *karaoke.LINE_SOURCES,
    ]
    for source in all_sources:
        song = load_song(source.song_id)
        features, residuals, _, source, targets = _rows_for_song(song)
        train_features.extend(features)
        train_residuals.extend(residuals)
        train_targets.extend(targets)
        item: dict[str, Any] = {
            "title": song.title,
            "level": source.level,
            "source_url": source.source_url,
            "offset": source.offset,
            "words": len(residuals),
            "baseline_mae": float(np.mean(np.abs(residuals))) if residuals else 0.0,
            "_features": features,
            "_residuals": residuals,
            "_targets": targets,
        }
        if isinstance(source, karaoke.KaraokeSource):
            item["usdb_id"] = source.usdb_id
            item["notes_url"] = source.notes_url
        else:
            item["lrclib_id"] = source.lrclib_id
            item["lyrics_url"] = source.lyrics_url
        per_song[song.id] = item
        if source.level == "line":
            print(
                f"  {song.title}: {len(residuals)} words across "
                f"{len({target.line_index for target in targets})} public lines"
            )
        else:
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
        "source": "USDB word timings and LRCLIB line timings",
        "feature_names": list(FEATURE_NAMES),
        "overall_baseline_mae": 0.0,
        "overall_mae": 0.0,
        "overall_line_baseline_mae": 0.0,
        "overall_line_mae": 0.0,
        "songs": {},
    }
    line_maes: list[float] = []
    line_baselines: list[float] = []
    for song_id, item in per_song.items():
        features_for_song = item.pop("_features")
        residuals_for_song = item.pop("_residuals")
        targets_for_song = item.pop("_targets")
        if item["level"] == "line":
            mae, baseline, line_count = _line_mae(
                model, features_for_song, targets_for_song, mean, std
            )
            item["lines"] = line_count
            line_maes.append(mae)
            line_baselines.append(baseline)
        else:
            mae = _target_mae(
                model, features_for_song, residuals_for_song, mean, std
            )
        item["mae"] = mae
        if item["level"] == "line":
            item["baseline_mae"] = baseline
        metrics["songs"][song_id] = item
        print(
            f"  {item['title']}: baseline {item['baseline_mae'] * 1000:.0f} ms, "
            f"model {mae * 1000:.0f} ms"
        )

    # Recompute overall from a final pass over all targets. Word-level and
    # line-level errors stay separate because the LRCLIB source has no
    # independent word timestamps.
    features = np.asarray(train_features, dtype=np.float32)
    normalized_all = (features - mean) / std
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized_all)).squeeze(-1).numpy()
    word_mask = np.asarray(
        [target.level == "word" for target in train_targets], dtype=bool
    )
    if word_mask.any():
        metrics["overall_mae"] = float(
            np.mean(np.abs(predictions[word_mask] - expected[word_mask]))
        )
        metrics["overall_baseline_mae"] = float(
            np.mean(np.abs(expected[word_mask]))
        )
    metrics["overall_line_mae"] = (
        float(np.mean(line_maes)) if line_maes else 0.0
    )
    metrics["overall_line_baseline_mae"] = (
        float(np.mean(line_baselines)) if line_baselines else 0.0
    )
    metrics["sources"] = [
        {
            "song_id": source.song_id,
            "usdb_id": source.usdb_id,
            "title": source.title,
            "level": source.level,
            "offset": source.offset,
            "source_url": source.source_url,
            "notes_url": source.notes_url,
        }
        for source in karaoke.SOURCES
    ]
    metrics["line_sources"] = [
        {
            "song_id": source.song_id,
            "lrclib_id": source.lrclib_id,
            "title": source.title,
            "level": source.level,
            "offset": source.offset,
            "source_url": source.source_url,
            "lyrics_url": source.lyrics_url,
        }
        for source in karaoke.LINE_SOURCES
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
    offsets: dict[str, dict[str, Any]] = {}
    for source in karaoke.SOURCES:
        offsets[source.song_id] = {
            "usdb_id": source.usdb_id,
            "title": source.title,
            "level": source.level,
            "offset": source.offset,
            "source_url": source.source_url,
            "notes_url": source.notes_url,
        }
    for source in karaoke.LINE_SOURCES:
        offsets[source.song_id] = {
            "lrclib_id": source.lrclib_id,
            "title": source.title,
            "level": source.level,
            "offset": source.offset,
            "source_url": source.source_url,
            "lyrics_url": source.lyrics_url,
        }
    WORD_OFFSETS_PATH.write_text(
        json.dumps(offsets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved {WORD_MODEL_PATH} · overall word MAE "
        f"{metrics['overall_mae'] * 1000:.0f} ms "
        f"(baseline {metrics['overall_baseline_mae'] * 1000:.0f} ms)"
    )
    if line_maes:
        print(
            f"  line-level MAE {metrics['overall_line_mae'] * 1000:.0f} ms "
            f"(baseline {metrics['overall_line_baseline_mae'] * 1000:.0f} ms)"
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
    source: karaoke.TimingSource,
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
            "level": item.get("level", "word"),
            "usdb_id": item.get("usdb_id"),
            "lrclib_id": item.get("lrclib_id"),
            "mae": item.get("mae"),
            "baseline_mae": item.get("baseline_mae"),
            "words": item.get("words"),
            "lines": item.get("lines"),
            "source_url": item.get("source_url"),
        }
    except Exception:
        return None
