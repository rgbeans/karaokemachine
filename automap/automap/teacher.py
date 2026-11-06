"""Teacher-guided note generation.

The app already has a much stronger melody extractor than the one-song policy:
``tools/analyze_vocals.py`` tracks the isolated vocal stem and segments stable
pitches into notes. That dense chart is a good proposal set but is too noisy to
ship unchanged. This module learns a denoiser over those teacher notes using
the hand-mapped Man in the Box chart and the public USDB note tracks as
references.

The model does not invent note onsets from an arbitrary candidate grid. It
scores teacher notes, chooses a target density, and applies small pitch/onset/
duration corrections. That makes the learning problem much closer to "which
sung notes did the human keep?" instead of "guess a note from thousands of
acoustic frames".
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import karaoke, note_calibration
from .audio import load_or_extract
from .candidates import CandidateSet, label_candidates
from .config import (
    PROJECT_ROOT,
    SUGGESTION_CACHE,
    TEACHER_AIC_MODEL_PATH,
    TEACHER_MODEL_PATH,
)
from .maps import Level, Song, flatten_words, load_approved_tracks, load_level, load_song
from .model import NotePolicy
from .notes import build_notes, match_note_sets

SUGGESTION_SCRIPT = PROJECT_ROOT / "tools" / "analyze_vocals.py"
TEACHER_METRICS_PATH = PROJECT_ROOT / "automap" / "teacher_metrics.json"
TEACHER_TARGET_DENSITY = 0.75

TEACHER_FEATURE_NAMES = (
    "start_frac",
    "log_duration",
    "midi_norm",
    "confidence",
    "velocity",
    "rms_mean",
    "onset_strength",
    "voiced_ratio",
    "pitch_std",
    "pitch_slope",
    "midi_dev_local",
    "prev_gap",
    "next_gap",
    "prev_pitch_delta",
    "next_pitch_delta",
    "run_length",
    "overlap_prev",
    "word_dist",
    "inside_word",
    "word_len",
    "line_frac",
    "line_start_dist",
    "line_end_dist",
    "is_word_start",
    "song_duration",
    "is_first",
    "is_last",
    "raw_pitch_start_delta",
    "raw_pitch_q1_delta",
    "raw_pitch_mid_delta",
    "raw_pitch_q3_delta",
    "raw_pitch_end_delta",
    "raw_pitch_range",
    "raw_pitch_jump_before",
    "raw_pitch_jump_after",
    "raw_rms_start",
    "raw_rms_mid",
    "raw_rms_end",
    "raw_flux_start",
    "raw_flux_mid",
    "raw_flux_end",
    "raw_voiced_start",
    "raw_voiced_mid",
    "raw_voiced_end",
    "raw_confidence_start",
    "raw_confidence_mid",
    "raw_confidence_end",
)
TEACHER_FEATURE_COUNT = len(TEACHER_FEATURE_NAMES)


@dataclass
class TeacherRuntime:
    model: NotePolicy
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metrics: dict[str, Any]
    step: int = 0
    pitch_correction: bool = True


def _suggestion_cache_path(song: Song) -> Path:
    try:
        stat = song.guide_path.stat()
        signature = f"{stat.st_size}-{stat.st_mtime_ns}"
    except OSError:
        signature = "missing"
    return SUGGESTION_CACHE / f"{song.id}-{signature}.json"


def load_or_build_suggestions(song: Song, force: bool = False) -> list[dict[str, Any]]:
    """Run the app's own melody extractor and cache its dense note chart."""
    cache = _suggestion_cache_path(song)
    if cache.exists() and not force:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return list(payload.get("notes") or [])
    if not SUGGESTION_SCRIPT.exists():
        return []
    SUGGESTION_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.json")
    try:
        subprocess.run(
            [
                sys.executable,
                str(SUGGESTION_SCRIPT),
                str(song.guide_path),
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        payload = json.loads(temporary.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"warning: melody teacher unavailable for {song.title}: {error}")
        temporary.unlink(missing_ok=True)
        return []
    notes: list[dict[str, Any]] = []
    for note in payload.get("notes") or []:
        try:
            start = float(note["start"])
            duration = float(note.get("dur") or note.get("end", start) - start)
            midi = int(round(float(note["midi"])))
        except Exception:
            continue
        if not np.isfinite(start) or not np.isfinite(duration):
            continue
        if duration < 0.09 or midi < 36 or midi > 92:
            continue
        if start >= song.duration - 0.05:
            continue
        notes.append(
            {
                "start": round(max(0.0, start), 3),
                "dur": round(min(4.0, duration), 3),
                "midi": midi,
                "conf": float(note.get("conf") or 0.5),
                "vel": float(note.get("vel") or 0.5),
                "cents": float(note.get("cents") or 0.0),
            }
        )
    notes.sort(key=lambda item: (item["start"], item["midi"]))
    cache.write_text(
        json.dumps({"notes": notes, "stats": payload.get("stats") or {}}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.unlink(missing_ok=True)
    return notes


def _word_context(song: Song) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    words = flatten_words(song.lines)
    starts = np.array([word.start for word in words], dtype=np.float32)
    ends = np.array([word.end for word in words], dtype=np.float32)
    lines = np.array([word.line for word in words], dtype=np.float32)
    lengths = np.maximum(0.05, ends - starts)
    return starts, ends, lines, lengths


def _teacher_features(
    song: Song,
    audio: dict[str, Any],
    notes: list[dict[str, Any]],
    index: int,
    word_starts: np.ndarray,
    word_ends: np.ndarray,
    word_lines: np.ndarray,
    word_lengths: np.ndarray,
) -> list[float]:
    note = notes[index]
    start = float(note["start"])
    duration = max(0.05, float(note["dur"]))
    midi = float(note["midi"])
    fps = float(audio["fps"])
    midi_track = np.asarray(audio["midi"], dtype=np.float32)
    rms = np.asarray(audio["rms"], dtype=np.float32)
    flux = np.asarray(audio["flux"], dtype=np.float32)
    voiced = np.asarray(audio["voiced"], dtype=bool)
    confidence = np.asarray(audio.get("confidence", np.zeros_like(rms)), dtype=np.float32)

    def sample(track: np.ndarray, time: float) -> float:
        position = int(round(time * fps))
        low = max(0, position - 1)
        high = min(track.size, position + 2)
        if high <= low:
            return 0.0
        window = track[low:high]
        if track is midi_track:
            finite_window = window[np.isfinite(window)]
            return float(np.median(finite_window)) if finite_window.size else midi
        if track is voiced:
            return float(np.mean(window))
        return float(np.mean(window))

    note_end = start + duration
    pitch_start = sample(midi_track, start)
    pitch_q1 = sample(midi_track, start + duration * 0.25)
    pitch_mid = sample(midi_track, start + duration * 0.5)
    pitch_q3 = sample(midi_track, start + duration * 0.75)
    pitch_end = sample(midi_track, note_end)
    pitch_values = [pitch_start, pitch_q1, pitch_mid, pitch_q3, pitch_end]
    raw_pitch_range = max(pitch_values) - min(pitch_values)
    pitch_before = sample(midi_track, start - 0.10)
    pitch_after = sample(midi_track, note_end + 0.10)

    first = max(0, int(np.floor(start * fps)))
    last = min(midi_track.size, int(np.ceil((start + duration) * fps)) + 1)
    pitch = midi_track[first:last]
    finite = pitch[np.isfinite(pitch)]
    local_midi = float(np.median(finite)) if finite.size else midi
    rms_mean = float(np.mean(rms[first:last])) if last > first else 0.0
    onset = float(np.max(flux[max(0, first - 1) : min(flux.size, first + 3)])) if flux.size else 0.0
    voiced_ratio = float(np.mean(voiced[first:last])) if last > first else 0.0
    pitch_std = float(np.nanstd(finite)) if finite.size else 0.0
    if finite.size >= 4:
        quarter = max(1, finite.size // 4)
        slope = (
            float(np.median(finite[-quarter:])) - float(np.median(finite[:quarter]))
        ) / max(0.05, duration)
    else:
        slope = 0.0

    previous = notes[index - 1] if index > 0 else None
    following = notes[index + 1] if index + 1 < len(notes) else None
    prev_gap = start - float(previous["start"] + previous["dur"]) if previous else 2.0
    next_gap = float(following["start"]) - (start + duration) if following else 2.0
    prev_pitch = abs(midi - float(previous["midi"])) if previous else 12.0
    next_pitch = abs(float(following["midi"]) - midi) if following else 12.0
    overlap_prev = (
        max(0.0, float(previous["start"] + previous["dur"]) - start)
        if previous
        else 0.0
    )
    run_length = 1
    cursor = index - 1
    while cursor >= 0:
        prior = notes[cursor]
        gap = notes[cursor + 1]["start"] - (prior["start"] + prior["dur"])
        if gap > 0.15 or abs(notes[cursor + 1]["midi"] - prior["midi"]) > 1:
            break
        run_length += 1
        cursor -= 1

    if word_starts.size:
        distances = np.abs(word_starts - start)
        nearest = int(np.argmin(distances))
        word_dist = float(min(2.0, distances[nearest]))
        inside = float(
            word_starts[nearest] - 0.05 <= start <= word_ends[nearest] + 0.05
        )
        word_len = min(12.0, float(np.maximum(1.0, word_ends[nearest] - word_starts[nearest]) * 2.0)) / 12.0
        line_frac = float(word_lines[nearest] / max(1.0, len(song.lines)))
        line_start_dist = float(min(2.0, max(0.0, start - word_starts[nearest])))
        line_end_dist = float(min(2.0, max(0.0, word_ends[nearest] - start)))
        is_word_start = float(np.min(distances) <= 0.06)
    else:
        word_dist = 2.0
        inside = 0.0
        word_len = 0.0
        line_frac = 0.0
        line_start_dist = 2.0
        line_end_dist = 2.0
        is_word_start = 0.0

    return [
        start / max(1.0, song.duration),
        float(np.log(max(0.05, duration))),
        (midi - 60.0) / 24.0,
        float(note.get("conf") or 0.5),
        float(note.get("vel") or 0.5),
        rms_mean,
        min(4.0, onset),
        voiced_ratio,
        min(4.0, pitch_std / 2.0),
        float(np.clip(slope / 12.0, -4.0, 4.0)),
        float(np.clip((midi - local_midi) / 12.0, -2.0, 2.0)),
        float(np.clip(prev_gap, -1.0, 2.0)),
        float(np.clip(next_gap, -1.0, 2.0)),
        prev_pitch / 12.0,
        next_pitch / 12.0,
        min(5.0, float(run_length)) / 5.0,
        min(1.0, overlap_prev),
        word_dist / 2.0,
        inside,
        word_len,
        line_frac,
        line_start_dist / 2.0,
        line_end_dist / 2.0,
        is_word_start,
        song.duration / 400.0,
        1.0 if index == 0 else 0.0,
        1.0 if index + 1 == len(notes) else 0.0,
        (pitch_start - midi) / 12.0,
        (pitch_q1 - midi) / 12.0,
        (pitch_mid - midi) / 12.0,
        (pitch_q3 - midi) / 12.0,
        (pitch_end - midi) / 12.0,
        raw_pitch_range / 12.0,
        (pitch_before - pitch_start) / 12.0,
        (pitch_after - pitch_end) / 12.0,
        sample(rms, start),
        sample(rms, start + duration * 0.5),
        sample(rms, note_end),
        min(4.0, sample(flux, start)),
        min(4.0, sample(flux, start + duration * 0.5)),
        min(4.0, sample(flux, note_end)),
        sample(voiced, start),
        sample(voiced, start + duration * 0.5),
        sample(voiced, note_end),
        sample(confidence, start),
        sample(confidence, start + duration * 0.5),
        sample(confidence, note_end),
    ]


def build_teacher_candidates(
    song: Song,
    audio: dict[str, Any],
    suggestions: list[dict[str, Any]] | None = None,
) -> CandidateSet:
    notes = suggestions if suggestions is not None else load_or_build_suggestions(song)
    if not notes:
        return CandidateSet(
            features=np.zeros((0, TEACHER_FEATURE_COUNT), dtype=np.float32),
            starts=np.zeros(0, dtype=np.float32),
            durations=np.zeros(0, dtype=np.float32),
            midis=np.zeros(0, dtype=np.float32),
        )
    starts = np.array([float(note["start"]) for note in notes], dtype=np.float32)
    durations = np.array([float(note["dur"]) for note in notes], dtype=np.float32)
    midis = np.array([float(note["midi"]) for note in notes], dtype=np.float32)
    word_starts, word_ends, word_lines, word_lengths = _word_context(song)
    rows = [
        _teacher_features(
            song,
            audio,
            notes,
            index,
            word_starts,
            word_ends,
            word_lines,
            word_lengths,
        )
        for index in range(len(notes))
    ]
    return CandidateSet(
        features=np.asarray(rows, dtype=np.float32),
        starts=starts,
        durations=durations,
        midis=midis,
    )


def _reference_notes(song: Song, audio: dict[str, Any]) -> list[dict[str, Any]] | None:
    for track in load_approved_tracks():
        if track.song_id == song.id:
            return load_level(track.song_id, track.map_relative).notes
    source = karaoke.TRAINING_NOTE_BY_SONG.get(song.id)
    if source is not None:
        public = note_calibration.public_notes(song, audio, source)
        if public:
            return public
    return None


def teacher_training_songs(exclude_song_ids: tuple[str, ...] = ()) -> list[Song]:
    songs: list[Song] = []
    for track in load_approved_tracks():
        songs.append(load_song(track.song_id))
    for source in (*karaoke.SOURCES, *karaoke.NOTE_SOURCES):
        songs.append(load_song(source.song_id))
    seen = set()
    unique: list[Song] = []
    for song in songs:
        if song.id in exclude_song_ids:
            continue
        if song.id not in seen:
            seen.add(song.id)
            unique.append(song)
    return unique


def _label_candidates(
    song: Song,
    audio: dict[str, Any],
    candidates: CandidateSet,
    reference: list[dict[str, Any]],
) -> CandidateSet:
    level = Level(
        notes=reference,
        lines=[],
        placed_words=[],
        custom_words=[],
        ribbon_links=[],
        low=0,
        high=0,
        path=Path("teacher-reference"),
    )
    candidates = label_candidates(
        candidates,
        level,
        note_onset_tolerance=0.12,
        note_pitch_tolerance=2.0,
    )
    # The policy head is intentionally bounded to a small local correction.
    candidates.onset_offsets = np.clip(candidates.onset_offsets, -0.08, 0.08)
    return candidates


def _target_count(song: Song) -> int:
    return max(20, int(round(song.duration * TEACHER_TARGET_DENSITY)))


def load_teacher_anchor_data() -> list[Any]:
    """Return labeled teacher candidates for policy-gradient replay.

    The live grading update expects the same ``TrackData`` shape as the old
    one-song policy. Teacher candidates fit that contract; only the feature
    layout differs.
    """
    from .dataset import TrackData

    anchor_data: list[Any] = []
    for track in load_approved_tracks():
        song = load_song(track.song_id)
        audio = load_or_extract(song)
        suggestions = load_or_build_suggestions(song)
        candidates = build_teacher_candidates(song, audio, suggestions)
        reference = _reference_notes(song, audio)
        if not reference or len(candidates) == 0:
            continue
        candidates = _label_candidates(song, audio, candidates, reference)
        rng = np.random.default_rng(0)
        train_mask = rng.random(len(candidates)) < 0.85
        mean = candidates.features[train_mask].mean(axis=0).astype(np.float32)
        std = candidates.features[train_mask].std(axis=0).astype(np.float32)
        std[std < 1e-5] = 1.0
        anchor_data.append(
            TrackData(
                track=track,
                song=song,
                level=Level(
                    notes=reference,
                    lines=song.lines,
                    placed_words=[],
                    custom_words=[],
                    ribbon_links=[],
                    low=0,
                    high=0,
                    path=Path("teacher-anchor"),
                ),
                candidates=candidates,
                train_mask=train_mask,
                feature_mean=mean,
                feature_std=std,
            )
        )
    return anchor_data


def select_teacher_actions(
    runtime: TeacherRuntime,
    candidates: CandidateSet,
    target_count: int,
) -> dict[str, np.ndarray]:
    normalized = (candidates.features - runtime.feature_mean) / runtime.feature_std
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.T[None, :, :])).float()
    runtime.model.eval()
    with torch.no_grad():
        output = runtime.model(tensor)
        logits = output["select_logit"][0]
        probabilities = torch.sigmoid(logits)
        count = min(target_count, len(probabilities))
        selected = torch.zeros_like(probabilities)
        if count > 0:
            chosen = torch.topk(probabilities, count).indices
            selected[chosen] = 1.0
        return {
            "select_logit": logits.cpu().numpy().astype(np.float32),
            "select_prob": probabilities.cpu().numpy().astype(np.float32),
            "selection_bias": np.array([0.0], dtype=np.float32),
            "selected": selected.cpu().numpy().astype(np.int8),
            # The teacher already has the local vocal pitch. Let the model
            # choose notes and timing, but do not let a USDB-trained pitch
            # correction pull the melody away from the local audio.
            "pitch_offset": (
                output["pitch_offset"][0].cpu().numpy().astype(np.float32)
                if runtime.pitch_correction
                else np.zeros(len(candidates), dtype=np.float32)
            ),
            "onset_offset": np.zeros(len(candidates), dtype=np.float32),
            "log_scale": np.zeros(len(candidates), dtype=np.float32),
        }


def train_teacher_model(
    epochs: int = 300,
    seed: int = 4,
    output_path: Path | None = None,
    metrics_path: Path | None = None,
    exclude_song_ids: tuple[str, ...] = (),
    pitch_correction: bool = True,
) -> dict[str, Any]:
    output_path = output_path or TEACHER_MODEL_PATH
    metrics_path = metrics_path or TEACHER_METRICS_PATH
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    pitch_parts: list[np.ndarray] = []
    onset_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    per_song: dict[str, dict[str, Any]] = {}
    for song in teacher_training_songs(exclude_song_ids):
        audio = load_or_extract(song)
        suggestions = load_or_build_suggestions(song)
        candidates = build_teacher_candidates(song, audio, suggestions)
        reference = _reference_notes(song, audio)
        if not reference or len(candidates) == 0:
            continue
        candidates = _label_candidates(song, audio, candidates, reference)
        features_parts.append(candidates.features)
        label_parts.append(candidates.labels)
        pitch_parts.append(candidates.pitch_offsets)
        onset_parts.append(candidates.onset_offsets)
        scale_parts.append(candidates.log_scales)
        per_song[song.id] = {
            "song": song,
            "audio": audio,
            "suggestions": suggestions,
            "candidates": candidates,
            "reference": reference,
            "positive": int(candidates.labels.sum()),
            "total": len(candidates),
        }
        print(
            f"  {song.title}: {int(candidates.labels.sum())}/{len(candidates)} "
            f"teacher notes kept"
        )

    if not features_parts:
        raise RuntimeError("no teacher training data was found")

    features = np.concatenate(features_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    pitch = np.concatenate(pitch_parts, axis=0)
    onset = np.concatenate(onset_parts, axis=0)
    scale = np.concatenate(scale_parts, axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    normalized = (features - mean) / std
    model = NotePolicy(TEACHER_FEATURE_COUNT, hidden=128, dropout=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = np.where(labels > 0.5)[0]
    negative = np.where(labels <= 0.5)[0]
    if positive.size == 0 or negative.size == 0:
        raise RuntimeError("teacher training needs both kept and rejected notes")

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for _ in range(max(1, len(normalized) // 192)):
            chosen = np.concatenate(
                [
                    rng.choice(positive, size=min(64, positive.size), replace=True),
                    rng.choice(negative, size=min(128, negative.size), replace=False),
                ]
            )
            tensor = torch.from_numpy(normalized[chosen].T[None, :, :]).float()
            output = model(tensor)
            keep = torch.from_numpy(labels[chosen]).float()[None, :]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["select_logit"], keep, pos_weight=torch.tensor(2.0)
            )
            mask = labels[chosen] > 0.5
            if bool(mask.any()):
                mask_tensor = torch.from_numpy(mask)
                loss = (
                    loss
                    + 0.5
                    * torch.nn.functional.smooth_l1_loss(
                        output["pitch_offset"][0][mask_tensor],
                        torch.from_numpy(pitch[chosen][mask]).float(),
                    )
                    + torch.nn.functional.smooth_l1_loss(
                        output["onset_offset"][0][mask_tensor],
                        torch.from_numpy(onset[chosen][mask]).float(),
                    )
                    + 0.5
                    * torch.nn.functional.smooth_l1_loss(
                        output["log_scale"][0][mask_tensor],
                        torch.from_numpy(scale[chosen][mask]).float(),
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch % 50 == 0:
            print(f"teacher epoch {epoch:04d} loss={np.mean(losses):.4f}")

    runtime = TeacherRuntime(
        model=model,
        feature_mean=mean,
        feature_std=std,
        metrics={},
        step=0,
        pitch_correction=pitch_correction,
    )
    metrics: dict[str, Any] = {
        "model": "teacher-note denoiser",
        "target_density": TEACHER_TARGET_DENSITY,
        "features": list(TEACHER_FEATURE_NAMES),
        "songs": {},
    }
    for song_id, item in per_song.items():
        candidates = item["candidates"]
        actions = select_teacher_actions(runtime, candidates, _target_count(item["song"]))
        predicted = build_notes(
            candidates, actions, song_duration=item["song"].duration
        )
        precision, recall, f1 = match_note_sets(
            predicted, item["reference"], onset_tolerance=0.16, pitch_tolerance=1.5
        )
        raw_precision, raw_recall, raw_f1 = match_note_sets(
            item["suggestions"], item["reference"], onset_tolerance=0.16, pitch_tolerance=1.5
        )
        metrics["songs"][song_id] = {
            "title": item["song"].title,
            "teacher_notes": item["total"],
            "kept": item["positive"],
            "generated": len(predicted),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "teacher_f1": raw_f1,
        }
        print(
            f"  {item['song'].title}: generated {len(predicted)} notes, "
            f"F1 {f1:.3f} (teacher {raw_f1:.3f})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "hidden": 128,
            "feature_names": list(TEACHER_FEATURE_NAMES),
            "metrics": metrics,
            "step": 0,
            "pitch_correction": pitch_correction,
        },
        output_path,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {output_path}")
    return metrics


def load_teacher_model(model_path: Path | None = None) -> TeacherRuntime | None:
    model_path = model_path or TEACHER_MODEL_PATH
    if not model_path.exists():
        return None
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        names = payload.get("feature_names")
        if names is not None and list(names) != list(TEACHER_FEATURE_NAMES):
            print("warning: teacher checkpoint uses an older feature layout")
            return None
        if names is None and mean.size != TEACHER_FEATURE_COUNT:
            print("warning: teacher checkpoint has an unknown feature layout")
            return None
        std = np.asarray(payload["feature_std"], dtype=np.float32)
        model = NotePolicy(int(mean.size), hidden=int(payload.get("hidden", 128)))
        model.load_state_dict(payload["model"])
        model.eval()
        return TeacherRuntime(
            model=model,
            feature_mean=mean,
            feature_std=std,
            metrics=payload.get("metrics") or {},
            step=int(payload.get("step", 0)),
            pitch_correction=bool(payload.get("pitch_correction", True)),
        )
    except Exception as error:
        print(f"warning: could not load teacher model: {error}")
        return None


def ensure_word_coverage(
    song: Song,
    notes: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
    audio: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep at least one note on every timed lyric word.

    The teacher is allowed to omit a word when the vocal is quiet or the
    tracker split it oddly. A missing word is more damaging to a karaoke map
    than one extra short note, so fill those gaps from the teacher chart and
    only fall back to the audio pitch when the teacher has nothing nearby.
    """
    if not notes:
        return notes
    fps = float(audio["fps"])
    midi_track = np.asarray(audio["midi"], dtype=np.float32)
    words = flatten_words(song.lines)
    added = list(notes)
    for word in words:
        covered = any(
            float(note["start"]) <= word.start + 0.35
            and float(note["start"]) + float(note["dur"]) > word.start - 0.15
            for note in added
        )
        if covered:
            continue
        nearby = [
            note
            for note in suggestions
            if abs(float(note["start"]) - word.start) <= 0.6
            and not any(
                abs(float(existing["start"]) - float(note["start"])) < 0.02
                for existing in added
            )
        ]
        if nearby:
            chosen = min(nearby, key=lambda item: abs(float(item["start"]) - word.start))
            start = max(0.0, float(chosen["start"]))
            duration = max(0.12, min(4.0, float(chosen["dur"])))
            midi = int(chosen["midi"])
        else:
            first = max(0, int(word.start * fps))
            last = min(midi_track.size, int(max(word.start + 0.15, word.end) * fps) + 1)
            values = midi_track[first:last]
            finite = values[np.isfinite(values)]
            midi = int(round(float(np.median(finite)))) if finite.size else 60
            start = max(0.0, word.start)
            duration = max(0.12, min(4.0, word.end - word.start))
        added.append(
            {
                "start": round(start, 3),
                "dur": round(duration, 3),
                "midi": int(np.clip(midi, 24, 96)),
                "vel": 0.8,
                "word": None,
                "wordIndex": None,
                "line": None,
                "ribbon": None,
            }
        )
    added.sort(key=lambda note: (float(note["start"]), int(note["midi"])))
    # Keep the added notes from overlapping the selected melody notes.
    cleaned: list[dict[str, Any]] = []
    for note in added:
        if cleaned:
            previous = cleaned[-1]
            previous_end = float(previous["start"]) + float(previous["dur"])
            if float(note["start"]) < previous_end - 0.02:
                trimmed = max(0.08, float(note["start"]) - float(previous["start"]) - 0.02)
                if trimmed >= 0.08:
                    previous["dur"] = round(trimmed, 3)
                else:
                    continue
        cleaned.append(note)
    return cleaned


def teacher_model_path_for_song(song: Song) -> Path:
    """Use the AIC-specialized denoiser for Alice in Chains, broad USDB for others."""
    if "aliceinchains" in karaoke.normalize(song.artist) and TEACHER_AIC_MODEL_PATH.exists():
        return TEACHER_AIC_MODEL_PATH
    return TEACHER_MODEL_PATH


def generate_teacher_notes(
    song: Song,
    audio: dict[str, Any],
    seed: int | None = None,
    model_path: Path | None = None,
) -> tuple[list[dict[str, Any]], CandidateSet, dict[str, np.ndarray], TeacherRuntime] | None:
    runtime = load_teacher_model(model_path or teacher_model_path_for_song(song))
    if runtime is None:
        return None
    suggestions = load_or_build_suggestions(song)
    if not suggestions:
        return None
    candidates = build_teacher_candidates(song, audio, suggestions)
    if len(candidates) == 0:
        return None
    actions = select_teacher_actions(runtime, candidates, _target_count(song))
    notes = build_notes(candidates, actions, song_duration=song.duration)
    if not notes:
        return None
    notes = ensure_word_coverage(song, notes, suggestions, audio)
    # Some songs contain long melismatic phrases that the denoiser collapses
    # too aggressively. Keep the dense teacher chart when the selected track is
    # much sparser than the song's vocal activity.
    if len(notes) < song.duration * 0.80 and len(suggestions) > len(notes) * 1.5:
        notes = [
            {
                "start": round(float(note["start"]), 3),
                "dur": round(float(note["dur"]), 3),
                "midi": int(note["midi"]),
                "vel": 0.8,
                "word": None,
                "wordIndex": None,
                "line": None,
                "ribbon": None,
            }
            for note in suggestions
        ]
    return notes, candidates, actions, runtime

