"""Convert scored proposals into KaraokeMachine notes and map documents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .candidates import CandidateSet
from .maps import Song, WordTiming, flatten_words


def build_notes(
    candidates: CandidateSet,
    actions: dict[str, np.ndarray],
    song_duration: float | None = None,
    min_duration: float = 0.08,
    max_duration: float = 4.0,
) -> list[dict[str, Any]]:
    selected = np.where(actions["selected"] > 0)[0]
    if selected.size == 0:
        return []
    starts = candidates.starts[selected] + actions["onset_offset"][selected]
    durations = candidates.durations[selected] * np.exp(actions["log_scale"][selected])
    midis = candidates.midis[selected] + actions["pitch_offset"][selected]
    probabilities = actions["select_prob"][selected]

    starts = np.maximum(0.0, starts)
    durations = np.clip(durations, min_duration, max_duration)
    midis = np.clip(np.round(midis), 24, 96)

    order = np.argsort(-probabilities)
    accepted: list[dict[str, Any]] = []
    for index in order:
        start = float(starts[index])
        if song_duration is not None:
            start = min(start, max(0.0, song_duration - min_duration))
        duration = float(durations[index])
        if song_duration is not None:
            duration = min(duration, max(min_duration, song_duration - start))
        end = start + duration
        if any(min(end, note["start"] + note["dur"]) - max(start, note["start"]) > 0.055 for note in accepted):
            continue
        accepted.append(
            {
                "start": round(start, 3),
                "dur": round(duration, 3),
                "midi": int(midis[index]),
                "vel": 0.8,
                "word": None,
                "wordIndex": None,
                "line": None,
                "ribbon": None,
                "_score": float(probabilities[index]),
            }
        )
    accepted.sort(key=lambda note: (note["start"], note["midi"]))
    # Avoid pathological density: keep the strongest few notes per second.
    limited: list[dict[str, Any]] = []
    for note in accepted:
        recent = [other for other in limited if note["start"] - other["start"] < 1.0]
        if len(recent) >= 6:
            continue
        limited.append(note)
    for note in limited:
        note.pop("_score", None)
    return limited


def match_note_sets(
    predicted: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    onset_tolerance: float = 0.16,
    pitch_tolerance: float = 1.5,
) -> tuple[float, float, float]:
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for truth_index, target in enumerate(truth):
            onset = abs(float(pred["start"]) - float(target.get("start", 0.0)))
            pitch = abs(float(pred["midi"]) - float(target.get("midi", 0.0)))
            if onset <= onset_tolerance and pitch <= pitch_tolerance:
                pairs.append((onset / onset_tolerance + pitch / pitch_tolerance, pred_index, truth_index))
    pairs.sort()
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    for _, pred_index, truth_index in pairs:
        if pred_index in used_pred or truth_index in used_truth:
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
    true_positive = len(used_pred)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def place_words(
    song: Song,
    notes: list[dict[str, Any]],
    audio: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not notes:
        return []
    words: list[WordTiming] = flatten_words(song.lines)
    if not words:
        return []
    midis = [note["midi"] for note in notes]
    low, high = min(midis), max(midis)
    span = max(12, high - low)
    center = (low + high) / 2.0
    runtime = None
    karaoke_source = None
    if audio is not None:
        from . import karaoke, word_model

        karaoke_source = karaoke.BY_SONG.get(song.id)
        if karaoke_source is not None:
            runtime = word_model.load_word_model()
    placed: list[dict[str, Any]] = []
    for word in words:
        line = song.lines[word.line] if word.line < len(song.lines) else {}
        line_words = list(line.get("words") or [])
        word_count = len(line_words) or sum(1 for item in words if item.line == word.line)
        word_dict = (
            line_words[word.word_index]
            if word.word_index < len(line_words)
            else {"text": word.text, "start": word.start, "end": word.end}
        )
        start = float(word.start)
        if runtime is not None and karaoke_source is not None and audio is not None:
            start = word_model.predict_word_time(
                runtime,
                song,
                audio,
                karaoke_source,
                line,
                word.line,
                word_dict,
                word.word_index,
                word_count,
            )
        candidates = [
            note
            for note in notes
            if note["start"] <= start + 0.35
            and note["start"] + note["dur"] >= start - 0.35
        ]
        if candidates:
            note = max(
                candidates,
                key=lambda item: min(
                    float(item["start"]) + float(item["dur"]), start + 0.35
                )
                - max(float(item["start"]), start - 0.35),
            )
            y = 0.5 - (float(note["midi"]) - center) / span * 0.42
        else:
            y = 0.5
        placed.append(
            {
                "text": word.text,
                "start": round(start, 3),
                "y": round(float(np.clip(y, 0.12, 0.88)), 4),
                "line": word.line,
                "wordIndex": word.word_index,
            }
        )
    return placed


def build_ribbon_links(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for index in range(len(notes) - 1):
        current = notes[index]
        following = notes[index + 1]
        gap = float(following["start"]) - (float(current["start"]) + float(current["dur"]))
        if -0.02 <= gap <= 0.12 and abs(int(following["midi"]) - int(current["midi"])) <= 12:
            links.append(
                {
                    "from": index,
                    "to": index + 1,
                    "fromSide": "right",
                    "toSide": "left",
                }
            )
    return links


def build_map_document(
    song: Song,
    notes: list[dict[str, Any]],
    audio: dict[str, Any] | None = None,
    word_alignment: str = "lyrics",
    note_alignment: str = "policy",
) -> dict[str, Any]:
    midis = [int(note["midi"]) for note in notes]
    low = max(0, min(midis) - 2) if midis else 48
    high = min(127, max(midis) + 2) if midis else 72
    return {
        "format": "karaokemachine-level",
        "version": 1,
        "draft": True,
        "notes": notes,
        "lines": song.lines,
        "placedWords": place_words(song, notes, audio),
        "customWords": [],
        "ribbonLinks": build_ribbon_links(notes),
        "wordAlignment": word_alignment,
        "noteAlignment": note_alignment,
        "low": low,
        "high": high,
    }
