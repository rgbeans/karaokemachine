"""Use public UltraStar note tracks as calibrated note anchors.

The karaoke timing files include note pitch and duration as well as text. For
the four songs with public timing data, we can rebuild a clean monophonic note
track instead of relying on the one-song neural policy. Pitches are placed in
the octave nearest the local vocal track.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from . import karaoke
from .maps import Song


def _pitch_median(audio: dict[str, Any], start: float, end: float) -> float | None:
    fps = float(audio["fps"])
    midi = np.asarray(audio["midi"], dtype=np.float32)
    low = max(0, int(round(start * fps)))
    high = min(midi.size, int(round(end * fps)) + 1)
    if high <= low:
        return None
    values = midi[low:high]
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def public_notes(song: Song, audio: dict[str, Any]) -> list[dict[str, Any]]:
    source = karaoke.BY_SONG.get(song.id)
    if source is None:
        return []
    try:
        timing = karaoke.fetch_timing(source)
    except Exception as error:
        print(f"warning: public note timings unavailable for {song.title}: {error}")
        return []
    bpm = float(timing["bpm"])
    beat_seconds = 60.0 / (bpm * 4.0)
    raw: list[dict[str, Any]] = []
    notes = list(timing["notes"])
    index = 0
    while index < len(notes):
        note = notes[index]
        text = str(note.get("text") or "").strip()
        if not text or text == "~":
            index += 1
            continue
        start = float(note["time"]) + source.effective_note_offset
        duration = float(note["length"]) * beat_seconds
        pitch_class = int(note["pitch"]) % 12
        # UltraStar uses tilde notes for a held syllable.
        cursor = index + 1
        while cursor < len(notes) and str(notes[cursor].get("text") or "").strip() == "~":
            duration += float(notes[cursor]["length"]) * beat_seconds
            cursor += 1
        raw.append(
            {
                "start": start,
                "duration": duration,
                "pitch_class": pitch_class,
                "text": text,
            }
        )
        index = cursor

    if not raw:
        return []

    # Find the octave shift that best matches the local vocal pitch.
    shifts: list[int] = []
    for item in raw:
        local = _pitch_median(
            audio,
            item["start"],
            item["start"] + min(1.0, item["duration"]),
        )
        if local is not None:
            shifts.append(
                int(round((local - item["pitch_class"]) / 12.0))
            )
    default_shift = int(statistics.median(shifts)) if shifts else 4

    result: list[dict[str, Any]] = []
    for item in raw:
        start = max(0.0, min(song.duration - 0.05, item["start"]))
        duration = max(
            0.08,
            min(4.0, item["duration"], max(0.08, song.duration - start)),
        )
        local = _pitch_median(audio, start, start + min(1.0, duration))
        shift = default_shift
        if local is not None:
            shift = int(round((local - item["pitch_class"]) / 12.0))
        midi = int(np.clip(item["pitch_class"] + 12 * shift, 24, 96))
        result.append(
            {
                "start": round(start, 3),
                "dur": round(duration, 3),
                "midi": midi,
                "vel": 0.8,
                "word": None,
                "wordIndex": None,
                "line": None,
                "ribbon": None,
            }
        )

    result.sort(key=lambda note: (note["start"], note["midi"]))
    # Keep the track monophonic if USDB has overlapping held notes.
    cleaned: list[dict[str, Any]] = []
    for note in result:
        if cleaned:
            previous = cleaned[-1]
            previous_end = previous["start"] + previous["dur"]
            if note["start"] < previous_end - 0.02:
                trimmed = max(0.08, note["start"] - previous["start"] - 0.02)
                if trimmed >= 0.08:
                    previous["dur"] = round(trimmed, 3)
                else:
                    continue
        cleaned.append(note)
    return cleaned


def alignment_for_song(song_id: str) -> dict[str, Any] | None:
    source = karaoke.BY_SONG.get(song_id)
    if source is None:
        return None
    return {
        "source": "USDB public UltraStar note timings",
        "usdb_id": source.usdb_id,
        "offset": source.effective_note_offset,
        "source_url": source.source_url,
        "notes_url": source.notes_url,
    }
