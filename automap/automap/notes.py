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

        karaoke_source = karaoke.BY_SONG.get(song.id) or karaoke.LINE_BY_SONG.get(song.id)
        if karaoke_source is not None:
            runtime = word_model.load_word_model()
    predicted_starts: dict[tuple[int, int], float] = {}
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
        predicted_starts[(word.line, word.word_index)] = start

    # A line-level source only knows where each line starts. Apply one
    # predicted shift per line rather than letting individual word features
    # introduce jitter inside the line.
    line_shifts: dict[int, float] = {}
    if karaoke_source is not None and karaoke_source.level == "line":
        grouped: dict[int, list[float]] = {}
        for word in words:
            start = predicted_starts[(word.line, word.word_index)]
            grouped.setdefault(word.line, []).append(start - float(word.start))
        line_shifts = {
            line_index: float(np.mean(shifts))
            for line_index, shifts in grouped.items()
        }

    placed: list[dict[str, Any]] = []
    for word in words:
        start = predicted_starts[(word.line, word.word_index)]
        if line_shifts:
            start = float(word.start) + line_shifts.get(word.line, 0.0)
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


def _phrase_groups(notes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a vocal line only where the singer actually breathes/pauses."""
    if not notes:
        return []
    groups: list[list[dict[str, Any]]] = [[notes[0]]]
    for note in notes[1:]:
        previous = groups[-1][-1]
        gap = float(note["start"]) - (float(previous["start"]) + float(previous["dur"]))
        if gap > 0.35:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def _compress_group(
    group: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Turn a long held phrase into a small ribbon run of equal-timed notes."""
    if not group or count <= 0:
        return []
    if len(group) <= count:
        return [dict(note) for note in group]
    start = min(float(note["start"]) for note in group)
    end = max(float(note["start"]) + float(note["dur"]) for note in group)
    span = max(0.1, end - start)
    result: list[dict[str, Any]] = []
    for index in range(count):
        segment_start = start + span * index / count
        segment_end = start + span * (index + 1) / count
        pitches = [
            int(note["midi"])
            for note in group
            if segment_start - 1e-6 <= float(note["start"]) < segment_end
        ]
        if not pitches:
            pitches = [int(group[min(len(group) - 1, index * len(group) // count)]["midi"])]
        midi = int(round(float(np.median(pitches))))
        result.append(
            {
                "start": round(segment_start, 3),
                "dur": round(max(0.12, segment_end - segment_start), 3),
                "midi": midi,
                "vel": 0.8,
                "word": None,
                "wordIndex": None,
                "line": None,
                "ribbon": None,
            }
        )
    return result


def _distribute_by_weights(
    notes: list[dict[str, Any]],
    words: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not words:
        return []
    if not notes:
        return [[] for _ in words]
    ordered = sorted(notes, key=lambda item: (item["start"], item["midi"]))
    if len(ordered) < len(words):
        result: list[list[dict[str, Any]]] = []
        for index in range(len(words)):
            position = round(index * (len(ordered) - 1) / max(1, len(words) - 1))
            result.append([ordered[int(position)]])
        return result
    weights = [max(2, len(str(word.get("text") or "").strip()) + 1) for word in words]
    total = sum(weights)
    counts = [max(1, int(round(len(ordered) * weight / total))) for weight in weights]
    while sum(counts) > len(ordered):
        index = max(range(len(counts)), key=lambda item: counts[item])
        counts[index] -= 1
    while sum(counts) < len(ordered):
        counts[-1] += 1
    result = []
    cursor = 0
    for count in counts:
        result.append(ordered[cursor : cursor + count])
        cursor += count
    return result


def assign_word_runs(
    song: Song,
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign notes to words by actual sung phrases, not equal word spacing.

    A long first phrase with a short first word is treated as a held word and
    compressed into a four-note ribbon run. The remaining words are distributed
    across the later phrase by text length, so they start where the second
    phrase actually starts.
    """
    if not notes:
        return place_words(song, notes)
    midis = [int(note["midi"]) for note in notes]
    low, high = min(midis), max(midis)
    span = max(12, high - low)
    center = (low + high) / 2.0
    starts = [float(line.get("start", 0.0)) for line in song.lines]
    line_notes: list[list[dict[str, Any]]] = [[] for _ in song.lines]
    if starts:
        for note in notes:
            note_start = float(note["start"])
            position = int(np.searchsorted(starts, note_start, side="right") - 1)
            if position < 0:
                continue
            line_end = float(song.lines[position].get("end", starts[position]))
            if note_start > line_end + 0.5:
                continue
            line_notes[position].append(note)

    placed: list[dict[str, Any]] = []
    for line_index, line in enumerate(song.lines):
        words = list(line.get("words") or [])
        if not words:
            continue
        candidates = sorted(
            line_notes[line_index], key=lambda item: (item["start"], item["midi"])
        )
        if not candidates:
            for word_index, word in enumerate(words):
                placed.append(
                    {
                        "text": str(word.get("text") or ""),
                        "start": round(float(word.get("start", line.get("start", 0.0))), 3),
                        "y": 0.5,
                        "line": line_index,
                        "wordIndex": word_index,
                    }
                )
            continue

        groups = _phrase_groups(candidates)
        assignments: list[list[dict[str, Any]]] = [[] for _ in words]
        first_word = str(words[0].get("text") or "").strip().lower()
        first_group = groups[0] if groups else []
        first_span = (
            max(float(note["start"]) + float(note["dur"]) for note in first_group)
            - min(float(note["start"]) for note in first_group)
            if first_group
            else 0.0
        )
        held_first = (
            len(words) >= 2
            and len(first_word.replace("'", "")) <= 3
            and first_span >= 1.5
            and len(first_group) >= 4
        )
        if held_first:
            compressed = _compress_group(first_group, 4)
            assignments[0] = compressed
            first_ids = {id(note) for note in first_group}
            notes[:] = [note for note in notes if id(note) not in first_ids]
            notes.extend(compressed)
            notes.sort(key=lambda note: (float(note["start"]), int(note["midi"])))
            remaining = [note for group in groups[1:] for note in group]
            if remaining:
                assignments[1:] = _distribute_by_weights(remaining, words[1:])
        elif len(groups) >= len(words):
            if len(words) == 1:
                assignments[0] = candidates
            else:
                assignments[0] = groups[0]
                assignments[-1] = groups[-1]
                middle_groups = groups[1:-1]
                middle_words = words[1:-1]
                if middle_words and middle_groups:
                    distributed = _distribute_by_weights(
                        [note for group in middle_groups for note in group],
                        middle_words,
                    )
                    for index, group_notes in enumerate(distributed):
                        assignments[1 + index] = group_notes
        else:
            assignments = _distribute_by_weights(candidates, words)

        if any(not assignment for assignment in assignments):
            assignments = _distribute_by_weights(candidates, words)

        for word_index, word in enumerate(words):
            assigned = assignments[word_index]
            if assigned:
                first_note = min(assigned, key=lambda item: float(item["start"]))
                start = float(first_note["start"])
                y = 0.5 - (float(first_note["midi"]) - center) / span * 0.42
                for note in assigned:
                    note["word"] = str(word.get("text") or "")
                    note["wordIndex"] = word_index
                    note["line"] = line_index
            else:
                start = float(word.get("start", line.get("start", 0.0)))
                y = 0.5
            placed.append(
                {
                    "text": str(word.get("text") or ""),
                    "start": round(start, 3),
                    "y": round(float(np.clip(y, 0.12, 0.88)), 4),
                    "line": line_index,
                    "wordIndex": word_index,
                }
            )
    return placed


def place_words_from_notes(
    song: Song,
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach lyric words to the generated note track.

    Line-level word calibration is a useful prior, but once a real note track
    exists the words should sit on the notes that carry them. Notes are assigned
    to lyric lines by midpoint boundaries, then words are distributed across
    the notes of each line. Lines with no notes fall back to their local word
    timings so the lyrics never disappear.
    """
    if not notes:
        return place_words(song, notes)
    midis = [int(note["midi"]) for note in notes]
    low, high = min(midis), max(midis)
    span = max(12, high - low)
    center = (low + high) / 2.0
    starts = [float(line.get("start", 0.0)) for line in song.lines]
    line_notes: list[list[dict[str, Any]]] = [[] for _ in song.lines]
    if starts:
        for note in notes:
            note_start = float(note["start"])
            position = int(np.searchsorted(starts, note_start, side="right") - 1)
            # Notes before the first lyric line are intro/vocalisation material
            # and must not grab the first line's words.
            if position < 0:
                continue
            line_end = float(song.lines[position].get("end", starts[position]))
            # Notes in a long instrumental gap belong to no lyric line.
            if note_start > line_end + 0.5:
                continue
            line_notes[position].append(note)

    placed: list[dict[str, Any]] = []
    for line_index, line in enumerate(song.lines):
        words = list(line.get("words") or [])
        if not words:
            continue
        candidates = sorted(
            line_notes[line_index], key=lambda item: (item["start"], item["midi"])
        )
        for word_index, word in enumerate(words):
            local_time = float(word.get("start", line.get("start", 0.0)))
            if not candidates:
                placed.append(
                    {
                        "text": str(word.get("text") or ""),
                        "start": round(local_time, 3),
                        "y": 0.5,
                        "line": line_index,
                        "wordIndex": word_index,
                    }
                )
                continue
            if len(candidates) == 1:
                note = candidates[0]
            elif len(candidates) >= len(words):
                position = round(word_index * (len(candidates) - 1) / max(1, len(words) - 1))
                note = candidates[int(position)]
            else:
                position = int(word_index * len(candidates) / max(1, len(words)))
                note = candidates[min(len(candidates) - 1, position)]
            y = 0.5 - (float(note["midi"]) - center) / span * 0.42
            placed.append(
                {
                    "text": str(word.get("text") or ""),
                    "start": round(float(note["start"]), 3),
                    "y": round(float(np.clip(y, 0.12, 0.88)), 4),
                    "line": line_index,
                    "wordIndex": word_index,
                }
            )
    return placed


def build_ribbon_links(
    notes: list[dict[str, Any]],
    song: Song | None = None,
) -> list[dict[str, Any]]:
    """Connect adjacent notes that belong to the same sung run.

    A learned ribbon policy is used when a song context is available. It was
    trained on the explicit connectors in the hand-mapped Man in the Box chart.
    The gap/pitch rule remains as a fallback for tests and old callers.
    """
    from .config import RIBBON_MODEL_PATH
    from .ribbon_model import predict_ribbon_links

    if song is not None and RIBBON_MODEL_PATH.exists():
        links = predict_ribbon_links(song, notes)
    else:
        links = []
        for index in range(len(notes) - 1):
            current = notes[index]
            following = notes[index + 1]
            gap = float(following["start"]) - (float(current["start"]) + float(current["dur"]))
            if gap <= 0.40 and abs(int(following["midi"]) - int(current["midi"])) <= 12:
                links.append(
                    {
                        "from": index,
                        "to": index + 1,
                        "fromSide": "right",
                        "toSide": "left",
                    }
                )
    if any(note.get("word") is not None for note in notes):
        # Never bridge two lyric words with one ribbon. Word runs are allowed
        # to be long; separate words must remain visually separate.
        links = [
            link
            for link in links
            if notes[link["from"]].get("word") is not None
            and notes[link["to"]].get("word") is not None
            and notes[link["from"]].get("word") == notes[link["to"]].get("word")
            and notes[link["from"]].get("line") == notes[link["to"]].get("line")
            and notes[link["from"]].get("wordIndex") == notes[link["to"]].get("wordIndex")
        ]
    links.sort(key=lambda link: (link["from"], link["to"]))
    return links


def build_map_document(
    song: Song,
    notes: list[dict[str, Any]],
    audio: dict[str, Any] | None = None,
    word_alignment: str = "lyrics",
    note_alignment: str = "policy",
    note_source: dict[str, Any] | None = None,
    words_from_notes: bool = False,
) -> dict[str, Any]:
    midis = [int(note["midi"]) for note in notes]
    low = max(0, min(midis) - 2) if midis else 48
    high = min(127, max(midis) + 2) if midis else 72
    placed_words = (
        assign_word_runs(song, notes)
        if words_from_notes
        else place_words(song, notes, audio)
    )
    document: dict[str, Any] = {
        "format": "karaokemachine-level",
        "version": 1,
        "draft": True,
        "notes": notes,
        "lines": song.lines,
        "placedWords": placed_words,
        "customWords": [],
        "ribbonLinks": build_ribbon_links(notes, song),
        "wordAlignment": word_alignment,
        "noteAlignment": note_alignment,
        "low": low,
        "high": high,
    }
    if note_source is not None:
        document["noteSource"] = note_source
    return document
