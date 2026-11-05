"""Turn frame-level vocal analysis into note-onset candidates and labels.

Each candidate is one plausible note start. The neural policy decides which
onsets to keep and predicts that note's pitch correction and duration. This is
much easier to learn from one approved map than choosing among hundreds of
overlapping duration hypotheses.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    CACHE_DIR,
    CANDIDATE_FPS,
    FEATURE_COUNT,
    FEATURE_NAMES,
    FPS,
)
from .maps import Level, Song, WordTiming, flatten_words

CANDIDATE_CACHE_VERSION = 3
_SIGNATURE = hashlib.sha1(
    (str(CANDIDATE_FPS) + "|" + "|".join(FEATURE_NAMES)).encode("utf-8")
).hexdigest()[:12]


def candidate_cache_path(song: Song) -> Path:
    try:
        stat = song.guide_path.stat()
        audio_signature = hashlib.sha1(
            f"{stat.st_size}-{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:8]
    except OSError:
        audio_signature = "missing"
    return (
        CACHE_DIR
        / f"{song.id}-candidates-v{CANDIDATE_CACHE_VERSION}-{_SIGNATURE}-{audio_signature}.npz"
    )


@dataclass
class CandidateSet:
    features: np.ndarray
    starts: np.ndarray
    durations: np.ndarray
    midis: np.ndarray
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pitch_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    onset_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    log_scales: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    matched_notes: int = 0
    total_notes: int = 0
    missing_notes: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.features.shape[0])


def _safe_nanmedian(values: np.ndarray, default: float = 0.0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmedian(values)
    return float(default if not np.isfinite(result) else result)


def _safe_nanmean(values: np.ndarray, default: float = 0.0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmean(values)
    return float(default if not np.isfinite(result) else result)


def _safe_nanstd(values: np.ndarray, default: float = 0.0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanstd(values)
    return float(default if not np.isfinite(result) else result)


def _rolling_nanmedian(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return values.copy()
    pad = size // 2
    padded = np.pad(values, (pad, pad), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def _onset_peaks(flux: np.ndarray, fps: float) -> np.ndarray:
    if flux.size < 3:
        return np.zeros(0, dtype=np.float32)
    smooth = np.convolve(flux, np.array([0.25, 0.5, 0.25]), mode="same")
    positive = smooth[smooth > 0]
    threshold = max(0.5, float(np.percentile(positive, 68)) if positive.size else 0.5)
    local = [
        index
        for index in range(1, smooth.size - 1)
        if smooth[index] >= threshold
        and smooth[index] >= smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
    ]
    min_gap = max(1, int(0.07 * fps))
    selected: list[int] = []
    for index in local:
        if selected and index - selected[-1] < min_gap:
            if smooth[index] > smooth[selected[-1]]:
                selected[-1] = index
        else:
            selected.append(index)
    return np.array([index / fps for index in selected], dtype=np.float32)


def _nearest_distances(times: np.ndarray, targets: np.ndarray, cap: float) -> np.ndarray:
    if targets.size == 0:
        return np.full(times.shape, cap, dtype=np.float32)
    positions = np.searchsorted(targets, times)
    left = np.clip(positions - 1, 0, targets.size - 1)
    right = np.clip(positions, 0, targets.size - 1)
    distance = np.minimum(np.abs(times - targets[left]), np.abs(times - targets[right]))
    return np.minimum(distance, cap).astype(np.float32)


def _next_distance(times: np.ndarray, targets: np.ndarray, cap: float) -> np.ndarray:
    if targets.size == 0:
        return np.full(times.shape, cap, dtype=np.float32)
    positions = np.searchsorted(targets, times, side="left")
    result = np.full(times.shape, cap, dtype=np.float32)
    valid = positions < targets.size
    result[valid] = np.minimum(targets[positions[valid]] - times[valid], cap)
    result[result < 0] = 0.0
    return result


def _fill_gaps(voiced: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    count = voiced.size
    previous = np.full(count, 10.0, dtype=np.float32)
    last = -np.inf
    for index in range(count):
        if voiced[index]:
            last = index
        if np.isfinite(last):
            previous[index] = min(10.0, (index - last) / fps)
    following = np.full(count, 10.0, dtype=np.float32)
    nxt = np.inf
    for index in range(count - 1, -1, -1):
        if voiced[index]:
            nxt = index
        if np.isfinite(nxt):
            following[index] = min(10.0, (nxt - index) / fps)
    return previous, following


def _word_features(words: list[WordTiming], times: np.ndarray) -> tuple[np.ndarray, ...]:
    count = times.size
    word_dist = np.zeros(count, dtype=np.float32)
    inside = np.zeros(count, dtype=np.float32)
    word_len = np.zeros(count, dtype=np.float32)
    word_duration = np.zeros(count, dtype=np.float32)
    is_start = np.zeros(count, dtype=np.float32)
    line_frac = np.zeros(count, dtype=np.float32)
    line_pos = np.zeros(count, dtype=np.float32)
    line_start_dist = np.full(count, 2.0, dtype=np.float32)
    line_end_dist = np.full(count, 2.0, dtype=np.float32)
    if not words:
        return (
            word_dist,
            inside,
            word_len,
            word_duration,
            is_start,
            line_frac,
            line_pos,
            line_start_dist,
            line_end_dist,
        )
    starts = np.array([word.start for word in words], dtype=np.float32)
    ends = np.array([word.end for word in words], dtype=np.float32)
    line_ids = np.array([word.line for word in words], dtype=np.float32)
    max_line = max(1.0, float(line_ids.max()))
    lengths = np.maximum(0.05, ends - starts)
    char_lengths = np.array([max(1, len(word.text)) for word in words], dtype=np.float32)

    for index, time in enumerate(times):
        position = int(np.searchsorted(starts, time, side="right") - 1)
        chosen: int | None = None
        if position >= 0 and time <= ends[position] + 1e-6:
            chosen = position
            inside[index] = 1.0
        else:
            left = position if position >= 0 else None
            right = position + 1 if position + 1 < len(words) else None
            left_dist = abs(time - ends[left]) if left is not None else np.inf
            right_dist = abs(starts[right] - time) if right is not None else np.inf
            if right is not None and right_dist <= left_dist:
                chosen = right
                word_dist[index] = min(1.0, (starts[right] - time))
            elif left is not None:
                chosen = left
                word_dist[index] = max(-1.0, -(time - ends[left]))
        if chosen is not None:
            word_len[index] = min(12.0, char_lengths[chosen]) / 12.0
            word_duration[index] = min(2.0, lengths[chosen]) / 2.0
            line_frac[index] = line_ids[chosen] / max_line
            line_start_dist[index] = min(2.0, max(0.0, time - starts[chosen])) / 2.0
            line_end_dist[index] = min(2.0, max(0.0, ends[chosen] - time)) / 2.0
            line_pos[index] = min(
                1.0, max(0.0, (time - starts[chosen]) / max(0.1, lengths[chosen]))
            )
        nearest = int(np.argmin(np.abs(starts - time)))
        is_start[index] = 1.0 if abs(float(starts[nearest]) - time) <= 0.06 else 0.0
    return (
        word_dist,
        inside,
        word_len,
        word_duration,
        is_start,
        line_frac,
        line_pos,
        line_start_dist,
        line_end_dist,
    )


def _pitch_at(midi: np.ndarray, confidence: np.ndarray, start: float, end: float, fps: float) -> tuple[float, float]:
    first = int(np.floor(start * fps))
    last = min(midi.size, int(np.ceil(end * fps)) + 1)
    window = midi[max(0, first) : max(0, last)]
    confidence_window = confidence[max(0, first) : max(0, last)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pitch = np.nanmedian(window) if window.size else np.nan
    if np.isfinite(pitch):
        return float(pitch), _safe_nanmean(confidence_window)
    # Fall back to the nearest voiced frame within 120 ms.
    center = int(round(start * fps))
    for radius in range(1, int(0.12 * fps) + 1):
        for candidate in (center - radius, center + radius):
            if 0 <= candidate < midi.size and np.isfinite(midi[candidate]):
                return float(midi[candidate]), float(confidence[candidate])
    return float("nan"), 0.0


def _merge_filter_xy(
    values: np.ndarray,
    scores: np.ndarray,
    min_distance: float,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    order = np.argsort(values)
    values = values[order]
    scores = scores[order]
    kept: list[float] = []
    for index, value in enumerate(values):
        if kept and abs(float(value) - kept[-1]) < min_distance:
            # Keep the point with the stronger local evidence.
            if scores[index] > scores[np.searchsorted(values, kept[-1])]:
                kept[-1] = float(value)
            continue
        kept.append(float(value))
    return np.array(kept, dtype=np.float32)


def build_candidates(song: Song, features: dict[str, np.ndarray | float]) -> CandidateSet:
    fps = float(features["fps"])
    midi_track = np.asarray(features["midi"], dtype=np.float32)
    confidence = np.asarray(features["confidence"], dtype=np.float32)
    rms = np.asarray(features["rms"], dtype=np.float32)
    flux = np.asarray(features["flux"], dtype=np.float32)
    voiced = np.asarray(features["voiced"], dtype=bool)
    duration = float(features["duration"])
    if midi_track.size == 0:
        return CandidateSet(
            np.zeros((0, FEATURE_COUNT), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    grid = np.arange(0.0, max(0.0, duration - 0.06), 1.0 / CANDIDATE_FPS, dtype=np.float32)
    grid_frames = np.clip(np.round(grid * fps).astype(int), 0, midi_track.size - 1)
    voiced_grid = grid[voiced[grid_frames] | (confidence[grid_frames] > 0.10)]
    onsets = _onset_peaks(flux, fps)
    voiced_starts = np.where(np.logical_and(voiced[1:], ~voiced[:-1]))[0] / fps
    words = flatten_words(song.lines)
    word_starts = np.array([word.start for word in words], dtype=np.float32)
    onset_scores = np.interp(
        onsets,
        np.arange(flux.size, dtype=np.float32) / fps,
        flux,
        left=0.0,
        right=0.0,
    ) if onsets.size else np.zeros(0, dtype=np.float32)
    starts = np.concatenate(
        [
            voiced_grid.astype(np.float32),
            _merge_filter_xy(onsets, onset_scores, 0.06),
            voiced_starts.astype(np.float32),
            word_starts.astype(np.float32),
        ]
    )
    if starts.size == 0:
        return CandidateSet(
            np.zeros((0, FEATURE_COUNT), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    starts = np.unique(np.round(np.clip(starts, 0.0, duration - 0.05), 3)).astype(np.float32)

    local_median = _rolling_nanmedian(midi_track, max(3, int(fps * 2) + 1))
    prev_gap, next_gap = _fill_gaps(voiced, fps)
    onset_dist = _nearest_distances(starts, onsets, 0.5)
    next_onset_dist = _next_distance(starts, onsets, 1.5)
    is_onset = (onset_dist <= 0.045).astype(np.float32)
    (
        word_dist,
        inside_word,
        word_len,
        word_duration,
        is_word_start,
        line_frac,
        line_pos,
        line_start_dist,
        line_end_dist,
    ) = _word_features(words, starts)

    # Event times give every candidate a melodic base duration. The policy
    # learns whether to stretch or shorten it.
    events = np.unique(
        np.concatenate([onsets, voiced_starts.astype(np.float32), word_starts])
    ).astype(np.float32)
    voiced_midi = midi_track[np.isfinite(midi_track)]
    song_median = float(np.median(voiced_midi)) if voiced_midi.size else 60.0
    positive_rms = rms[rms > 1e-4]
    rms_median = float(np.median(positive_rms)) if positive_rms.size else 1.0

    rows: list[list[float]] = []
    kept_starts: list[float] = []
    kept_durations: list[float] = []
    kept_midis: list[float] = []
    for feature_index, start in enumerate(starts):
        position = int(np.searchsorted(events, float(start) + 0.05, side="left"))
        if position < events.size:
            base_duration = float(events[position] - start)
        else:
            base_duration = 0.8
        base_duration = float(np.clip(base_duration, 0.15, 4.0))
        frame = int(np.clip(round(float(start) * fps), 0, midi_track.size - 1))
        note_midi, confidence_mean = _pitch_at(
            midi_track, confidence, float(start), float(start) + min(0.30, base_duration), fps
        )
        if not np.isfinite(note_midi):
            continue
        end = min(duration, float(start) + base_duration)
        first = max(0, int(np.floor(float(start) * fps)))
        last = min(midi_track.size, int(np.ceil(end * fps)) + 1)
        if last - first < 2:
            continue
        pitch = midi_track[first:last]
        voiced_ratio = float(voiced[first:last].mean())
        rms_mean = _safe_nanmean(rms[first:last])
        rms_start = _safe_nanmean(rms[first : min(last, first + 3)])
        onset_start = float(np.max(flux[max(0, first - 1) : min(flux.size, first + 2)])) if flux.size else 0.0
        onset_local = float(np.max(flux[first : min(flux.size, first + max(3, int(0.2 * fps)))])) if flux.size else 0.0
        finite = pitch[np.isfinite(pitch)]
        if finite.size >= 4:
            quarter = max(1, finite.size // 4)
            slope = (
                float(np.median(finite[-quarter:])) - float(np.median(finite[:quarter]))
            ) / max(0.05, base_duration)
        else:
            slope = 0.0
        pitch_std = _safe_nanstd(pitch)
        rows.append(
            [
                float(start) / max(1.0, duration),
                float(np.log(max(0.05, base_duration))),
                (float(note_midi) - 60.0) / 24.0,
                confidence_mean,
                rms_mean,
                rms_start,
                onset_start,
                onset_local,
                float(onset_dist[feature_index]),
                float(np.clip(slope / 12.0, -4.0, 4.0)),
                float(np.clip(pitch_std / 2.0, 0.0, 4.0)),
                voiced_ratio,
                float(np.clip((note_midi - local_median[frame]) / 12.0, -2.0, 2.0)),
                float(prev_gap[frame]),
                float(next_gap[frame]),
                float(word_dist[feature_index]),
                float(inside_word[feature_index]),
                float(word_len[feature_index]),
                float(word_duration[feature_index]),
                float(line_frac[feature_index]),
                float(line_pos[feature_index]),
                float(line_start_dist[feature_index]),
                float(line_end_dist[feature_index]),
                float(next_onset_dist[feature_index]) / 1.5,
                float(np.clip((note_midi - song_median) / 12.0, -3.0, 3.0)),
                float(np.clip(rms_mean / max(1e-4, rms_median), 0.0, 4.0)) / 4.0,
                float(is_word_start[feature_index]),
                float(is_onset[feature_index]),
            ]
        )
        kept_starts.append(float(start))
        kept_durations.append(base_duration)
        kept_midis.append(float(note_midi))

    if not rows:
        return CandidateSet(
            np.zeros((0, FEATURE_COUNT), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    features_array = np.asarray(rows, dtype=np.float32)
    starts_array = np.asarray(kept_starts, dtype=np.float32)
    durations_array = np.asarray(kept_durations, dtype=np.float32)
    midis_array = np.asarray(kept_midis, dtype=np.float32)
    order = np.argsort(starts_array)
    return CandidateSet(
        features=features_array[order],
        starts=starts_array[order],
        durations=durations_array[order],
        midis=midis_array[order],
    )


def label_candidates(
    candidates: CandidateSet,
    level: Level,
    note_onset_tolerance: float = 0.14,
    note_pitch_tolerance: float = 1.7,
) -> CandidateSet:
    count = len(candidates)
    candidates.labels = np.zeros(count, dtype=np.float32)
    candidates.pitch_offsets = np.zeros(count, dtype=np.float32)
    candidates.onset_offsets = np.zeros(count, dtype=np.float32)
    candidates.log_scales = np.zeros(count, dtype=np.float32)
    candidates.total_notes = len(level.notes)
    if count == 0 or not level.notes:
        return candidates

    starts = candidates.starts
    midis = candidates.midis
    pairs: list[tuple[float, int, int]] = []
    for note_index, note in enumerate(level.notes):
        note_start = float(note.get("start", 0.0))
        note_midi = float(note.get("midi", 0.0))
        left = int(np.searchsorted(starts, note_start - note_onset_tolerance, side="left"))
        right = int(np.searchsorted(starts, note_start + note_onset_tolerance, side="right"))
        if right <= left:
            continue
        window = np.arange(left, right)
        onset_delta = np.abs(starts[window] - note_start)
        pitch_delta = np.abs(midis[window] - note_midi)
        mask = pitch_delta <= note_pitch_tolerance
        for offset in np.where(mask)[0]:
            candidate_index = int(window[offset])
            score = (
                float(onset_delta[offset]) / note_onset_tolerance
                + float(pitch_delta[offset]) / note_pitch_tolerance
            )
            pairs.append((score, note_index, candidate_index))

    pairs.sort()
    used_notes: set[int] = set()
    used_candidates: set[int] = set()
    for _, note_index, candidate_index in pairs:
        if note_index in used_notes or candidate_index in used_candidates:
            continue
        note = level.notes[note_index]
        used_notes.add(note_index)
        used_candidates.add(candidate_index)
        candidates.labels[candidate_index] = 1.0
        candidates.pitch_offsets[candidate_index] = float(note.get("midi", 0.0)) - midis[candidate_index]
        candidates.onset_offsets[candidate_index] = float(note.get("start", 0.0)) - starts[candidate_index]
        target_scale = float(
            np.log(max(0.05, float(note.get("dur", 0.0))) / max(0.05, candidates.durations[candidate_index]))
        )
        candidates.log_scales[candidate_index] = float(np.clip(target_scale, -1.5, 1.5))

    candidates.matched_notes = len(used_notes)
    candidates.missing_notes = [
        index for index in range(len(level.notes)) if index not in used_notes
    ]
    return candidates


def load_or_build_candidates(
    song: Song,
    features: dict[str, np.ndarray | float],
    force: bool = False,
) -> CandidateSet:
    path = candidate_cache_path(song)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        return CandidateSet(
            features=data["features"],
            starts=data["starts"],
            durations=data["durations"],
            midis=data["midis"],
        )
    candidates = build_candidates(song, features)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=candidates.features,
        starts=candidates.starts,
        durations=candidates.durations,
        midis=candidates.midis,
    )
    return candidates


def build_labeled(
    song: Song,
    features: dict[str, np.ndarray | float],
    level: Level,
    force: bool = False,
) -> CandidateSet:
    return label_candidates(load_or_build_candidates(song, features, force=force), level)
