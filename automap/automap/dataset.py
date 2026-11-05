"""Approved-map loading, feature normalization and augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import load_or_extract
from .candidates import CandidateSet, build_labeled
from .config import FEATURE_INDEX, PITCH_SLOPE_FEATURE, TIME_FEATURES
from .maps import ApprovedTrack, Level, Song, load_level, load_song


@dataclass
class TrackData:
    track: ApprovedTrack
    song: Song
    level: Level
    candidates: CandidateSet
    train_mask: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def normalized(self, indices: np.ndarray | slice | None = None) -> np.ndarray:
        features = self.candidates.features
        if indices is not None:
            features = features[indices]
        return ((features - self.feature_mean) / self.feature_std).astype(np.float32)


def load_track(track: ApprovedTrack, force_audio: bool = False) -> TrackData:
    song = load_song(track.song_id)
    level = load_level(track.song_id, track.map_relative)
    audio = load_or_extract(song, force=force_audio)
    candidates = build_labeled(song, audio, level)
    if len(candidates) == 0:
        raise RuntimeError(f"{song.title}: no note candidates were extracted")
    # A time-based holdout is too harsh with one approved song: the last
    # section can be a completely different musical idea. Use a stable random
    # split instead so validation measures the policy's feature discrimination.
    rng = np.random.default_rng(0)
    train_mask = rng.random(len(candidates)) < 0.85
    # Keep at least one positive example on each side when the map allows it.
    positive = candidates.labels > 0.5
    if not np.any(positive & train_mask):
        train_mask[np.where(positive)[0][0]] = True
    if not np.any(positive & ~train_mask):
        train_mask[np.where(positive)[0][-1]] = False
    train_features = candidates.features[train_mask]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-5] = 1.0
    return TrackData(
        track=track,
        song=song,
        level=level,
        candidates=candidates,
        train_mask=train_mask,
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
    )


def augment_features(
    features: np.ndarray, time_scale: float, pitch_shift: float
) -> np.ndarray:
    """Apply a tempo/pitch transformation in feature space."""
    result = features.copy()
    result[:, FEATURE_INDEX["log_duration"]] += np.log(max(0.5, time_scale))
    for name in TIME_FEATURES:
        result[:, FEATURE_INDEX[name]] *= time_scale
    result[:, FEATURE_INDEX[PITCH_SLOPE_FEATURE]] /= max(0.5, time_scale)
    result[:, FEATURE_INDEX["midi_norm"]] += pitch_shift / 24.0
    return result


def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def selection_indices(
    labels: np.ndarray,
    negative_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """All positives plus a random sample of negatives for one epoch."""
    positive = np.where(labels > 0.5)[0]
    negative = np.where(labels <= 0.5)[0]
    if positive.size == 0:
        return rng.permutation(negative)
    keep = min(negative.size, max(1, int(positive.size * negative_ratio)))
    chosen = rng.choice(negative, size=keep, replace=False)
    return np.concatenate([positive, chosen])


def contiguous_chunks(indices: np.ndarray, size: int) -> list[np.ndarray]:
    """Split an ordered index array into contiguous candidate chunks."""
    if indices.size == 0:
        return []
    order = np.argsort(indices)
    ordered = indices[order]
    chunks: list[np.ndarray] = []
    for start in range(0, ordered.size, size):
        chunks.append(ordered[start : start + size])
    return chunks
