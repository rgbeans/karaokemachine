"""Vocal-track decoding and frame-level pitch/volume/onset analysis.

The tool deliberately avoids heavy audio libraries. ffmpeg decodes the vocal
stem to mono float32, then numpy computes RMS, spectral flux and an
autocorrelation pitch track. Results are cached per song.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, FPS, FRAME_SIZE, SAMPLE_RATE
from .maps import Song

CACHE_VERSION = 2


def _cache_path(song: Song) -> Path:
    try:
        stat = song.guide_path.stat()
        signature = f"{stat.st_size}-{stat.st_mtime_ns}"
    except OSError:
        signature = "missing"
    return CACHE_DIR / f"{song.id}-audio-v{CACHE_VERSION}-{signature}.npz"


def decode_vocal(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required to decode vocal tracks") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {message}") from error
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio from {path}")
    return samples.astype(np.float32, copy=False)


def _frames(samples: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))
    count = 1 + max(0, (samples.size - 1) // hop)
    needed = (count - 1) * hop + frame_size
    if samples.size < needed:
        samples = np.pad(samples, (0, needed - samples.size))
    return np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop]


def _nanmedian_filter(values: np.ndarray, size: int = 5) -> np.ndarray:
    pad = size // 2
    padded = np.pad(values, (pad, pad), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        filtered = np.nanmedian(windows, axis=1)
    return filtered


def _pitch_track(
    frames: np.ndarray,
    sample_rate: int,
    chunk_frames: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (midi, confidence) using the YIN difference function.

    Plain autocorrelation tends to jump an octave on the singer's harmonics.
    YIN's cumulative mean normalized difference is much more stable for this.
    """
    count = frames.shape[0]
    midi = np.full(count, np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    min_lag = max(2, int(sample_rate / 1000.0))
    max_lag = min(frames.shape[1] - 2, int(sample_rate / 65.0))

    for start in range(0, count, chunk_frames):
        block = frames[start : start + chunk_frames]
        windowed = block * window
        # Difference function via the autocorrelation of the same frame.
        spectrum = np.fft.rfft(windowed, 2 * windowed.shape[1], axis=1)
        autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[
            :, : windowed.shape[1]
        ]
        autocorrelation /= np.maximum(autocorrelation[:, :1], 1e-9)
        difference = np.maximum(0.0, 2.0 * (1.0 - autocorrelation))
        difference[:, 0] = 0.0
        cumulative = np.cumsum(difference[:, 1:], axis=1)
        lags = np.arange(1, difference.shape[1], dtype=np.float32)[None, :]
        cmnd = difference[:, 1:] * lags / np.maximum(cumulative, 1e-9)
        search = cmnd[:, min_lag:max_lag]

        best = np.argmin(search, axis=1)
        values = search[np.arange(search.shape[0]), best]
        for row in range(search.shape[0]):
            line = search[row]
            local = np.where(
                (line < 0.20)
                & (line <= np.roll(line, 1))
                & (line <= np.roll(line, -1))
            )[0]
            if local.size:
                best[row] = int(local[0])
                values[row] = line[best[row]]
        lags_out = (best + min_lag).astype(np.float64)
        for row in range(search.shape[0]):
            lag = int(lags_out[row])
            if lag <= 0 or lag + 1 >= cmnd.shape[1]:
                continue
            left = cmnd[row, lag - 1]
            center = cmnd[row, lag]
            right = cmnd[row, lag + 1]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-9:
                lags_out[row] += 0.5 * (left - right) / denominator
        frequency = sample_rate / np.maximum(lags_out, 1e-6)
        certainty = np.clip(1.0 - values, 0.0, 1.0)
        valid = (certainty > 0.20) & (frequency >= 65.0) & (frequency <= 1000.0)
        output = 69.0 + 12.0 * np.log2(np.maximum(frequency, 1e-6) / 440.0)
        midi[start : start + search.shape[0]] = np.where(valid, output, np.nan)
        confidence[start : start + search.shape[0]] = np.where(valid, certainty, 0.0)

    smoothed = _nanmedian_filter(midi, 5)
    valid = np.isfinite(smoothed)
    midi = np.where(valid, smoothed, np.nan).astype(np.float32)
    return midi, confidence


def compute_features(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, np.ndarray | float]:
    hop = max(1, int(round(sample_rate / FPS)))
    frames = _frames(samples, FRAME_SIZE, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)

    chunk_frames = 1024
    flux_parts: list[np.ndarray] = []
    previous = None
    for start in range(0, frames.shape[0], chunk_frames):
        block = frames[start : start + chunk_frames]
        magnitude = np.abs(np.fft.rfft(block * np.hanning(block.shape[1]), axis=1))
        if previous is not None:
            magnitude = np.vstack([previous[None, :], magnitude])
        difference = np.maximum(0.0, magnitude[1:] - magnitude[:-1]).sum(axis=1)
        flux_parts.append(difference.astype(np.float32))
        previous = magnitude[-1]
    flux = np.concatenate(flux_parts) if flux_parts else np.zeros(0, dtype=np.float32)
    if flux.size:
        positive = flux[flux > 0]
        scale = float(np.median(positive)) + 1e-6 if positive.size else 1e-6
        flux = np.clip(flux / scale, 0.0, 20.0)
        flux = np.concatenate([np.zeros(1, dtype=np.float32), flux])[: frames.shape[0]]

    midi, confidence = _pitch_track(frames, sample_rate)
    nonzero = rms[rms > 1e-4]
    rms_floor = max(1e-4, float(np.median(nonzero)) * 0.12) if nonzero.size else 1e-4
    voiced = (confidence > 0.22) & (rms > rms_floor) & np.isfinite(midi)

    return {
        "midi": midi,
        "confidence": confidence,
        "rms": rms,
        "flux": flux,
        "voiced": voiced,
        "duration": float(samples.size) / float(sample_rate),
        "fps": float(FPS),
        "sample_rate": float(sample_rate),
        "rms_floor": float(rms_floor),
    }


def load_or_extract(song: Song, force: bool = False) -> dict[str, np.ndarray | float]:
    path = _cache_path(song)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    samples = decode_vocal(song.guide_path)
    features = compute_features(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)
    return features
