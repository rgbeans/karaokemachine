"""Reading approved songs/maps and writing KaraokeMachine map files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import APPROVED_PATH, SONGS_DIR


@dataclass(frozen=True)
class ApprovedTrack:
    song_id: str
    map_relative: str
    title: str
    artist: str


@dataclass
class Song:
    id: str
    title: str
    artist: str
    duration: float
    guide_path: Path
    lines: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass
class Level:
    notes: list[dict[str, Any]]
    lines: list[dict[str, Any]]
    placed_words: list[dict[str, Any]]
    custom_words: list[dict[str, Any]]
    ribbon_links: list[dict[str, Any]]
    low: int
    high: int
    path: Path


@dataclass(frozen=True)
class WordTiming:
    text: str
    start: float
    end: float
    line: int
    word_index: int


def load_approved_tracks() -> list[ApprovedTrack]:
    """Return only the explicitly allowlisted, finished maps.

    There is deliberately no directory scan here. Draft maps and unapproved
    finished maps are never loaded as training data.
    """
    if not APPROVED_PATH.exists():
        raise FileNotFoundError(f"missing allowlist: {APPROVED_PATH}")
    payload = json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
    tracks: list[ApprovedTrack] = []
    for item in payload.get("tracks", []):
        song_id = str(item.get("song_id", "")).strip()
        map_relative = str(item.get("map", "")).strip()
        if not song_id or not map_relative:
            raise ValueError("approved.json entries need song_id and map")
        song_dir = SONGS_DIR / song_id
        map_path = song_dir / map_relative
        if not map_path.exists():
            raise FileNotFoundError(f"approved map is missing: {map_path}")
        data = json.loads(map_path.read_text(encoding="utf-8"))
        if data.get("draft"):
            raise ValueError(
                f"approved map {map_path} is a draft; only finished levels may be used"
            )
        if not data.get("notes"):
            raise ValueError(f"approved map {map_path} has no notes")
        tracks.append(
            ApprovedTrack(
                song_id=song_id,
                map_relative=map_relative,
                title=str(item.get("title") or song_id),
                artist=str(item.get("artist") or ""),
            )
        )
    if not tracks:
        raise ValueError("approved.json does not list any tracks")
    return tracks


def load_song(song_id: str) -> Song:
    song_dir = SONGS_DIR / song_id
    path = song_dir / "song.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    audio = raw.get("audio") or {}
    guide_name = audio.get("vocals") or audio.get("guide") or audio.get("backing")
    if not guide_name:
        raise ValueError(f"{song_id} has no vocal/guide audio")
    guide_path = song_dir / str(guide_name)
    if not guide_path.exists():
        raise FileNotFoundError(f"vocal/guide audio is missing: {guide_path}")
    duration = float(raw.get("duration") or 0.0)
    lines = list(raw.get("lines") or [])
    if duration <= 0.0:
        duration = max(
            (float(note.get("start", 0.0)) + float(note.get("dur", 0.0)) for note in raw.get("notes", [])),
            default=1.0,
        )
    return Song(
        id=song_id,
        title=str(raw.get("title") or song_id),
        artist=str(raw.get("artist") or ""),
        duration=duration,
        guide_path=guide_path,
        lines=lines,
        raw=raw,
    )


def load_level(song_id: str, map_relative: str) -> Level:
    path = SONGS_DIR / song_id / map_relative
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("draft"):
        raise ValueError(f"{path} is a draft and cannot be used as training data")
    notes = sorted(
        list(data.get("notes") or []),
        key=lambda note: (float(note.get("start", 0.0)), int(note.get("midi", 0))),
    )
    return Level(
        notes=notes,
        lines=list(data.get("lines") or []),
        placed_words=list(data.get("placedWords") or []),
        custom_words=list(data.get("customWords") or []),
        ribbon_links=list(data.get("ribbonLinks") or []),
        low=int(data.get("low") or 0),
        high=int(data.get("high") or 0),
        path=path,
    )


def flatten_words(lines: Iterable[dict[str, Any]]) -> list[WordTiming]:
    """Flatten timed lyric words.

    Lines with word timings use them directly. Lines without word timings are
    split evenly across the line so generated maps can still display lyrics.
    """
    result: list[WordTiming] = []
    for line_index, line in enumerate(lines):
        words = list(line.get("words") or [])
        if words:
            for word_index, word in enumerate(words):
                result.append(
                    WordTiming(
                        text=str(word.get("text") or ""),
                        start=float(word.get("start", line.get("start", 0.0))),
                        end=float(word.get("end", line.get("end", 0.0))),
                        line=line_index,
                        word_index=word_index,
                    )
                )
            continue
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        pieces = text.split()
        start = float(line.get("start", 0.0))
        end = float(line.get("end", start))
        span = max(0.05, end - start)
        step = span / max(1, len(pieces))
        for word_index, piece in enumerate(pieces):
            result.append(
                WordTiming(
                    text=piece,
                    start=start + step * word_index,
                    end=start + step * (word_index + 1),
                    line=line_index,
                    word_index=word_index,
                )
            )
    result.sort(key=lambda word: (word.start, word.line, word.word_index))
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
