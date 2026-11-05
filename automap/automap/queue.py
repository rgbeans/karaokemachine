"""Persistent generation targets and the resumable grading queue."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import QUEUE_PATH, TARGETS_PATH
from .maps import SONGS_DIR, load_approved_tracks, load_song, write_json


@dataclass(frozen=True)
class Target:
    song_id: str
    title: str
    album: str


@dataclass
class QueueState:
    next_index: int = 0
    current_song_id: str | None = None
    current_round: int = 1
    generated_count: int = 0
    songs: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_targets() -> list[Target]:
    """Load the explicit generation queue.

    Training data is separate (approved.json). The approved training song is
    never a generation target, and the queue file can also list explicit
    never-generate IDs.
    """
    if not TARGETS_PATH.exists():
        raise FileNotFoundError(f"missing targets file: {TARGETS_PATH}")
    payload = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    forbidden = set(payload.get("never_generate") or [])
    forbidden.update(track.song_id for track in load_approved_tracks())
    targets: list[Target] = []
    seen: set[str] = set()
    for item in payload.get("tracks") or []:
        song_id = str(item.get("song_id") or "").strip()
        if not song_id or song_id in seen or song_id in forbidden:
            continue
        song_dir = SONGS_DIR / song_id
        if not (song_dir / "song.json").exists():
            print(f"warning: target {song_id} has no song.json; skipping")
            continue
        try:
            song = load_song(song_id)
        except Exception as error:
            print(f"warning: target {song_id} has no usable audio ({error}); skipping")
            continue
        targets.append(
            Target(
                song_id=song_id,
                title=str(item.get("title") or song.title),
                album=str(item.get("album") or song.album),
            )
        )
        seen.add(song_id)
    if not targets:
        raise ValueError("targets.json did not produce any usable songs")
    return targets


def load_queue() -> QueueState:
    if not QUEUE_PATH.exists():
        return QueueState()
    try:
        payload = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"warning: ignoring unreadable queue ({error})")
        return QueueState()
    return QueueState(
        next_index=int(payload.get("next_index") or 0),
        current_song_id=payload.get("current_song_id"),
        current_round=int(payload.get("current_round") or 1),
        generated_count=int(payload.get("generated_count") or 0),
        songs=dict(payload.get("songs") or {}),
    )


def save_queue(state: QueueState) -> None:
    write_json(
        QUEUE_PATH,
        {
            "version": 1,
            "next_index": state.next_index,
            "current_song_id": state.current_song_id,
            "current_round": state.current_round,
            "generated_count": state.generated_count,
            "songs": state.songs,
        },
    )


def mark_generated(
    state: QueueState,
    targets: list[Target],
    index: int,
    target: Target,
    info: dict[str, Any],
) -> None:
    total = max(1, len(targets))
    state.current_round = state.generated_count // total + 1
    state.generated_count += 1
    state.current_song_id = target.song_id
    state.next_index = (index + 1) % total
    progress = state.songs.setdefault(target.song_id, {})
    progress["generated"] = int(progress.get("generated") or 0) + 1
    progress["last_candidate"] = str(info.get("archive") or "")
    progress["last_generated_at"] = datetime.now(timezone.utc).isoformat()
    save_queue(state)


def mark_retried(
    state: QueueState,
    target: Target,
    info: dict[str, Any],
) -> None:
    """Record another attempt at the current song without advancing the queue."""
    progress = state.songs.setdefault(target.song_id, {})
    progress["generated"] = int(progress.get("generated") or 0) + 1
    progress["last_candidate"] = str(info.get("archive") or "")
    progress["last_generated_at"] = datetime.now(timezone.utc).isoformat()
    save_queue(state)


def mark_graded(
    state: QueueState,
    song_id: str,
    grade: int,
    reward: float,
) -> None:
    progress = state.songs.setdefault(song_id, {})
    progress["graded"] = int(progress.get("graded") or 0) + 1
    progress["last_grade"] = int(grade)
    progress["last_reward"] = float(reward)
    progress["last_graded_at"] = datetime.now(timezone.utc).isoformat()
    save_queue(state)


def mark_skipped(state: QueueState, song_id: str) -> None:
    progress = state.songs.setdefault(song_id, {})
    progress["skipped"] = int(progress.get("skipped") or 0) + 1
    save_queue(state)


def mark_position(
    state: QueueState,
    song_id: str,
    candidate_id: str,
    seconds: float,
) -> None:
    progress = state.songs.setdefault(song_id, {})
    progress["last_position"] = float(max(0.0, seconds))
    progress["last_position_candidate"] = candidate_id
    save_queue(state)
