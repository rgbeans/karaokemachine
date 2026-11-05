"""Public karaoke timing sources and word-level alignment targets.

The timing files come from USDB's public UltraStar "Show Notes" pages. They
contain note/syllable timings and lyric text but no audio. Only the derived
timing alignments are cached locally; raw pages stay in the ignored cache.
"""

from __future__ import annotations

import difflib
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import CACHE_DIR
from .maps import Song

KARAOKE_CACHE = CACHE_DIR / "karaoke"
USER_AGENT = "KaraokeMachine-Automap/0.1 (timing research)"


@dataclass(frozen=True)
class KaraokeSource:
    song_id: str
    usdb_id: int
    title: str
    # USDB karaoke timeline -> local song timeline.
    offset: float

    @property
    def source_url(self) -> str:
        return f"https://usdb.animux.de/index.php?link=detail&id={self.usdb_id}"

    @property
    def notes_url(self) -> str:
        return f"https://usdb.animux.de/view.php?id={self.usdb_id}&database1=deluxe_songs"


SOURCES: tuple[KaraokeSource, ...] = (
    KaraokeSource("yt-zTuD8k3JvxQ", 1597, "Them Bones", 0.40),
    KaraokeSource("yt-6JA25BIxgtk", 24234, "We Die Young", 0.35),
    KaraokeSource("yt-r80HF68KM8g", 16056, "No Excuses", -3.70),
    KaraokeSource("yt-Nco_kh8xJDs", 63, "Would?", -0.25),
)
BY_SONG = {source.song_id: source for source in SOURCES}
BY_INDEX = {index: source for index, source in enumerate(SOURCES)}


@dataclass
class WordTarget:
    line_index: int
    word_index: int
    word_count: int
    word: dict[str, Any]
    target_time: float
    local_time: float


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("’", "'").replace("‘", "'"))


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_timing(source: KaraokeSource, force: bool = False) -> dict[str, Any]:
    """Fetch and parse the public USDB timing payload for one source."""
    KARAOKE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = KARAOKE_CACHE / f"{source.usdb_id}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    detail = _fetch(source.source_url)
    detail_text = re.sub(r"<[^>]+>", " ", detail)
    detail_text = re.sub(r"\s+", " ", detail_text)
    bpm_match = re.search(r"BPM\s*([0-9.]+)", detail_text)
    gap_match = re.search(r"GAP\s*(-?[0-9.]+)", detail_text)
    if not bpm_match or not gap_match:
        raise RuntimeError(f"USDB timing metadata missing for {source.title}")
    bpm = float(bpm_match.group(1))
    gap = float(gap_match.group(1))

    viewer = _fetch(source.notes_url)
    entries = re.findall(
        r"giveinfo0\('([^']*)','([^']*)','([^']*)','([^']*)','([^']*)','([^']*)'\)",
        viewer,
    )
    notes: list[dict[str, Any]] = []
    for note_type, beat, length, pitch, note_name, text in entries:
        notes.append(
            {
                "type": note_type,
                "beat": int(beat),
                "length": int(length),
                "pitch": pitch,
                "note": note_name,
                "text": text,
                # UltraStar BPM is in quarter-note units; its note grid is /4.
                "time": round(gap / 1000.0 + int(beat) * 60.0 / (bpm * 4.0), 3),
            }
        )
    payload = {
        "source_url": source.source_url,
        "notes_url": source.notes_url,
        "usdb_id": source.usdb_id,
        "title": source.title,
        "bpm": bpm,
        "gap": gap,
        "notes": notes,
    }
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def fetch_notes(source: KaraokeSource, force: bool = False) -> list[dict[str, Any]]:
    return list(fetch_timing(source, force=force)["notes"])


def _map_word_targets(
    song: Song,
    notes: list[dict[str, Any]],
    offset: float,
    max_residual: float = 2.5,
) -> list[WordTarget]:
    timed = [note for note in notes if normalize(note.get("text", ""))]
    targets: list[WordTarget] = []
    for line_index, line in enumerate(song.lines):
        words = list(line.get("words") or [])
        line_text = normalize(str(line.get("text") or ""))
        if not words or not line_text:
            continue
        start = float(line.get("start", 0.0))
        end = float(line.get("end", start))
        segment = [
            note
            for note in timed
            if start - offset - 0.7 <= note["time"] <= end - offset + 0.7
        ]
        if not segment:
            continue
        char_starts: list[int] = []
        char_times: list[float] = []
        pieces: list[str] = []
        cursor = 0
        for note in segment:
            piece = normalize(note["text"])
            if not piece:
                continue
            char_starts.append(cursor)
            char_times.append(float(note["time"]))
            pieces.append(piece)
            cursor += len(piece)
        joined = "".join(pieces)
        if not joined:
            continue
        mapping: dict[int, int] = {}
        matcher = difflib.SequenceMatcher(None, line_text, joined)
        for block in matcher.get_matching_blocks():
            for index in range(block.size):
                mapping[block.a + index] = block.b + index
        if not mapping:
            continue
        word_cursor = 0
        for word_index, word in enumerate(words):
            word_text = normalize(str(word.get("text") or ""))
            if not word_text:
                continue
            position = line_text.find(word_text, word_cursor)
            if position < 0:
                continue
            word_cursor = position + len(word_text)
            nearest = min(mapping, key=lambda key: abs(key - position))
            if abs(nearest - position) > 4:
                continue
            mapped = mapping[nearest]
            syllable_index = next(
                (
                    i
                    for i, char_start in enumerate(char_starts)
                    if char_start <= mapped < char_start + len(pieces[i])
                ),
                None,
            )
            if syllable_index is None:
                syllable_index = min(
                    range(len(char_starts)),
                    key=lambda i: abs(char_starts[i] - mapped),
                )
            target_time = char_times[syllable_index] + offset
            local_time = float(word.get("start", 0.0))
            if abs(target_time - local_time) > max_residual:
                continue
            targets.append(
                WordTarget(
                    line_index=line_index,
                    word_index=word_index,
                    word_count=len(words),
                    word=word,
                    target_time=target_time,
                    local_time=local_time,
                )
            )
    return targets


def load_targets(song: Song) -> tuple[list[WordTarget], KaraokeSource | None]:
    source = BY_SONG.get(song.id)
    if source is None:
        return [], None
    notes = fetch_notes(source)
    return _map_word_targets(song, notes, source.offset), source
