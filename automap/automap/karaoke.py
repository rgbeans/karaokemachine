"""Public karaoke timing sources and alignment targets.

The word-level sources come from USDB's public UltraStar "Show Notes" pages.
They contain note/syllable timings and lyric text but no audio. The line-level
source comes from LRCLIB's public synced lyrics; it only provides a timestamp
for each lyric line, not for each word or note. Only derived timing alignments
are cached locally; raw pages stay in the ignored cache.
"""

from __future__ import annotations

import difflib
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Union

from .config import CACHE_DIR
from .maps import Song

KARAOKE_CACHE = CACHE_DIR / "karaoke"
LRCLIB_CACHE = CACHE_DIR / "lrclib"
USER_AGENT = "KaraokeMachine-Automap/0.1 (timing research)"


@dataclass(frozen=True)
class KaraokeSource:
    song_id: str
    usdb_id: int
    title: str
    # USDB karaoke timeline -> local song timeline.
    offset: float

    kind = "usdb"
    level = "word"

    @property
    def source_url(self) -> str:
        return f"https://usdb.animux.de/index.php?link=detail&id={self.usdb_id}"

    @property
    def notes_url(self) -> str:
        return f"https://usdb.animux.de/view.php?id={self.usdb_id}&database1=deluxe_songs"


@dataclass(frozen=True)
class LineSource:
    song_id: str
    lrclib_id: int
    title: str
    # LRCLIB timeline -> local song timeline.
    offset: float

    kind = "lrclib"
    level = "line"

    @property
    def source_url(self) -> str:
        return f"https://lrclib.net/api/get/{self.lrclib_id}"

    @property
    def lyrics_url(self) -> str:
        return f"https://lrclib.net/api/get/{self.lrclib_id}"


TimingSource = Union[KaraokeSource, LineSource]


SOURCES: tuple[KaraokeSource, ...] = (
    KaraokeSource("yt-zTuD8k3JvxQ", 1597, "Them Bones", 0.40),
    KaraokeSource("yt-6JA25BIxgtk", 24234, "We Die Young", 0.35),
    KaraokeSource("yt-r80HF68KM8g", 16056, "No Excuses", -3.70),
    KaraokeSource("yt-Nco_kh8xJDs", 63, "Would?", -0.25),
)
BY_SONG = {source.song_id: source for source in SOURCES}
BY_INDEX = {index: source for index, source in enumerate(SOURCES)}

# Songs with only public line timings. The local lyric lines for Sunshine were
# imported from LRCLIB 819361 and shifted by 0.29 s to match the local audio.
LINE_SOURCES: tuple[LineSource, ...] = (
    LineSource("yt-2maHkdezdEc", 819361, "Sunshine", 0.29),
)
LINE_BY_SONG = {source.song_id: source for source in LINE_SOURCES}


def source_index(source: TimingSource) -> int:
    if isinstance(source, KaraokeSource):
        return SOURCES.index(source)
    return len(SOURCES) + LINE_SOURCES.index(source)


@dataclass
class WordTarget:
    line_index: int
    word_index: int
    word_count: int
    word: dict[str, Any]
    target_time: float
    local_time: float
    level: str = "word"
    line_target_start: float | None = None
    line_local_start: float | None = None


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("’", "'").replace("‘", "'"))


def _fetch(url: str, accept: str | None = None) -> str:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
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


_LRC_TIMESTAMP = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def fetch_line_timing(source: LineSource, force: bool = False) -> dict[str, Any]:
    """Fetch and parse the public LRCLIB line timing payload for one source."""
    LRCLIB_CACHE.mkdir(parents=True, exist_ok=True)
    cache = LRCLIB_CACHE / f"{source.lrclib_id}.json"
    if cache.exists() and not force:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("lines"):
            return payload

    raw = json.loads(_fetch(source.source_url, accept="application/json"))
    synced = str(raw.get("syncedLyrics") or "")
    lines: list[dict[str, Any]] = []
    for raw_line in synced.splitlines():
        matches = _LRC_TIMESTAMP.findall(raw_line)
        text = _LRC_TIMESTAMP.sub("", raw_line)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not matches or not text:
            continue
        for minutes, seconds in matches:
            lines.append(
                {
                    "start": round(int(minutes) * 60.0 + float(seconds), 3),
                    "text": text,
                }
            )
    lines.sort(key=lambda line: line["start"])
    if not lines:
        raise RuntimeError(f"LRCLIB has no synced lyric lines for {source.title}")

    payload = {
        "source_url": source.source_url,
        "lrclib_id": source.lrclib_id,
        "title": source.title,
        "lines": lines,
        "synced": True,
    }
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _match_line_indices(
    song: Song, public_lines: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Match local lyric lines to public LRC lines by normalized text."""
    local_text = [normalize(str(line.get("text") or "")) for line in song.lines]
    public_text = [normalize(str(line.get("text") or "")) for line in public_lines]
    matched: list[tuple[int, int]] = []
    used_public: set[int] = set()
    matcher = difflib.SequenceMatcher(None, local_text, public_text, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            local_index = block.a + offset
            public_index = block.b + offset
            if local_text[local_index] and public_text[public_index]:
                matched.append((local_index, public_index))
                used_public.add(public_index)

    matched_local = {item[0] for item in matched}
    for local_index, text in enumerate(local_text):
        if local_index in matched_local or not text:
            continue
        best_index = None
        best_ratio = 0.58
        for public_index in range(
            max(0, local_index - 3), min(len(public_text), local_index + 4)
        ):
            if public_index in used_public or not public_text[public_index]:
                continue
            ratio = difflib.SequenceMatcher(
                None, text, public_text[public_index], autojunk=False
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = public_index
        if best_index is not None:
            matched.append((local_index, best_index))
            used_public.add(best_index)

    matched.sort()
    return matched


def _map_line_targets(
    song: Song,
    public_lines: list[dict[str, Any]],
    offset: float,
) -> list[WordTarget]:
    """Create line-start calibration targets from public LRC lines.

    LRCLIB times each lyric line, not each word. Every word in a matched line
    therefore gets the same target shift: the shift between the public line
    start and the local line start. This calibrates line placement only; it is
    not word-level ground truth.
    """
    targets: list[WordTarget] = []
    for local_index, public_index in _match_line_indices(song, public_lines):
        line = song.lines[local_index]
        words = list(line.get("words") or [])
        if not words:
            continue
        local_start = float(line.get("start", 0.0))
        target_start = float(public_lines[public_index]["start"]) + offset
        shift = target_start - local_start
        for word_index, word in enumerate(words):
            local_time = float(word.get("start", local_start))
            targets.append(
                WordTarget(
                    line_index=local_index,
                    word_index=word_index,
                    word_count=len(words),
                    word=word,
                    target_time=local_time + shift,
                    local_time=local_time,
                    level="line",
                    line_target_start=target_start,
                    line_local_start=local_start,
                )
            )
    return targets


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


def load_targets(song: Song) -> tuple[list[WordTarget], TimingSource | None]:
    source = BY_SONG.get(song.id)
    if source is not None:
        notes = fetch_notes(source)
        return _map_word_targets(song, notes, source.offset), source
    line_source = LINE_BY_SONG.get(song.id)
    if line_source is not None:
        payload = fetch_line_timing(line_source)
        return _map_line_targets(song, payload["lines"], line_source.offset), line_source
    return [], None
