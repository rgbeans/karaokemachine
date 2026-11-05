"""Local browser UI: sidescrolling preview, thumbs grading and a resumable queue."""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import PACKAGE_DIR
from .dataset import TrackData, load_track
from .generate import generate_candidate, pending_path
from .grade import load_pending, update_from_grade
from .maps import Song, load_approved_tracks, load_song
from .queue import (
    QueueState,
    Target,
    load_queue,
    load_targets,
    mark_generated,
    mark_graded,
    mark_position,
    mark_retried,
    mark_skipped,
    save_queue,
)

ASSET_DIR = PACKAGE_DIR / "ui_assets"
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}
AUDIO_TYPES = {
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
}


class Session:
    def __init__(
        self,
        targets: list[Target],
        queue: QueueState,
        install: bool = True,
    ) -> None:
        self.targets = targets
        self.queue = queue
        self.install = install
        self.anchor_data: list[TrackData] = [
            load_track(track) for track in load_approved_tracks()
        ]
        self.lock = threading.Lock()
        self.songs: dict[str, Song] = {}
        self.current_target: Target | None = None
        self.current_song: Song | None = None
        self.state: dict[str, Any] | None = None
        self.last_event: dict[str, Any] | None = None
        self._resume_or_generate()

    def _song(self, song_id: str) -> Song:
        if song_id not in self.songs:
            self.songs[song_id] = load_song(song_id)
        return self.songs[song_id]

    def _target_index(self, song_id: str | None) -> int | None:
        if song_id is None:
            return None
        for index, target in enumerate(self.targets):
            if target.song_id == song_id:
                return index
        return None

    def _set_state(self, info: dict[str, Any]) -> None:
        target = self.current_target
        song = self.current_song
        if target is None or song is None:
            raise RuntimeError("no current generation target")
        path = info.get("installed") or info.get("archive")
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        index = self._target_index(target.song_id) or 0
        total = len(self.targets)
        next_target = self.targets[(index + 1) % total]
        progress = self.queue.songs.get(target.song_id, {})
        vocals_available = song.guide_path != song.backing_path
        backing_available = song.backing_path is not None
        candidate_id = str(info["archive"])
        resume_position = 0.0
        if progress.get("last_position_candidate") == candidate_id:
            resume_position = min(
                max(0.0, float(progress.get("last_position") or 0.0)),
                max(0.0, song.duration - 0.5),
            )
        self.state = {
            "song_id": song.id,
            "title": song.title,
            "artist": song.artist,
            "album": target.album or song.album,
            "duration": song.duration,
            "song_index": index + 1,
            "song_total": total,
            "round": self.queue.current_round,
            "audio": {
                "vocals": "/api/audio/vocals",
                "backing": "/api/audio/backing",
                "vocals_available": vocals_available,
                "backing_available": backing_available,
            },
            "notes": document.get("notes") or [],
            "lines": document.get("lines") or [],
            "placedWords": document.get("placedWords") or [],
            "low": int(document.get("low") or 0),
            "high": int(document.get("high") or 0),
            "note_count": len(document.get("notes") or []),
            "candidate_id": candidate_id,
            "archive": candidate_id,
            "resume_position": resume_position,
            "installed": str(info["installed"]) if info.get("installed") else "",
            "model_step": int(info.get("model_step") or 0),
            "grades_for_song": int(progress.get("graded") or 0),
            "generated_for_song": int(progress.get("generated") or 0),
            "last_event": self.last_event,
            "next_title": next_target.title,
            "next_album": next_target.album,
        }

    def _set_state_from_pending(
        self, pending: dict[str, Any], target: Target, song: Song
    ) -> None:
        archive = Path(str(pending["archive"][0]))
        installed = str(pending["installed"][0])
        info = {
            "archive": archive,
            "installed": Path(installed) if installed else None,
            "model_step": int(pending["model_step"][0]),
        }
        self.current_target = target
        self.current_song = song
        self._set_state(info)

    def _resume_or_generate(self) -> None:
        song_id = self.queue.current_song_id
        index = self._target_index(song_id)
        if index is not None:
            target = self.targets[index]
            pending = load_pending(target.song_id)
            if pending is not None:
                song = self._song(target.song_id)
                self.queue.next_index = (index + 1) % len(self.targets)
                save_queue(self.queue)
                self._set_state_from_pending(pending, target, song)
                return
        self._generate_next()

    def _generate_next(self) -> None:
        total = len(self.targets)
        index = self.queue.next_index % total
        errors: list[str] = []
        for _ in range(total):
            target = self.targets[index]
            try:
                info = generate_candidate(target.song_id, install=self.install)
            except Exception as error:
                errors.append(f"{target.title}: {error}")
                print(f"warning: skipping {target.title}: {error}")
                index = (index + 1) % total
                continue
            song = self._song(target.song_id)
            self.current_target = target
            self.current_song = song
            mark_generated(self.queue, self.targets, index, target, info)
            self._set_state(info)
            return
        raise RuntimeError("could not generate any target: " + "; ".join(errors))

    def _generate_same(self) -> None:
        if self.current_target is None or self.current_song is None:
            raise RuntimeError("no current target to retry")
        info = generate_candidate(self.current_target.song_id, install=self.install)
        mark_retried(self.queue, self.current_target, info)
        self._set_state(info)

    def grade(self, grade: int, candidate_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.current_target is None or self.current_song is None:
                raise RuntimeError("no current candidate")
            pending = load_pending(self.current_target.song_id)
            if pending is None:
                raise RuntimeError("no candidate is waiting for a grade; generate a new one")
            archive = str(pending["archive"][0])
            if candidate_id and candidate_id != archive:
                raise RuntimeError("that candidate was already replaced")
            record = update_from_grade(
                self.current_target.song_id,
                grade,
                anchor_data=self.anchor_data,
            )
            mark_graded(
                self.queue,
                self.current_target.song_id,
                grade,
                float(record["reward"]),
            )
            graded_progress = self.queue.songs.get(self.current_target.song_id, {})
            retry = float(record["reward"]) < 0.0
            self.last_event = {
                "kind": "grade",
                "grade": grade,
                "reward": float(record["reward"]),
                "title": self.current_target.title,
                "album": self.current_target.album,
                "grades_for_song": int(graded_progress.get("graded") or 0),
                "action": "retry" if retry else "next",
            }
            if retry:
                try:
                    self._generate_same()
                except Exception as error:
                    print(f"warning: retry failed for {self.current_target.title}: {error}")
                    self.last_event["action"] = "next"
                    self._generate_next()
            else:
                self._generate_next()
            return self.state or {}

    def next_candidate(self) -> dict[str, Any]:
        with self.lock:
            if self.current_target is not None:
                pending_path(self.current_target.song_id).unlink(missing_ok=True)
                mark_skipped(self.queue, self.current_target.song_id)
                self.last_event = {
                    "kind": "skip",
                    "grade": 0,
                    "reward": 0.0,
                    "title": self.current_target.title,
                    "album": self.current_target.album,
                }
            self._generate_next()
            return self.state or {}

    def update_position(self, seconds: float, candidate_id: str | None = None) -> None:
        with self.lock:
            if self.current_target is None:
                return
            pending = load_pending(self.current_target.song_id)
            if pending is None:
                return
            archive = str(pending["archive"][0])
            if candidate_id and candidate_id != archive:
                return
            mark_position(
                self.queue,
                self.current_target.song_id,
                archive,
                float(seconds),
            )


class Handler(BaseHTTPRequestHandler):
    server_version = "KaraokeMachineAutomap/0.1"

    @property
    def session(self) -> Session:
        return self.server.session  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str) -> None:
        path = ASSET_DIR / name
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _audio_path(self, kind: str) -> Path | None:
        song = self.session.current_song
        if song is None:
            return None
        if kind == "backing":
            return song.backing_path
        return song.guide_path

    def _send_audio(self, kind: str = "vocals", head_only: bool = False) -> None:
        path = self._audio_path(kind)
        if path is None or not path.exists():
            self.send_error(404)
            return
        size = path.stat().st_size
        content_type = AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                spec = range_header[6:].split(",", 1)[0]
                left, _, right = spec.partition("-")
                if left:
                    start = int(left)
                    end = int(right) if right else size - 1
                elif right:
                    start = max(0, size - int(right))
                    end = size - 1
                if start >= size or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route_audio(self, path: str, head_only: bool = False) -> bool:
        if path in {"/api/audio", "/api/audio/vocals"}:
            self._send_audio("vocals", head_only=head_only)
            return True
        if path == "/api/audio/backing":
            self._send_audio("backing", head_only=head_only)
            return True
        return False

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if self._route_audio(path, head_only=True):
            return
        if path in STATIC_FILES:
            static = ASSET_DIR / STATIC_FILES[path]
            if not static.exists():
                self.send_error(404)
                return
            body = static.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type", mimetypes.guess_type(static.name)[0] or "application/octet-stream"
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_error(405)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(200, self.session.state or {})
            return
        if self._route_audio(path):
            return
        if path in STATIC_FILES:
            self._send_static(STATIC_FILES[path])
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/grade", "/api/next", "/api/position"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return
        try:
            if path == "/api/position":
                self.session.update_position(
                    float(payload.get("seconds") or 0.0),
                    payload.get("candidate_id"),
                )
                self.send_response(204)
                self.end_headers()
                return
            elif path == "/api/next":
                state = self.session.next_candidate()
                self._send_json(200, {"state": state})
            else:
                grade = int(payload.get("grade", 0))
                if grade not in {1, 2, 3, 4, 5}:
                    self._send_json(400, {"error": "grade must be 1-5"})
                    return
                state = self.session.grade(grade, payload.get("candidate_id"))
            self._send_json(200, {"state": state})
        except Exception as error:
            self._send_json(409, {"error": str(error)})


def run_ui(
    song_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    install: bool = True,
    open_browser: bool = True,
) -> None:
    targets = load_targets()
    queue = load_queue()
    if song_id is not None:
        index = next(
            (i for i, target in enumerate(targets) if target.song_id == song_id),
            None,
        )
        if index is None:
            raise ValueError(f"{song_id} is not in targets.json")
        queue.current_song_id = song_id
        queue.next_index = index
        save_queue(queue)
    print(
        f"Autogenerator targets: {len(targets)} songs "
        f"({', '.join(sorted({target.album for target in targets}))})"
    )
    print("Loading the approved-map anchor…")
    session = Session(targets, queue, install=install)
    server = ThreadingHTTPServer((host, port), Handler)
    server.session = session  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(f"Autogenerator UI: {url}")
    print("Thumbs down / up grade the current map and continue the queue.")
    print("Progress is saved after every candidate and every grade. Ctrl+C stops the server.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping autogenerator UI.")
    finally:
        server.server_close()
