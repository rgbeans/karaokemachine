"""Local browser UI: note preview, thumbs grading and the reinforcement loop."""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import FEEDBACK_PATH, PACKAGE_DIR
from .dataset import TrackData, load_track
from .generate import generate_candidate, pending_path
from .grade import load_pending, update_from_grade
from .maps import load_approved_tracks, load_song

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
    def __init__(self, song_id: str, install: bool = True, resume: bool = False) -> None:
        self.song_id = song_id
        self.install = install
        self.song = load_song(song_id)
        self.anchor_data: list[TrackData] = [
            load_track(track) for track in load_approved_tracks()
        ]
        self.lock = threading.Lock()
        self.state: dict[str, Any] | None = None
        self.last_reward: float | None = None
        self.last_grade: int | None = None
        self.grades = self._previous_grades()
        if resume and load_pending(song_id) is not None:
            pending = load_pending(song_id)
            assert pending is not None
            candidate = Path(str(pending["archive"][0]))
            info = {
                "song": self.song,
                "archive": candidate,
                "installed": Path(str(pending["installed"][0])) if str(pending["installed"][0]) else None,
                "notes": int(pending["note_count"][0]),
                "placed_words": 0,
                "mean_probability": 0.0,
                "model_step": int(pending["model_step"][0]),
            }
            self._set_state(info)
        else:
            self.generate()

    def _previous_grades(self) -> int:
        if not FEEDBACK_PATH.exists():
            return 0
        count = 0
        try:
            for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if record.get("song_id") == self.song_id:
                    count += 1
                    self.last_reward = record.get("reward")
                    self.last_grade = record.get("grade")
        except OSError:
            pass
        return count

    def _set_state(self, info: dict[str, Any]) -> None:
        path = info.get("installed") or info.get("archive")
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        self.state = {
            "song_id": self.song_id,
            "title": self.song.title,
            "artist": self.song.artist,
            "duration": self.song.duration,
            "audio_url": "/api/audio",
            "notes": document.get("notes") or [],
            "lines": document.get("lines") or [],
            "placedWords": document.get("placedWords") or [],
            "low": int(document.get("low") or 0),
            "high": int(document.get("high") or 0),
            "note_count": len(document.get("notes") or []),
            "candidate_id": str(info["archive"]),
            "archive": str(info["archive"]),
            "installed": str(info["installed"]) if info.get("installed") else "",
            "model_step": int(info.get("model_step") or 0),
            "grades": self.grades,
            "last_reward": self.last_reward,
            "last_grade": self.last_grade,
        }

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        info = generate_candidate(self.song_id, seed=seed, install=self.install)
        self._set_state(info)
        return self.state or {}

    def grade(self, grade: int, candidate_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            pending = load_pending(self.song_id)
            if pending is None:
                raise RuntimeError("no candidate is waiting for a grade; generate a new one")
            archive = str(pending["archive"][0])
            if candidate_id and candidate_id != archive:
                raise RuntimeError("that candidate was already replaced")
            record = update_from_grade(
                self.song_id, grade, anchor_data=self.anchor_data
            )
            self.grades += 1
            self.last_reward = float(record["reward"])
            self.last_grade = int(grade)
            self.generate()
            return self.state or {}

    def next_candidate(self) -> dict[str, Any]:
        with self.lock:
            # NEXT is a skip, not a grade: leave the model and baseline alone.
            pending_path(self.song_id).unlink(missing_ok=True)
            self.last_reward = 0.0
            self.last_grade = 3
            self.generate()
            return self.state or {}


class Handler(BaseHTTPRequestHandler):
    server_version = "KaraokeMachineAutomap/0.1"

    @property
    def session(self) -> Session:
        return self.server.session  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal focused on the grading loop.
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

    def _send_audio(self, head_only: bool = False) -> None:
        path = self.session.song.guide_path
        if not path.exists():
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

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/audio":
            self._send_audio(head_only=True)
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
        if path == "/api/audio":
            self._send_audio()
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
        if path not in {"/api/grade", "/api/next"}:
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
            if path == "/api/next":
                state = self.session.next_candidate()
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
    resume: bool = False,
) -> None:
    if song_id is None:
        song_id = load_approved_tracks()[0].song_id
    print(f"Loading {song_id} and the approved-map anchor…")
    session = Session(song_id, install=install, resume=resume)
    server = ThreadingHTTPServer((host, port), Handler)
    server.session = session  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(f"Autogenerator UI: {url}")
    print("Thumbs down / up grade the current map and generate the next one.")
    print("Press Ctrl+C to stop the server.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping autogenerator UI.")
    finally:
        server.server_close()
