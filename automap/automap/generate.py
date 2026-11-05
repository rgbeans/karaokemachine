"""Generate a playable draft map from the learned policy."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import load_or_extract
from .candidates import load_or_build_candidates
from .config import CANDIDATE_DIR, MODEL_PATH, SONGS_DIR, STATE_DIR, ensure_dirs
from .maps import Song, load_song, write_json
from .notes import build_map_document, build_notes
from .policy import load_checkpoint, sample_actions


def _slug(value: str) -> str:
    output = []
    for character in value.lower():
        if character.isalnum():
            output.append(character)
        elif not output or output[-1] != "-":
            output.append("-")
    return "".join(output).strip("-") or "map"


def pending_path(song_id: str) -> Path:
    return STATE_DIR / f"pending-{song_id}.npz"


def candidate_install_path(song_id: str) -> Path:
    return SONGS_DIR / song_id / "maps" / "AI-Candidate.json"


def generate_candidate(
    song_id: str | None = None,
    seed: int | None = None,
    install: bool = True,
    model_path: Path = MODEL_PATH,
) -> dict[str, object]:
    ensure_dirs()
    checkpoint = load_checkpoint(model_path)
    if song_id is None:
        from .maps import load_approved_tracks

        song_id = load_approved_tracks()[0].song_id
    song = load_song(song_id)
    audio = load_or_extract(song)
    candidates = load_or_build_candidates(song, audio)
    if len(candidates) == 0:
        raise RuntimeError(f"{song.title}: no candidates were extracted from the vocal track")
    if candidates.features.shape[1] != checkpoint.feature_mean.size:
        raise RuntimeError("the model was trained with a different feature layout; retrain it")
    normalized = (
        candidates.features - checkpoint.feature_mean
    ) / checkpoint.feature_std
    desired_notes = max(12, int(song.duration * 0.70))
    # Over-propose: NMS and the density cap remove overlapping notes.
    target_candidates = max(desired_notes, int(desired_notes * 2.5))
    actions = sample_actions(
        checkpoint.model,
        normalized,
        seed=seed,
        target_notes=target_candidates,
    )
    notes = build_notes(candidates, actions, song_duration=song.duration)

    # Keep early maps playable even when the policy is still conservative.
    if len(notes) < 8:
        order = np.argsort(-actions["select_prob"])
        forced = np.zeros_like(actions["selected"])
        forced[order[:24]] = 1
        actions["selected"] = forced
        notes = build_notes(candidates, actions, song_duration=song.duration)
    if len(notes) == 0:
        raise RuntimeError("the policy produced no notes; lower the threshold or retrain")

    document = build_map_document(song, notes)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    archive = CANDIDATE_DIR / song.id / f"{timestamp}-{_slug(song.title)}.json"
    write_json(archive, document)
    write_json(
        archive.with_suffix(".meta.json"),
        {
            "song_id": song.id,
            "title": song.title,
            "artist": song.artist,
            "seed": seed,
            "model_step": checkpoint.step,
            "notes": len(notes),
            "placedWords": len(document["placedWords"]),
            "duration": song.duration,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    installed: Path | None = None
    if install:
        installed = candidate_install_path(song.id)
        write_json(installed, document)

    np.savez_compressed(
        pending_path(song.id),
        features=candidates.features,
        starts=candidates.starts,
        durations=candidates.durations,
        midis=candidates.midis,
        select_logit=actions["select_logit"],
        select_prob=actions["select_prob"],
        selection_bias=actions["selection_bias"],
        selected=actions["selected"],
        pitch_offset=actions["pitch_offset"],
        onset_offset=actions["onset_offset"],
        log_scale=actions["log_scale"],
        song_id=np.array([song.id]),
        title=np.array([song.title]),
        archive=np.array([str(archive)]),
        installed=np.array([str(installed) if installed else ""]),
        model_step=np.array([checkpoint.step]),
        seed=np.array([-1 if seed is None else seed]),
        note_count=np.array([len(notes)]),
    )
    return {
        "song": song,
        "archive": archive,
        "installed": installed,
        "notes": len(notes),
        "placed_words": len(document["placedWords"]),
        "mean_probability": float(np.mean(actions["select_prob"])),
        "model_step": checkpoint.step,
    }


def print_generation(info: dict[str, object]) -> None:
    song: Song = info["song"]  # type: ignore[assignment]
    print(f"Candidate: {song.title} — {song.artist}")
    print(f"  notes: {info['notes']}  lyric labels: {info['placed_words']}")
    print(f"  archive: {info['archive']}")
    if info["installed"]:
        print(f"  installed: {info['installed']}")
        print("  Restart KaraokeMachine, choose AI-Candidate for this song, and listen.")
    else:
        print("  not installed into the app; use --install to write AI-Candidate.json")
    print(f"  mean selection probability: {float(info['mean_probability']):.3f}")
