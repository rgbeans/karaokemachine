"""Generate a playable draft map from the learned policy."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import load_or_extract
from .candidates import load_or_build_candidates
from .config import (
    CANDIDATE_DIR,
    MODEL_PATH,
    SONGS_DIR,
    STATE_DIR,
    TEACHER_MODEL_PATH,
    ensure_dirs,
)
from .maps import Song, flatten_words, load_song, write_json
from .notes import build_map_document, build_notes
from .policy import load_checkpoint, sample_actions


def _ensure_word_coverage(
    song: Song,
    candidates,
    actions: dict[str, np.ndarray],
    notes: list[dict],
) -> list[dict]:
    """Guarantee at least one note for each timed lyric word when possible."""
    words = flatten_words(song.lines)
    if not words:
        return notes
    for _ in range(3):
        changed = False
        for word in words:
            covered = any(
                note["start"] < word.end + 0.15
                and note["start"] + note["dur"] > word.start - 0.15
                for note in notes
            )
            if covered:
                continue
            mask = (
                (candidates.starts >= word.start - 0.30)
                & (candidates.starts <= word.end + 0.30)
                & (actions["selected"] == 0)
            )
            options = np.where(mask)[0]
            if options.size == 0:
                continue
            best = int(options[int(np.argmax(actions["select_prob"][options]))])
            actions["selected"][best] = 1
            changed = True
        if not changed:
            break
        notes = build_notes(candidates, actions, song_duration=song.duration)

    # If NMS still dropped a forced proposal, insert a gap-sized note at the
    # word start. This keeps intros and quiet lyric lines from being empty.
    notes.sort(key=lambda note: float(note["start"]))
    for word in words:
        covered = any(
            note["start"] < word.end + 0.15
            and note["start"] + note["dur"] > word.start - 0.15
            for note in notes
        )
        if covered:
            continue
        mask = (candidates.starts >= word.start - 0.30) & (
            candidates.starts <= word.end + 0.30
        )
        options = np.where(mask)[0]
        if options.size == 0:
            continue
        best = int(options[int(np.argmax(actions["select_prob"][options]))])
        previous_end = max(
            (float(note["start"]) + float(note["dur"]) for note in notes if note["start"] < word.start),
            default=0.0,
        )
        next_start = min(
            (float(note["start"]) for note in notes if note["start"] > word.start),
            default=song.duration,
        )
        start = max(float(word.start), previous_end + 0.02)
        if start >= next_start - 0.08:
            continue
        duration = min(
            4.0,
            max(0.15, float(candidates.durations[best]) * float(np.exp(actions["log_scale"][best]))),
            next_start - start - 0.02,
        )
        if duration < 0.08:
            continue
        midi = int(
            np.clip(
                round(float(candidates.midis[best]) + float(actions["pitch_offset"][best])),
                24,
                96,
            )
        )
        notes.append(
            {
                "start": round(start, 3),
                "dur": round(duration, 3),
                "midi": midi,
                "vel": 0.8,
                "word": None,
                "wordIndex": None,
                "line": None,
                "ribbon": None,
            }
        )
        notes.sort(key=lambda note: float(note["start"]))
    return notes


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
    if song_id is None:
        from .queue import load_targets

        song_id = load_targets()[0].song_id
    from .maps import load_approved_tracks

    approved_ids = {track.song_id for track in load_approved_tracks()}
    if song_id in approved_ids:
        raise RuntimeError(
            "that song already has an approved map; generation is disabled for it"
        )
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2**63)
    song = load_song(song_id)
    audio = load_or_extract(song)

    from .note_calibration import alignment_for_song as note_alignment_for_song
    from .note_calibration import public_notes

    calibrated_notes = public_notes(song, audio)
    teacher_result = None
    if not calibrated_notes and model_path == MODEL_PATH:
        from . import teacher

        teacher_result = teacher.generate_teacher_notes(song, audio, seed=seed)

    checkpoint = None
    step = 0
    if teacher_result is not None:
        notes, candidates, actions, runtime = teacher_result
        if len(notes) == 0:
            raise RuntimeError("the teacher model produced no notes")
        note_alignment = "teacher"
        step = runtime.step
        note_source = {
            "source": "Teacher melody + denoiser",
            "label": "TEACHER + DENOISER",
            "model_step": step,
            "teacher_notes": len(teacher.load_or_build_suggestions(song)),
        }
        words_from_notes = True
        word_alignment = "notes"
        action_model_path = str(TEACHER_MODEL_PATH)
    else:
        checkpoint = load_checkpoint(model_path)
        candidates = load_or_build_candidates(song, audio)
        if len(candidates) == 0:
            raise RuntimeError(
                f"{song.title}: no candidates were extracted from the vocal track"
            )
        if candidates.features.shape[1] != checkpoint.feature_mean.size:
            raise RuntimeError(
                "the model was trained with a different feature layout; retrain it"
            )
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
        notes = _ensure_word_coverage(song, candidates, actions, notes)
        if len(notes) == 0:
            raise RuntimeError("the policy produced no notes; lower the threshold or retrain")
        step = checkpoint.step
        action_model_path = str(model_path)
        if calibrated_notes:
            notes = calibrated_notes
            note_alignment = "usdb"
            note_source = {
                **(note_alignment_for_song(song.id) or {}),
                "label": "USDB CALIBRATED",
            }
            words_from_notes = True
            word_alignment = "notes"
        else:
            from .config import WORD_MODEL_PATH
            from .word_model import alignment_for_song as word_alignment_for_song

            note_alignment = "policy"
            note_source = None
            words_from_notes = False
            word_alignment = (
                "calibrated"
                if word_alignment_for_song(song.id) is not None
                and WORD_MODEL_PATH.exists()
                else "lyrics"
            )
    document = build_map_document(
        song,
        notes,
        audio,
        word_alignment=word_alignment,
        note_alignment=note_alignment,
        note_source=note_source,
        words_from_notes=words_from_notes,
    )
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
            "model_step": step,
            "model_path": action_model_path,
            "note_alignment": note_alignment,
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
        model_step=np.array([step]),
        model_path=np.array([action_model_path]),
        note_alignment=np.array([note_alignment]),
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
        "model_step": step,
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
