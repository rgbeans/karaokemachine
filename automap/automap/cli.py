"""Command-line interface for the standalone autogenerator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import MODEL_PATH, SONGS_DIR
from .generate import generate_candidate, pending_path, print_generation
from .grade import parse_grade, print_pending, update_from_grade
from .maps import load_approved_tracks
from .queue import load_targets
from .train import train
from .ui import run_ui
from .word_model import train_word_model
from .teacher import train_teacher_model
from .ribbon_model import train_ribbon_model


def _approved_ids() -> set[str]:
    return {track.song_id for track in load_approved_tracks()}


def command_songs(_: argparse.Namespace) -> None:
    approved = _approved_ids()
    targets = {target.song_id for target in load_targets()}
    rows = []
    for song_dir in sorted(SONGS_DIR.iterdir()):
        song_file = song_dir / "song.json"
        if not song_file.exists():
            continue
        data = json.loads(song_file.read_text(encoding="utf-8"))
        audio = data.get("audio") or {}
        guide_name = audio.get("vocals") or audio.get("guide") or audio.get("backing")
        guide = song_dir / str(guide_name) if guide_name else None
        maps = sorted((song_dir / "maps").glob("*.json"))
        finals = 0
        drafts = 0
        for path in maps:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("draft"):
                drafts += 1
            else:
                finals += 1
        rows.append(
            (
                song_dir.name,
                str(data.get("title") or song_dir.name),
                str(data.get("artist") or ""),
                bool(guide and guide.exists()),
                finals,
                drafts,
                song_dir.name in approved,
                song_dir.name in targets,
            )
        )
    print(
        f"{'song id':24} {'title':38} {'audio':5} {'maps':>6} "
        f"{'training':>9} {'target':>7}"
    )
    for song_id, title, artist, has_audio, finals, drafts, is_approved, is_target in rows:
        print(
            f"{song_id:24} {title[:38]:38} {'yes' if has_audio else 'no':5} "
            f"{finals:2d}F/{drafts:2d}D {('yes' if is_approved else 'no'):>9} "
            f"{('yes' if is_target else 'no'):>7}"
        )
    print()
    print("Training uses only maps in automap/approved.json.")
    print("Generation targets are listed in automap/targets.json.")
    print(f"Target queue: {len(targets)} songs; Man in the Box is excluded.")


def command_train(args: argparse.Namespace) -> None:
    train(
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.lr,
        patience=args.patience,
        force_audio=args.force_audio,
    )


def command_train_words(args: argparse.Namespace) -> None:
    train_word_model(epochs=args.epochs, seed=args.seed)


def command_train_teacher(args: argparse.Namespace) -> None:
    train_teacher_model(epochs=args.epochs, seed=args.seed)


def command_train_ribbons(args: argparse.Namespace) -> None:
    train_ribbon_model(epochs=args.epochs, seed=args.seed)


def command_generate(args: argparse.Namespace) -> None:
    info = generate_candidate(
        song_id=args.song,
        seed=args.seed,
        install=not args.no_install,
        model_path=Path(args.model),
    )
    print_generation(info)


def command_ui(args: argparse.Namespace) -> None:
    run_ui(
        song_id=args.song,
        host=args.host,
        port=args.port,
        install=not args.no_install,
        open_browser=not args.no_browser,
    )


def command_grade(args: argparse.Namespace) -> None:
    approved = load_approved_tracks()
    song_id = args.song or load_targets()[0].song_id
    install = not args.no_install
    if args.new or not print_pending(song_id):
        info = generate_candidate(song_id=song_id, seed=args.seed, install=install)
        print_generation(info)
    anchor_data = None
    if not args.no_anchor:
        from .dataset import load_track

        anchor_data = [load_track(track) for track in approved]

    print()
    print("Grade the candidate after listening. 1-5 maps to negative/positive reward;")
    print("+/- are shortcuts, 0/s skips the update, q keeps it pending and exits.")
    while True:
        try:
            raw = input("Grade [1-5, +/-/0, s=skip, q=quit]: ")
        except (EOFError, KeyboardInterrupt):
            print()
            print(f"Candidate kept at {pending_path(song_id)}")
            return
        if raw.strip().lower() in {"q", "quit"}:
            print(f"Candidate kept at {pending_path(song_id)}")
            return
        grade = parse_grade(raw)
        if grade is None:
            print("Enter 1-5, +, -, 0/s, or q.")
            continue
        record = update_from_grade(
            song_id,
            grade,
            anchor_data=anchor_data,
            model_path=Path(args.model),
        )
        if record["reward"] == 0.0:
            print("Neutral grade: no weight update.")
        else:
            print(
                f"reward {record['reward']:+.1f}  advantage {record['advantage']:+.2f}  "
                f"loss {record['loss']:.3f}  model step {record['step']}"
            )
        seed = (
            random.randint(1, 1_000_000)
            if args.seed is None
            else args.seed + int(record["step"])
        )
        info = generate_candidate(song_id=song_id, seed=seed, install=install)
        print()
        print_generation(info)
        if args.once:
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automap",
        description="Neural karaoke-map autogenerator with human grading.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    songs = sub.add_parser("songs", help="list local songs and their map counts")
    songs.set_defaults(func=command_songs)

    train_parser = sub.add_parser("train", help="train on the approved map(s)")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=2e-3)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--force-audio", action="store_true", help="rebuild the audio cache")
    train_parser.set_defaults(func=command_train)

    word_parser = sub.add_parser(
        "train-words", help="calibrate word placements against public karaoke timings"
    )
    word_parser.add_argument("--epochs", type=int, default=2000)
    word_parser.add_argument("--seed", type=int, default=2)
    word_parser.set_defaults(func=command_train_words)

    teacher_parser = sub.add_parser(
        "train-teacher",
        help="train the teacher-note denoiser against finished/public note tracks",
    )
    teacher_parser.add_argument("--epochs", type=int, default=300)
    teacher_parser.add_argument("--seed", type=int, default=4)
    teacher_parser.set_defaults(func=command_train_teacher)

    ribbon_parser = sub.add_parser(
        "train-ribbons",
        help="train the explicit ribbon-link policy from the hand map",
    )
    ribbon_parser.add_argument("--epochs", type=int, default=600)
    ribbon_parser.add_argument("--seed", type=int, default=5)
    ribbon_parser.set_defaults(func=command_train_ribbons)

    generate_parser = sub.add_parser("generate", help="generate one draft candidate")
    generate_parser.add_argument("--song", help="song id; defaults to the first approved track")
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--model", default=str(MODEL_PATH))
    generate_parser.add_argument("--no-install", action="store_true", help="do not write AI-Candidate.json")
    generate_parser.set_defaults(func=command_generate)

    ui_parser = sub.add_parser("ui", help="open the local thumbs-up/down grading window")
    ui_parser.add_argument("--song", help="song id; defaults to the first approved track")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8765)
    ui_parser.add_argument("--no-browser", action="store_true")
    ui_parser.add_argument("--no-install", action="store_true")
    ui_parser.set_defaults(func=command_ui)

    grade_parser = sub.add_parser("grade", help="grade a candidate, reinforce, and generate the next")
    grade_parser.add_argument("--song", help="song id; defaults to the first approved track")
    grade_parser.add_argument("--seed", type=int)
    grade_parser.add_argument("--model", default=str(MODEL_PATH))
    grade_parser.add_argument("--no-install", action="store_true")
    grade_parser.add_argument("--new", action="store_true", help="discard/ignore any pending candidate")
    grade_parser.add_argument("--once", action="store_true", help="stop after one grade and next generation")
    grade_parser.add_argument("--no-anchor", action="store_true", help="skip the supervised replay anchor")
    grade_parser.set_defaults(func=command_grade)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as error:
        print(f"error: {error}")
        return 1
    return 0
