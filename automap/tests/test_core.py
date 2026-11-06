from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from automap.config import FEATURE_COUNT
from automap.grade import parse_grade
from automap.maps import Song
from automap.model import NotePolicy
from automap.notes import (
    assign_word_runs,
    build_map_document,
    build_ribbon_links,
    match_note_sets,
    place_words_from_notes,
)
from automap.policy import calibrate_selection_bias


class CoreTests(unittest.TestCase):
    def test_parse_grade(self) -> None:
        self.assertEqual(parse_grade("5"), 5)
        self.assertEqual(parse_grade("+"), 5)
        self.assertEqual(parse_grade("-"), 1)
        self.assertEqual(parse_grade("0"), 3)
        self.assertEqual(parse_grade("s"), 3)
        self.assertIsNone(parse_grade("maybe"))

    def test_selection_bias_calibrates(self) -> None:
        logits = torch.zeros(1000)
        bias = calibrate_selection_bias(logits, target_count=100)
        probabilities = torch.sigmoid(logits + bias)
        self.assertAlmostEqual(float(probabilities.sum()), 100.0, delta=2.0)

    def test_policy_shapes_and_ranges(self) -> None:
        model = NotePolicy(FEATURE_COUNT, hidden=32)
        features = torch.randn(2, FEATURE_COUNT, 17)
        with torch.no_grad():
            output = model(features)
            self.assertEqual(output["select_logit"].shape, (2, 17))
            self.assertEqual(output["pitch_offset"].shape, (2, 17))
            self.assertLessEqual(float(output["pitch_offset"].abs().max()), 3.0)
            self.assertLessEqual(float(output["onset_offset"].abs().max()), 0.08)
            self.assertLessEqual(float(output["log_scale"].abs().max()), 1.5)

    def test_checkpoint_preserves_feature_names(self) -> None:
        from automap.policy import Checkpoint, load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            checkpoint = Checkpoint(
                model=NotePolicy(FEATURE_COUNT, hidden=16),
                feature_mean=torch.zeros(FEATURE_COUNT).numpy(),
                feature_std=torch.ones(FEATURE_COUNT).numpy(),
                optimizer_state=None,
                step=7,
                reward_baseline=0.25,
                reward_count=3,
                hidden=16,
                path=path,
                feature_names=[f"feature_{index}" for index in range(FEATURE_COUNT)],
            )
            save_checkpoint(checkpoint)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded.feature_names, checkpoint.feature_names)
            self.assertEqual(loaded.step, 7)

    def test_map_document_has_playable_shape(self) -> None:
        song = Song(
            id="test",
            title="Test Song",
            artist="Tester",
            album="Tests",
            duration=10.0,
            guide_path=__import__("pathlib").Path("/tmp/none"),
            backing_path=None,
            lines=[
                {
                    "start": 1.0,
                    "end": 3.0,
                    "text": "hello world",
                    "words": [
                        {"text": "hello", "start": 1.0, "end": 1.8},
                        {"text": "world", "start": 2.0, "end": 2.8},
                    ],
                }
            ],
            raw={},
        )
        notes = [
            {"start": 1.0, "dur": 0.5, "midi": 60},
            {"start": 2.0, "dur": 0.5, "midi": 62},
        ]
        document = build_map_document(song, notes)
        self.assertTrue(document["draft"])
        self.assertEqual(len(document["notes"]), 2)
        self.assertEqual(len(document["placedWords"]), 2)
        self.assertEqual(document["format"], "karaokemachine-level")
        self.assertEqual(document["noteAlignment"], "policy")
        self.assertEqual(document["low"], 58)
        self.assertEqual(document["high"], 64)
        # Word pills follow the generated note that carries the word.
        self.assertAlmostEqual(document["placedWords"][0]["start"], 1.0)
        self.assertAlmostEqual(document["placedWords"][1]["start"], 2.0)

    def test_targets_are_the_other_albums_and_exclude_man_in_the_box(self) -> None:
        import json

        from automap.config import TARGETS_PATH

        payload = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        ids = {track["song_id"] for track in payload["tracks"]}
        self.assertEqual(len(ids), 30)
        self.assertNotIn("yt-H1i5HaQV9ck", ids)
        self.assertEqual(set(payload["albums"]), {"Dirt", "Facelift", "Jar of Flies"})

    def test_word_calibration_metrics_are_present(self) -> None:
        import json

        from automap import karaoke
        from automap.config import WORD_METRICS_PATH, WORD_OFFSETS_PATH

        metrics = json.loads(WORD_METRICS_PATH.read_text(encoding="utf-8"))
        offsets = json.loads(WORD_OFFSETS_PATH.read_text(encoding="utf-8"))
        source_ids = {
            source.song_id for source in (*karaoke.SOURCES, *karaoke.LINE_SOURCES)
        }
        self.assertEqual(set(offsets), source_ids)
        self.assertLess(metrics["overall_mae"], 0.1)
        self.assertLess(metrics["overall_line_mae"], 0.1)
        self.assertEqual(set(metrics["songs"]), set(offsets))
        for item in metrics["songs"].values():
            self.assertLess(item["mae"], 0.1)
        sunshine = metrics["songs"]["yt-2maHkdezdEc"]
        self.assertEqual(sunshine["level"], "line")
        self.assertEqual(sunshine["lrclib_id"], 819361)
        self.assertGreater(sunshine["lines"], 0)

    def test_line_level_targets_shift_a_whole_line(self) -> None:
        from automap import karaoke

        song = Song(
            id="line-test",
            title="Line Test",
            artist="Tester",
            album="Tests",
            duration=5.0,
            guide_path=__import__("pathlib").Path("/tmp/none"),
            backing_path=None,
            lines=[
                {
                    "start": 1.0,
                    "end": 3.0,
                    "text": "hello world",
                    "words": [
                        {"text": "hello", "start": 1.0, "end": 1.8},
                        {"text": "world", "start": 2.0, "end": 2.8},
                    ],
                }
            ],
            raw={},
        )
        targets = karaoke._map_line_targets(
            song, [{"start": 1.0, "text": "hello world"}], offset=0.5
        )
        self.assertEqual(len(targets), 2)
        self.assertEqual({target.level for target in targets}, {"line"})
        for target in targets:
            self.assertAlmostEqual(target.target_time - target.local_time, 0.5)

    def test_public_note_calibration_sources(self) -> None:
        from automap import karaoke, note_calibration

        for source in (*karaoke.SOURCES, *karaoke.DIRECT_NOTE_SOURCES):
            alignment = note_calibration.alignment_for_song(source.song_id)
            self.assertIsNotNone(alignment)
            self.assertEqual(alignment["usdb_id"], source.usdb_id)
            self.assertEqual(alignment["offset"], source.effective_note_offset)
        them_bones = karaoke.BY_SONG["yt-zTuD8k3JvxQ"]
        self.assertEqual(them_bones.effective_note_offset, 3.75)
        self.assertEqual(
            karaoke.TRAINING_NOTE_BY_SONG["yt-G23iLGhh9lo"].effective_note_offset,
            -0.15,
        )

    def test_ribbon_links_cover_legato_notes(self) -> None:
        notes = [
            {"start": 1.0, "dur": 0.5, "midi": 60},
            {"start": 1.7, "dur": 0.5, "midi": 62},
            {"start": 3.0, "dur": 0.5, "midi": 64},
        ]
        links = build_ribbon_links(notes)
        self.assertEqual(links, [{"from": 0, "to": 1, "fromSide": "right", "toSide": "left"}])

    def test_word_runs_keep_a_melisma_together(self) -> None:
        song = Song(
            id="word-runs",
            title="Word Runs",
            artist="Tester",
            album="Tests",
            duration=5.0,
            guide_path=__import__("pathlib").Path("/tmp/none"),
            backing_path=None,
            lines=[
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "We chase misprinted lies",
                    "words": [
                        {"text": "We", "start": 0.0, "end": 1.0},
                        {"text": "chase", "start": 1.0, "end": 2.0},
                        {"text": "misprinted", "start": 2.0, "end": 3.0},
                        {"text": "lies", "start": 3.0, "end": 4.0},
                    ],
                }
            ],
            raw={},
        )
        notes = [
            {"start": 0.0, "dur": 0.2, "midi": 60},
            {"start": 0.25, "dur": 0.2, "midi": 62},
            {"start": 0.5, "dur": 0.2, "midi": 61},
            {"start": 0.75, "dur": 0.2, "midi": 63},
            {"start": 1.1, "dur": 0.4, "midi": 55},
            {"start": 2.1, "dur": 0.4, "midi": 54},
            {"start": 3.1, "dur": 0.4, "midi": 53},
        ]
        placed = assign_word_runs(song, notes)
        self.assertEqual(placed[0]["text"], "We")
        self.assertEqual(sum(1 for note in notes if note.get("word") == "We"), 4)
        self.assertEqual(placed[1]["text"], "chase")

    def test_words_follow_the_note_track(self) -> None:
        song = Song(
            id="word-notes",
            title="Word Notes",
            artist="Tester",
            album="Tests",
            duration=4.0,
            guide_path=__import__("pathlib").Path("/tmp/none"),
            backing_path=None,
            lines=[
                {
                    "start": 0.8,
                    "end": 3.5,
                    "text": "hello world",
                    "words": [
                        {"text": "hello", "start": 1.0, "end": 1.8},
                        {"text": "world", "start": 2.0, "end": 2.8},
                    ],
                }
            ],
            raw={},
        )
        notes = [
            {"start": 1.1, "dur": 0.5, "midi": 60},
            {"start": 2.2, "dur": 0.5, "midi": 62},
        ]
        placed = place_words_from_notes(song, notes)
        self.assertEqual([item["start"] for item in placed], [1.1, 2.2])

    def test_teacher_metrics_improve_on_the_teacher_chart(self) -> None:
        import json

        from automap.teacher import TEACHER_METRICS_PATH

        metrics = json.loads(TEACHER_METRICS_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(metrics["songs"]), 5)
        for item in metrics["songs"].values():
            self.assertGreater(item["f1"], item["teacher_f1"])

    def test_usdb_matches_are_loaded(self) -> None:
        from automap import karaoke

        self.assertGreaterEqual(len(karaoke.NOTE_SOURCES), 15)
        self.assertIn("yt-G23iLGhh9lo", karaoke.TRAINING_NOTE_BY_SONG)
        self.assertNotIn("yt-G23iLGhh9lo", karaoke.NOTE_BY_SONG)
        self.assertIn("yt-qgaRVvAKoqQ", karaoke.NOTE_BY_SONG)
        self.assertLess(
            karaoke.NOTE_BY_SONG["yt-qgaRVvAKoqQ"].effective_note_offset, 1.0
        )

    def test_ribbon_policy_metrics_are_present(self) -> None:
        import json

        from automap.ribbon_model import RIBBON_METRICS_PATH

        metrics = json.loads(RIBBON_METRICS_PATH.read_text(encoding="utf-8"))
        self.assertGreater(metrics["f1"], 0.5)
        self.assertGreater(metrics["positive"], 20)

    def test_word_features_match_the_model_contract(self) -> None:
        from automap import karaoke, word_model

        song = Song(
            id=karaoke.SOURCES[0].song_id,
            title="Test",
            artist="Tester",
            album="Tests",
            duration=2.0,
            guide_path=__import__("pathlib").Path("/tmp/none"),
            backing_path=None,
            lines=[],
            raw={},
        )
        line = {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"text": "hello", "start": 0.2, "end": 0.8}],
        }
        audio = {
            "fps": 50.0,
            "midi": np.full(100, 60.0, dtype=np.float32),
            "rms": np.full(100, 0.1, dtype=np.float32),
            "flux": np.zeros(100, dtype=np.float32),
            "voiced": np.ones(100, dtype=bool),
        }
        features = word_model.word_features(
            song, audio, line, 0, line["words"][0], 0, 1, karaoke.SOURCES[0]
        )
        self.assertEqual(len(features), len(word_model.FEATURE_NAMES))

        line_features = word_model.word_features(
            song, audio, line, 0, line["words"][0], 0, 1, karaoke.LINE_SOURCES[0]
        )
        self.assertEqual(len(line_features), len(word_model.FEATURE_NAMES))
        self.assertNotEqual(features[-1], line_features[-1])

    def test_note_matching(self) -> None:
        truth = [{"start": 1.0, "midi": 60}, {"start": 2.0, "midi": 62}]
        predicted = [
            {"start": 1.05, "midi": 60},
            {"start": 2.02, "midi": 61},
            {"start": 4.0, "midi": 64},
        ]
        precision, recall, f1 = match_note_sets(predicted, truth)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 1.0)
        self.assertGreater(f1, 0.7)


if __name__ == "__main__":
    unittest.main()
