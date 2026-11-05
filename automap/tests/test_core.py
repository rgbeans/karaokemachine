from __future__ import annotations

import unittest

import numpy as np
import torch

from automap.config import FEATURE_COUNT
from automap.grade import parse_grade
from automap.maps import Song
from automap.model import NotePolicy
from automap.notes import build_map_document, match_note_sets
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
        self.assertEqual(set(offsets), {source.song_id for source in karaoke.SOURCES})
        self.assertLess(metrics["overall_mae"], 0.1)
        self.assertEqual(set(metrics["songs"]), set(offsets))
        for item in metrics["songs"].values():
            self.assertLess(item["mae"], 0.1)

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
