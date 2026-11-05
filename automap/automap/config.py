"""Shared paths, constants and feature-column definitions."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
AUTOMAP_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = AUTOMAP_DIR.parent

SONGS_DIR = PROJECT_ROOT / "assets" / "songs"
APPROVED_PATH = AUTOMAP_DIR / "approved.json"
TARGETS_PATH = AUTOMAP_DIR / "targets.json"
CACHE_DIR = AUTOMAP_DIR / "cache"
CHECKPOINT_DIR = AUTOMAP_DIR / "checkpoints"
CANDIDATE_DIR = AUTOMAP_DIR / "candidates"
STATE_DIR = AUTOMAP_DIR / "state"

MODEL_PATH = CHECKPOINT_DIR / "note_policy.pt"
PENDING_PATH = STATE_DIR / "pending.npz"
QUEUE_PATH = STATE_DIR / "queue.json"
FEEDBACK_PATH = STATE_DIR / "feedback.jsonl"

# Audio analysis frame rate and candidate proposal rate.
FPS = 50
CANDIDATE_FPS = 20
SAMPLE_RATE = 22_050
FRAME_SIZE = 1024

FEATURE_NAMES = (
    "start_frac",
    "log_duration",
    "midi_norm",
    "confidence_mean",
    "rms_mean",
    "rms_start",
    "onset_start",
    "onset_local_max",
    "onset_dist",
    "pitch_slope",
    "pitch_std",
    "voiced_ratio",
    "midi_dev_local",
    "prev_gap",
    "next_gap",
    "word_dist",
    "inside_word",
    "word_len",
    "word_duration",
    "line_frac",
    "line_pos",
    "line_start_dist",
    "line_end_dist",
    "next_onset_dist",
    "midi_vs_median",
    "rms_ratio",
    "is_word_start",
    "is_onset_peak",
)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_COUNT = len(FEATURE_NAMES)

# Columns that scale with tempo in feature-space augmentation. The target
# onset offset also scales; the target duration ratio does not.
TIME_FEATURES = (
    "onset_dist",
    "prev_gap",
    "next_gap",
    "word_dist",
    "word_duration",
    "line_start_dist",
    "line_end_dist",
    "next_onset_dist",
)
# ``pitch_slope`` is semitones per second, so it divides by the tempo factor.
PITCH_SLOPE_FEATURE = "pitch_slope"

# Action noise used during sampling. Positive grades reinforce the sampled
# actions; negative grades push the policy away from them.
PITCH_SIGMA = 0.55
ONSET_SIGMA = 0.028
DURATION_SIGMA = 0.18
SELECT_TEMPERATURE = 1.0

# Grading map. 3 is neutral and performs no update.
REWARD_BY_GRADE = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}


def ensure_dirs() -> None:
    for path in (CACHE_DIR, CHECKPOINT_DIR, CANDIDATE_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
