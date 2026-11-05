# KaraokeMachine Autogenerator

A standalone neural autogenerator for KaraokeMachine maps. It reads the local
song library, learns from one explicitly approved finished map, writes playable
draft candidates, and updates its policy from your thumbs-up / thumbs-down
grades.

This folder is independent of the Rust app. Nothing here is compiled into
KaraokeMachine, and the app does not need to know it exists.

## Data policy

**Only maps listed in [`approved.json`](approved.json) are used for training.**

The allowlist currently contains exactly one finished map:

- **Man in the Box** — Alice in Chains — `yt-H1i5HaQV9ck/maps/Legacy.json`

The Salve Regina and Tequila maps are **not** used. No draft map is used. The
loader rejects drafts and never scans the library for extra training data.

## What it does

1. Decodes the song's vocal/guide track with `ffmpeg`.
2. Extracts frame-level pitch, volume, spectral onset and lyric-alignment
   features.
3. Proposes note onsets from vocal onsets, word starts and voiced segments.
4. A small neural policy scores each proposal and predicts pitch correction,
   onset correction and note duration.
5. The generated map is written as a playable draft:
   - `assets/songs/<song-id>/maps/AI-Candidate.json`
   - An archive copy under `automap/candidates/<song-id>/`
6. You listen and press thumbs down or thumbs up. The grade is converted to a
   reward, the policy is updated with policy gradient plus a supervised anchor,
   and the next candidate is generated immediately.

## Setup

Python 3.10+ and `ffmpeg` are required. This checkout already has a working
virtual environment at `automap/.venv`.

```bash
cd automap
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For CPU-only PyTorch:

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Train

```bash
cd automap
.venv/bin/python -m automap train --epochs 60
```

The first run extracts and caches the vocal analysis. Training prints:

- how many approved notes have a matching onset proposal
- full-song reconstruction note F1
- held-out candidate F1
- the checkpoint path

The checkpoint is `automap/checkpoints/note_policy.pt`.

## Grade in the UI

```bash
cd automap
./start-ui.sh
# or: .venv/bin/python -m automap ui
```

This starts `http://127.0.0.1:8765/` and opens it in your browser.

![Autogenerator grading window](docs/ui.png)

The window has:

- a large note preview in the middle
- red **BAD** thumbs-down on the left
- green **GOOD** thumbs-up on the right
- **NEXT** in the top-left to generate another candidate without updating

The vocal track plays under the preview. Use the play button or space bar;
click or drag the preview to seek. Grades update the model and immediately
generate the next candidate.

Keyboard shortcuts:

| Key | Action |
| --- | --- |
| `←` / `↓` | Bad map |
| `→` / `↑` | Good map |
| `N` | Next candidate without grading |
| `Space` | Play / pause |

If KaraokeMachine is open, restart it after a new candidate is written to load
`AI-Candidate` from the song's map picker.

## Other commands

```bash
cd automap

# List local songs, audio availability, map counts and approved status.
.venv/bin/python -m automap songs

# Generate one candidate without the browser UI.
.venv/bin/python -m automap generate --song yt-H1i5HaQV9ck --seed 1

# Grade in the terminal and keep generating candidates.
.venv/bin/python -m automap grade --song yt-H1i5HaQV9ck
```

Useful UI options:

```bash
.venv/bin/python -m automap ui --port 8770 --no-browser
.venv/bin/python -m automap ui --song <song-id>
.venv/bin/python -m automap ui --resume     # show a waiting candidate
```

## How grading works

The model is a Bernoulli policy over onset proposals plus Gaussian
distributions for pitch, onset and duration corrections.

- A grade of 5 (or `+`) has reward `+1`.
- A grade of 1 (or `-`) has reward `-1`.
- Grades 2 and 4 are `-0.5` and `+0.5`.
- Grade 3 or **NEXT** is neutral and does not update the weights.

Each update uses:

- REINFORCE with a moving-average reward baseline
- entropy regularization
- a supervised replay anchor on the approved Man in the Box map, so the policy
  does not forget the only good example

Candidates are draft map files. Once a candidate is good enough, open it in the
KaraokeMachine editor and submit it like any other map.

## Files and artifacts

| Path | Purpose |
| --- | --- |
| `approved.json` | the only training allowlist |
| `automap/audio.py` | ffmpeg decode, YIN pitch, RMS and onset analysis |
| `automap/candidates.py` | onset proposals and approved-map labels |
| `automap/model.py` | neural policy |
| `automap/train.py` | supervised training |
| `automap/generate.py` | candidate map writing and calibration |
| `automap/grade.py` | reward mapping and policy updates |
| `automap/ui.py` | local grading server |
| `automap/ui_assets/` | the thumbs-up/down window |
| `cache/` | extracted audio and onset features (ignored) |
| `checkpoints/` | trained policy (ignored) |
| `candidates/` | archived generated maps (ignored) |
| `state/` | pending candidate and feedback log (ignored) |

No audio is copied into this folder. The tool reads the licensed local library
and never embeds audio in a map.

## Known limitations

- With one approved song, the model learns that song's style and cannot fully
  generalize to other artists yet. The grading loop is how it adapts.
- The pitch tracker misses about 10 of the 184 approved notes; those notes are
  excluded from the supervised selection labels.
- The generated map is a starting point, not a replacement for submitting a
  reviewed map.
