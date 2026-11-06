# Man in the Box: generated vs. hand map

Generated without listening, using the teacher chart plus the current note
denoiser. The hand reference is `assets/songs/yt-H1i5HaQV9ck/maps/Legacy.json`.

| Metric | Current teacher model | Held-out model (MITB removed) | Hand map |
| --- | ---: | ---: | ---: |
| Notes | 185 | 178 | 184 |
| Matched notes | 132 | 65 | — |
| Note precision | 0.714 | 0.365 | — |
| Note recall | 0.717 | 0.353 | — |
| Note F1 | **0.715** | 0.359 | 1.000 |
| Matched onset MAE | **29 ms** | 70 ms | — |
| Pitch exactly equal | **97.7%** | 60.0% | — |
| Pitch within 1 semitone | **100%** | 100% | — |
| Duration ratio, median | **0.94×** | 0.90× | — |
| Placed words | 118 | 118 | 119 |
| Word-start MAE | 2134 ms | 2250 ms | — |
| Ribbon links | 98 | 74 | 69 |

The current model was trained on Man in the Box, so its 0.715 note F1 is a
training-set result. The held-out model, trained only on the five public USDB
note tracks, scores 0.359. That gap is the real generalization gap and is why
the other songs still need grading/hand correction.

The note lane is close. The weak part is **word placement**: the hand map places
lyrics independently of the note track, while the generated map still attaches
each word to a note. That is the next training target.

![Side-by-side comparison](mitb-comparison.png)
