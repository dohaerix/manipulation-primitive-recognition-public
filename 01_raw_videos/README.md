# Step 1: raw videos

The input to the whole pipeline: RGB videos of a single hand doing one action.

| Path | What it is |
|---|---|
| `Dataset_new/<Class>/` | 88 sample clips across the 10 classes (199 MB). The full set of 9,000 clips is 54 GB and is not on GitHub. |
| `Dataset_new/SAMPLE_INDEX.tsv` | every sample clip, its class, and its split (train / val / test) |
| `continuous_demonstration/Reach pick move place retract.mp4` | one uncut video (1920×1080, about 60 fps, 466 frames) in which the hand does Reach, Pick, Move, Place and Retract in a row |

## How the samples were chosen

For each class:

1. One recording together with its augmented copies. File-name suffixes say which augmentation was applied: `_blur`, `_flicker`, `_flip` (mirrored), `_jitter` (camera shake), `_noise`. All five exist for most classes. A few classes have only some of them.
2. Four more original recordings.

Some classes also contain clips cut to a fixed length, named like `3s_12.mp4` or `4s_3.mp4`.

## Next step

`02_landmark_extraction/` reads `Dataset_new/<Class>/*`.
