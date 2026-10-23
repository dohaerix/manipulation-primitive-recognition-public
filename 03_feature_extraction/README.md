# Step 3: feature extraction

This step turns the 21 hand points per frame into 196 motion and hand-shape numbers per frame.

| Path | What it is |
|---|---|
| `code/extract_handcentric_features.py` | the feature script |
| `outputs/processed_features_v1/<Class>/<clip>.npz` | features for all 9,000 clips |
| `outputs/feature_names_v1.json` | the 196 feature names, in order |
| `outputs/feature_extraction_summary.csv` | one row per clip: shape, how many frames were filled in, status |

## What the script does

1. **Fill gaps.** For frames where the hand was not found, each coordinate is filled by straight-line interpolation between the nearest frames where it was found. Which frames were filled is recorded in `interpolated`.
2. **Smooth.** A 3-frame moving average removes jitter.
3. **Compute features** from two views of the hand:
   - Image-space points, which keep position: where the wrist, palm centre and hand centre are, how far they have moved since the first frame, path length, how straight the path is, speed, acceleration, jerk, curvature, and back-and-forth (oscillation) measures over a 9-frame window.
   - Normalized points, which ignore position and size: fingertip distances, bone lengths, joint angles, finger curl, palm direction and how much it turns, how open or compact the hand is.

8 of the 196 features (`valid`, `interpolated`, `valid_ratio`, `missing_run_length`, `time01`, `phase_early`, `phase_middle`, `phase_late`) describe tracking quality or time position, not motion.

## Output file contents

| Key | Shape | Meaning |
|---|---|---|
| `features` | (96, 196) | the features |
| `feature_names` | (196,) | same order as `feature_names_v1.json` |
| `interpolated` | (96,) | 1 where a frame was filled in |
| `raw_landmarks`, `norm_landmarks`, `valid`, `valid_ratio`, `label`, `video_name` | | copied from step 2 |

Load with `np.load(path, allow_pickle=True)`.

## Run

From the repository root: `python 03_feature_extraction/code/extract_handcentric_features.py`

## Next step

`04_train_val_test_split/` lists these files by split, and `05_semantic_feature_groups/` groups the feature columns.
