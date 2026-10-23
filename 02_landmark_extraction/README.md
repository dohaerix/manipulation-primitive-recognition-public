# Step 2: landmark extraction

This step turns each video into 21 hand points per frame, using MediaPipe Hands.

| Path | What it is |
|---|---|
| `code/extract_landmarks_v2.py` | the extraction script |
| `code/generate_landmark_pairs.py` | saves a few frames next to the same frames with the landmarks drawn on |
| `outputs/processed_landmarks_v2/<Class>/<clip>.npz` | landmarks for all 9,000 clips |
| `outputs/landmark_audit.csv` | one row per clip: array shapes, number of frames where the hand was missing, status |
| `outputs/landmark_pairs/` | 5 example frames, raw and with landmarks |

## What the script does

1. It picks 96 frames at even spacing from the first frame to the last, so every clip has the same length.
2. It runs MediaPipe Hands on each of those frames (one hand, detection and tracking confidence 0.5).
3. If a hand is found, it saves the 21 points (x, y, z) and marks the frame valid. If not, it saves zeros and marks the frame invalid.
4. It also saves a normalized copy: each frame's points minus the wrist point, divided by the distance from the wrist to the middle-finger knuckle. This removes where the hand is in the image and how big it looks.

## Output file contents

| Key | Shape | Meaning |
|---|---|---|
| `raw_landmarks` | (96, 63) | 21 points × (x, y, z) per frame, image coordinates |
| `norm_landmarks` | (96, 63) | the normalized copy |
| `valid` | (96,) | 1 if the hand was found in that frame |
| `valid_ratio` | scalar | share of frames with the hand found |
| `label`, `video_name` | text | class and source video |

## Run

From the repository root: `python 02_landmark_extraction/code/extract_landmarks_v2.py`

## Next step

`03_feature_extraction/` reads `outputs/processed_landmarks_v2/`.
