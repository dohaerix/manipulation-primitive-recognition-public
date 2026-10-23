# Manipulation primitive recognition from hand landmarks: dataset and preprocessing

This repository has the dataset used in the paper, called OAMP-10 there (short for Object-Agnostic Manipulation Primitives, 10 classes), and the preprocessing code that turns a demonstration video into the per-frame motion features the recognition model reads.

It contains videos, landmark data, feature data, the train/val/test split, preprocessing code, and the paper's results as tables and figures. It does not contain model training or testing code or trained models.

## The 10 classes

Each class is a basic hand action, defined by how the hand moves and how it is shaped, not by which object is held. So "Grasp" is the same class whether the hand grasps a cup or a bottle.

| Label | Class | Clips | Avg. share of frames with the hand detected |
|---|---|---|---|
| 0 | Reach | 900 | 0.658 |
| 1 | Move | 900 | 0.572 |
| 2 | Pick | 900 | 0.771 |
| 3 | Place | 900 | 0.672 |
| 4 | Retract | 900 | 0.753 |
| 5 | Grasp | 900 | 0.657 |
| 6 | Tilt | 900 | 0.526 |
| 7 | Release | 900 | 0.609 |
| 8 | Mop | 900 | 0.921 |
| 9 | Give | 900 | 0.867 |

That makes 9,000 clips in total. On average the hand is detected in 70% of frames, and the remaining 30% are filled in during step 3.

## Folder structure and order

The folders are numbered in the order the data flows. Each step reads the output of the step before it.

```
01_raw_videos/                  STEP 1  the input: demonstration videos
02_landmark_extraction/         STEP 2  video → 21 hand points per frame
03_feature_extraction/          STEP 3  hand points → 196 motion features per frame
04_train_val_test_split/        STEP 4  which clips are for training, validation, testing
05_semantic_feature_groups/     STEP 5  the 196 features sorted into named groups
06_paper_results/               STEP 6  the paper's result tables and figures (images only, no code)
```

Inside each step folder:

- `code/` holds the script for that step (where there is one),
- `outputs/` holds what that step produced,
- `README.md` explains the step in more detail.

```
 01_raw_videos/Dataset_new/<Class>/*.mp4
        │
        │  02_landmark_extraction/code/extract_landmarks_v2.py
        ▼
 02_landmark_extraction/outputs/processed_landmarks_v2/<Class>/*.npz      (96 frames × 21 points)
        │
        │  03_feature_extraction/code/extract_handcentric_features.py
        ▼
 03_feature_extraction/outputs/processed_features_v1/<Class>/*.npz        (96 frames × 196 features)
        │
        ├──► 04_train_val_test_split/outputs/grouped_splits.csv           (train / val / test per clip)
        │
        │  05_semantic_feature_groups/code/step1b_refine_feature_groups.py
        ▼
 05_semantic_feature_groups/outputs/novelty_step1_feature_groups_v2/      (feature → group)
        │
        │  (model training and testing, described in the paper, not included)
        ▼
 06_paper_results/{figures,tables}/                                        (paper Tables I to IV, Figs. 1 to 4)
```

## What each step does

**Step 1, raw videos.** RGB videos of a single hand doing one action per clip. The full set is 54 GB, too big for GitHub, so this folder holds 88 sample clips that cover every class. For each class there is one recording together with its augmented copies (blurred, flickering, mirrored, jittered, noisy), plus four more original recordings. There is also one continuous video in which the hand does Reach, Pick, Move, Place and Retract in a row, without cuts.

**Step 2, landmark extraction.** For each video, 96 frames are picked at even spacing. MediaPipe Hands finds 21 points on the hand in each frame (wrist, knuckles, finger joints, fingertips). The output also records whether the hand was found in each frame, and stores a normalized copy of the points that is centred on the wrist and scaled by hand size. Every one of the 9,000 clips is included.

**Step 3, feature extraction.** Frames where the hand was not found are filled in from neighbouring frames, and the sequence is lightly smoothed. Then 196 numbers are computed for every frame, describing where the hand is and how it travels (path, direction, back-and-forth motion) and how the hand is shaped (finger distances, bone lengths, joint angles, palm direction). Every one of the 9,000 clips is included.

**Step 4, train/val/test split.** The clips are divided 70/15/15 (6,300 / 1,350 / 1,350). A recording and all its augmented copies are always kept on the same side, so the test set never contains a copy of a training clip.

**Step 5, semantic feature groups.** The features are sorted into 16 named groups by what they describe (for example `trajectory_displacement`, `finger_distance`, `joint_angle`). 188 features fall into these groups. The other 8 only describe tracking quality or time position, and are set aside.

**Step 6, paper results.** The four tables and four figures from the paper, cut directly from the paper PDF so they match it exactly, with the tables also typed out in `06_paper_results/README.md`. The headline result: VAGM-TCN reaches 88.71% accuracy and 89.57% macro-F1 on this dataset with 0.138M parameters and 0.0259 GFLOPs (Table I).

![Table I](06_paper_results/tables/table1_oamp10_comparison.png)

![Fig. 2](06_paper_results/figures/fig2_group_importance.png)

## Running the preprocessing

Run every command from the repository root. The scripts use paths relative to it.

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python 02_landmark_extraction/code/extract_landmarks_v2.py
python 03_feature_extraction/code/extract_handcentric_features.py
python 05_semantic_feature_groups/code/step1b_refine_feature_groups.py
```

Each script overwrites files with the same name in its `outputs/` folder. Step 2 only sees the 88 sample videos, so it rebuilds only those clips' landmark files and leaves the rest alone. Step 4 has no script here, because the split was made by a script that also trains models, which this repository leaves out. The split itself is complete.

To load one clip in Python:

```python
import numpy as np
z = np.load("03_feature_extraction/outputs/processed_features_v1/Give/1.npz", allow_pickle=True)
z["features"].shape   # (96, 196)
z["label"], z["valid_ratio"]
```

The `file` column in `04_train_val_test_split/outputs/grouped_splits.csv` starts with `processed_features_v1/`. Put `03_feature_extraction/outputs/` in front of it to get the path in this repository.

## Following one clip through every step

A sample video and its processed files share the same file name, so you can trace a single clip. For `Give/1`:

| Step | File |
|---|---|
| 1 | `01_raw_videos/Dataset_new/Give/1.avi` |
| 2 | `02_landmark_extraction/outputs/processed_landmarks_v2/Give/1.npz` |
| 3 | `03_feature_extraction/outputs/processed_features_v1/Give/1.npz` |
| 4 | the row for `processed_features_v1/Give/1.npz` in `04_train_val_test_split/outputs/grouped_splits.csv` |

`01_raw_videos/Dataset_new/SAMPLE_INDEX.tsv` lists all 88 sample videos.
