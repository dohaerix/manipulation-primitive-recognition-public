# Step 4: train / validation / test split

This step decides which clips are used for training, which for validation, and which for testing.

| Path | What it is |
|---|---|
| `outputs/grouped_splits.csv` | one row per feature file: `file, class, video_name, group_id, valid_ratio, split` |
| `outputs/grouped_splits_landmarks.csv` | the same split, pointing at the step-2 landmark files |
| `outputs/dataset_manifest.csv` | one row per clip with frame and feature counts and the label number |
| `outputs/label_map.json` | class name to label number (Reach 0 … Give 9) |
| `outputs/group_leakage_check.txt` | the check that no recording appears in two splits |

## How the split was made

1. Every clip gets a group name made from its file name, with augmentation endings removed and the class in front. For example, `Give/17.avi`, `Give/17_blur.mp4` and `Give/17_noise.mp4` all become `Give::17`.
2. Within each class, whole groups are assigned to train, validation or test. A seeded (seed 42) search picks the assignment that comes closest to 70 / 15 / 15.

Result: 6,300 / 1,350 / 1,350 clips (630 / 135 / 135 per class), 3,854 groups, and no group split across two sets. A test clip therefore never has a blurred or mirrored twin in the training set.

There is no script in this folder, because the split was made by a script that also trains models. To use the split, read `grouped_splits.csv` and select rows by the `split` column. To check it, `group_leakage_check.txt` lists the number of groups (3,854) and the number found in more than one split (0).

The `file` column starts with `processed_features_v1/`. Add `03_feature_extraction/outputs/` in front of it to get the path in this repository.
