# Manipulation primitive recognition from hand landmarks: dataset, preprocessing and results

This repository goes with the paper *Learning the "What" and "How" of Manipulation for Human Demonstration-to-Robot Imitation*. A person shows a task to an ordinary camera, for example picking up a bottle and pouring it. The system works out two things:

- **What** was done: the ordered list of basic hand actions (Reach, Grasp, Pick, Tilt, ...).
- **How** it was done: the path the wrist followed.

A robot arm then repeats the task using those two pieces.

The repository contains the dataset (called OAMP-10 in the paper: Object-Agnostic Manipulation Primitives, 10 classes), all the preprocessing code that turns a video into model inputs, and the paper's results as tables and plots. It does **not** contain the model training or testing code, or trained models.

![Fig. 1: overview of the framework](06_paper_results/figures/fig1_pipeline_overview.png)

## How the whole system works

Fig. 1 shows the full system. Read it from the top left.

1. **Human demonstration (yellow box).** A person does the task once in front of a normal RGB camera. Nothing is attached to the hand.
2. **Hand landmarks.** In every frame, Google's MediaPipe Hands finds 21 points on the hand: the wrist, the knuckles, the finger joints and the fingertips. Each frame also gets a yes/no flag saying whether a hand was found at all. This flag matters later.
3. **Kinematic features.** The 21 points are turned into 196 numbers per frame that describe the motion and the hand's shape: how far and how fast the wrist has moved, whether it is going back and forth, how open the hand is, how bent each finger is, which way the palm faces. These are sorted into named groups. The 7 groups that matter most are kept, which leaves 122 numbers per frame.
4. **VAGM-TCN.** The recognition model reads 96 frames of those 122 numbers and says which of the 10 primitives it is. It is described in detail in the next section.
5. **Primitive sequence.** On a long, uncut video, the model looks at a 96-frame window, slides it forward 8 frames, and looks again. Neighbouring windows with the same answer are merged, which gives an ordered list such as Reach → Grasp → Pick → Tilt → Retract. This is the "what to do".
6. **Wrist trajectory (purple box).** At the same time, the wrist's path through the video is kept. This is the "how it was done".
7. **Robot execution (grey box).** The wrist path is moved from the camera's frame of reference into the robot's. The object is located with an object detector and the depth camera. The path is smoothed into a spline curve, and a model-predictive controller (MPC) drives a Kinova Gen3 arm along it while respecting speed and acceleration limits.

Steps 2 and 3 are fully in this repository, as code and as data for all 9,000 clips. Steps 4 to 7 are described here and in the paper, and their results are shown as plots and tables. Their code is not included.

## The recognition model: VAGM-TCN

VAGM-TCN stands for **validity-aware group-masked temporal convolutional network**. It is small (0.138 million parameters) and fast: it needs 0.0259 GFLOPs and 0.92 ms to classify one clip.

### The problem it solves

Hand tracking from a single camera is unreliable. In this dataset the hand is detected in only about 70% of frames on average. The rest are lost to blur, fast motion, or the hand turning away. Missing frames are filled in by interpolation, but filled-in frames are guesses.

A model trained on such data can also learn shortcuts. It can lean on one kind of cue, such as the hand's position in the image, and fall apart when that cue is noisy. VAGM-TCN is designed against both problems.

### What happens to one clip, step by step

The input is 96 frames × 122 features.

1. **Scale the features.** Each feature is standardized: its training-set mean is subtracted and the result divided by its standard deviation. These two statistics are computed on the training clips only and reused unchanged for validation, testing and live use.
2. **Group masking (training only).** For each training clip, each of the 7 feature groups is switched off (set to zero across all 96 frames) with some probability. If a clip happens to lose all 7, one is kept. This is like covering one of the model's senses at random during practice, so it learns to use all the cues together instead of relying on one.
   - **The "validity-aware" part.** The masking probability is multiplied by the share of frames in which the hand was actually detected. A clip with good tracking gets masked more. A clip with poor tracking, which is already hard, is masked less.
   - **The schedule.** No masking for the first 10 epochs, so the model first learns the basics. After that the base probability rises linearly, reaching 20% at epoch 80.
3. **Project.** A layer normalization, then a linear layer that maps the 122 numbers per frame to 64, then GELU and dropout.
4. **Temporal convolution.** Five residual blocks look along time. Each block has two 1-D convolutions with kernel size 3, each followed by batch normalization, GELU and 30% dropout, and adds its input back at the end (a residual connection).
   - The blocks use dilations 1, 2, 4, 8 and 16. The first block compares neighbouring frames, and the last compares frames 16 apart.
   - Stacked together, the five blocks see 125 frames, which is more than the whole 96-frame clip. So the model picks up both quick finger movements and slow arm motion, without a recurrent network.
5. **Attention pooling.** Each of the 96 frames gets a score: a small layer (64 → 64, tanh, → 1) estimates how informative that frame is. Frames where the hand was not detected are given zero weight, so the model never relies on a filled-in guess. The scores are turned into weights that add up to 1, and the frames are averaged with those weights into a single 64-number summary of the clip.
6. **Classify.** Layer normalization, dropout, and a linear layer to 10 outputs. The largest output is the predicted primitive.

Fig. 3 in `06_paper_results/` shows what the attention learns: it peaks when the fingers close for Grasp, when they open for Release, when the palm turns for Tilt, and when the motion reverses for Mop.

### Training settings

| Setting | Value |
|---|---|
| Input | 96 frames × 122 features (7 groups) |
| Hidden size | 64 |
| Convolution | 5 residual blocks, kernel 3, dilations 1, 2, 4, 8, 16, dropout 0.3 |
| Attention size | 64 |
| Loss | cross-entropy |
| Optimizer | AdamW, learning rate 3 × 10⁻⁴, weight decay 10⁻⁴ |
| Learning-rate schedule | halved when validation macro-F1 has not improved for 5 epochs |
| Batch size | 64 |
| Epochs | up to 80, early stopping after 15 epochs without improvement in validation macro-F1 |
| Group masking | maximum 20%, 10 warm-up epochs, scaled by each clip's detected-frame share |
| Size and speed | 137,918 parameters, 0.0259 GFLOPs per clip, 0.92 ms per clip |
| Hardware | Apple-silicon Mac (PyTorch `mps` backend) |

## Repository layout

The folders are numbered in the order the work happens. Each one reads what the one before it produced.

```
01_raw_videos/                     STEP 1  input videos (88 samples covering every class) + 1 uncut demonstration
02_landmark_extraction/            STEP 2  video → 21 hand points per frame          code + data for all 9,000 clips
03_feature_extraction/             STEP 3  hand points → 196 features per frame      code + data for all 9,000 clips
04_train_val_test_split/           STEP 4  which clips are train / validation / test  data
05_semantic_feature_groups/        STEP 5  196 features → 188 in 16 named groups     code + data
06_paper_results/                  STEP 6  the paper's tables and figures, cut from the PDF
07_model_testing_and_comparison/   STEP 7  comparison plots, training curves, t-SNE, continuous recognition
requirements.txt                           package versions for the preprocessing code
```

Each step folder has a `README.md` explaining it in detail. Steps 2, 3 and 5 also have a `code/` folder (the script) and an `outputs/` folder (what the script produced).

```
 01_raw_videos/Dataset_new/<Class>/*.mp4
        │  02_landmark_extraction/code/extract_landmarks_v2.py
        ▼
 02_landmark_extraction/outputs/processed_landmarks_v2/<Class>/*.npz      96 frames × 21 points + valid flag
        │  03_feature_extraction/code/extract_handcentric_features.py
        ▼
 03_feature_extraction/outputs/processed_features_v1/<Class>/*.npz        96 frames × 196 features
        │
        ├──► 04_train_val_test_split/outputs/grouped_splits.csv            train / val / test for every clip
        │
        │  05_semantic_feature_groups/code/step1b_refine_feature_groups.py
        ▼
 05_semantic_feature_groups/outputs/novelty_step1_feature_groups_v2/      feature → group (16 groups, 188 features)
        │
        │  group importance → keep 7 groups (122 features) → train VAGM-TCN → test → compare
        │  (described below; code not included)
        ▼
 06_paper_results/   and   07_model_testing_and_comparison/                tables, figures, plots
```

## Every step in detail

### Step 1. Record the demonstrations (`01_raw_videos/`)

Each clip shows a single hand doing one primitive. According to the paper, the clips were recorded with a monocular RGB camera, with 12 repetitions from 5 subjects in 3 indoor environments, using different objects and arrangements so that pose, grip and motion vary. That gave 180 original clips per primitive.

Each original was then turned into 4 more versions by offline augmentation. The file names show which one: `_blur`, `_flicker` (brightness flicker), `_flip` (mirrored), `_jitter` (camera shake), `_noise`. The result is 900 clips per class and 9,000 in total.

The full video set is 54 GB, which is too large for GitHub, so this folder holds 88 sample clips that cover every class, plus one uncut demonstration video. Steps 2 onward include all 9,000 clips.

### Step 2. Find the hand in every frame (`02_landmark_extraction/`)

Script: `extract_landmarks_v2.py`. For each video:

1. **Pick 96 frames.** It takes 96 frames evenly spaced from the first to the last, so every clip has the same length whatever its duration.
2. **Run MediaPipe Hands** on each frame, with one hand and minimum detection and tracking confidence 0.5.
3. **Check the frame is valid** (line 86 of the script). If the frame was read and a hand was found, the 21 (x, y, z) points are saved and the frame is marked valid (1). Otherwise zeros are saved and the frame is marked invalid (0).
4. **Save a normalized copy** of the points. The wrist point is subtracted from all 21 points, and the result is divided by the distance from the wrist to the middle-finger knuckle. This copy describes the hand's shape regardless of where it is in the image or how close it is to the camera.

The output is one `.npz` file per clip with `raw_landmarks` (96×63), `norm_landmarks` (96×63), `valid` (96), `valid_ratio`, `label` and `video_name`. `landmark_audit.csv` checks every file. A second script, `generate_landmark_pairs.py`, only draws example images of the detected skeleton.

### Step 3. Turn hand points into motion features (`03_feature_extraction/`)

Script: `extract_handcentric_features.py`. For each landmark file:

1. **Fill gaps.** For invalid frames, each coordinate is filled by straight-line interpolation between the nearest valid frames. The filled frames are recorded in `interpolated`.
2. **Smooth.** A 3-frame moving average removes jitter.
3. **Compute 196 features per frame**, from two views of the hand:
   - **Position and motion,** from the image-space points: where the wrist, palm centre and hand centre are; displacement from the first frame and direction of travel; path length and straightness; speed, acceleration and jerk; path curvature; and back-and-forth measures (zero crossings, reversal strength, local speed variation) over a 9-frame window.
   - **Hand shape,** from the normalized points: fingertip-to-palm and fingertip-to-fingertip distances, finger bone lengths, the bend at each finger joint, finger curl, palm direction and how much it turns between frames, hand openness and compactness, and bounding-box and palm area.
4. **Bookkeeping features.** 8 of the 196 describe tracking quality or time position, not motion: `valid`, `interpolated`, `valid_ratio`, `missing_run_length`, `time01`, `phase_early`, `phase_middle`, `phase_late`. They are kept in the files but never given to the model.

### Step 4. Split into training, validation and test (`04_train_val_test_split/`)

Each clip is given a group name made from its class and file name, with the augmentation ending removed, so `Give/17.avi`, `Give/17_blur.mp4` and `Give/17_flip.mp4` all belong to group `Give::17`. Within each class, whole groups are assigned to train, validation or test. A seeded search (seed 42) picks the assignment closest to 70 / 15 / 15.

The result is 6,300 / 1,350 / 1,350 clips (630 / 135 / 135 per class), with no group in two sets. This matters: if a blurred copy of a test clip were in the training set, the test score would be inflated.

### Step 5. Sort the features into named groups (`05_semantic_feature_groups/`)

Script: `step1b_refine_feature_groups.py`. It drops the 8 bookkeeping features (196 → 188) and assigns each remaining feature to one of 16 groups by what it measures, for example `trajectory_displacement` (27 features), `finger_distance` (23) or `joint_angle` (16). Grouping lets the later steps treat a whole family of related cues as one unit.

### Step 6. Measure which groups matter, and keep 7 (no code here)

A first VAGM-TCN was trained on all 188 features. Then each group was switched off in turn, and the drop in macro-F1 was measured (paper Fig. 2).

The 7 groups whose removal hurt most were kept:

| Group | Features |
|---|---|
| raw global shape | 12 |
| oscillation/reversal | 18 |
| trajectory/displacement | 27 |
| joint angle | 16 |
| palm orientation | 4 |
| finger distance | 23 |
| bone length | 22 |

That is 122 features. The other 9 groups held 66 features: velocity, acceleration, jerk, finger curl, bounding box, curvature, openness, compactness and palm area. Removing any one of them lowered macro-F1 by less than 0.1 point or not at all, and for four of them (finger curl, openness, jerk, curvature) macro-F1 even went up slightly.

### Step 7. Train and test the final VAGM-TCN (no code here)

The model described above was trained on the 6,300 training clips, using the 122 features. The epoch with the best validation macro-F1 was kept. It was then tested once on the 1,350 test clips: 88.71% accuracy and 89.57% macro-F1 (paper Table I).

### Step 8. Compare with other models (no code here)

Eight other models were trained on the same split:

- **Graph models** (ST-GCN, CTR-GCN, HD-GCN, DA-GCN) take the 21 landmarks as a hand skeleton.
- **Sequence models** (BiLSTM, Vanilla TCN, MS-TCN++, ASFormer) take the same 122 features as VAGM-TCN.

For each model, parameters, GFLOPs per clip and inference time were measured as well. See `06_paper_results/` (Table I) and `07_model_testing_and_comparison/`.

### Step 9. Test on a public benchmark, FPHA (no code here)

To check that the approach is not tied to this one dataset, the same features and all nine models were retrained and tested on FPHA (First-Person Hand Action, 45 classes) with a reconstructed 1:1 split. VAGM-TCN reached 89.66% accuracy and 89.10% macro-F1 with the lowest compute of all nine models (paper Table II).

### Step 10. Recognize primitives in uncut videos (no code here)

A 96-frame window slides along a long demonstration 8 frames at a time. Each window is classified, and neighbouring windows with the same label are merged into segments. This gives the ordered primitive sequence, while the wrist path over the same frames is kept. The perception side runs at about 15 frames per second, limited by the camera and hand tracking. See the plots in `07_model_testing_and_comparison/continuous_recognition/`.

### Step 11. Robot execution (no code here)

The robot side has four parts:

1. **Move to the robot's frame.** The wrist path is transformed from the camera's frame to the robot's using a calibrated camera-to-robot transform.
2. **Find the object.** Where a primitive needs an object, YOLOv8 finds it in the image and the depth image gives its 3-D position.
3. **Smooth the path.** The path is turned into a smooth cubic spline parameterized by arc length.
4. **Track it.** A receding-horizon MPC, solved with CasADi, makes a Kinova Gen3 arm follow the spline within speed and acceleration limits.

In 80 trials (10 demonstrators, 4 tasks, twice each), 66 succeeded. The MPC tracked the path with lower error than P, PD, the Kinova built-in controller and an LQR controller. See Tables III and IV and Fig. 4 in `06_paper_results/`.

## Running the preprocessing (steps 2, 3 and 5)

Use Python 3.10 and run everything from the repository root:

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # mediapipe 0.10.9, opencv-python, numpy, pandas, tqdm

python 02_landmark_extraction/code/extract_landmarks_v2.py            # videos → landmarks
python 03_feature_extraction/code/extract_handcentric_features.py     # landmarks → 196 features
python 05_semantic_feature_groups/code/step1b_refine_feature_groups.py   # features → 16 groups
```

Notes:

- `mediapipe` must be a version that still has `mp.solutions.hands`. 0.10.9 was used.
- Step 2 can only process the 88 sample videos included here. Its outputs for all 9,000 clips are already in the repository, so you can start from step 3.
- Each script overwrites files with the same name in its `outputs/` folder.
- Step 4 has no script here, because the split was made by a script that also trains models. The split itself is complete, and the method is described above.

To load one clip in Python:

```python
import numpy as np
z = np.load("03_feature_extraction/outputs/processed_features_v1/Give/1.npz", allow_pickle=True)
z["features"].shape          # (96, 196)
z["feature_names"][:5]       # names of the first 5 features
z["label"], z["valid_ratio"]
```

The `file` column in `04_train_val_test_split/outputs/grouped_splits.csv` starts with `processed_features_v1/`. Put `03_feature_extraction/outputs/` in front of it to get the path in this repository.

## Following one clip through every step

A sample video and its processed files share the same file name. For `Give/1`:

| Step | File |
|---|---|
| 1 | `01_raw_videos/Dataset_new/Give/1.avi` |
| 2 | `02_landmark_extraction/outputs/processed_landmarks_v2/Give/1.npz` |
| 3 | `03_feature_extraction/outputs/processed_features_v1/Give/1.npz` |
| 4 | the row for `processed_features_v1/Give/1.npz` in `04_train_val_test_split/outputs/grouped_splits.csv` |

`01_raw_videos/Dataset_new/SAMPLE_INDEX.tsv` lists all 88 sample videos.

## The 10 classes

Each class is a basic hand action, defined by how the hand moves and how it is shaped, not by which object is held. "Grasp" is the same class whether the hand grasps a cup or a bottle.

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

## Results in brief

| | OAMP-10 (this dataset) | FPHA (45 classes) |
|---|---|---|
| Accuracy | 88.71% | 89.66% |
| Macro-F1 | 89.57% | 89.10% |
| Parameters | 0.138M | 0.138M |
| GFLOPs per clip | 0.0259 | 0.0259 |

The full tables are in `06_paper_results/`, and the comparison plots are in `07_model_testing_and_comparison/`.
