import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp


DATASET_DIR = Path("01_raw_videos/Dataset_new")
OUTPUT_DIR = Path("02_landmark_extraction/outputs/processed_landmarks_v2")

CLASSES = [
    "Mop"
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

SEQ_LEN = 96

MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

mp_hands = mp.solutions.hands


def sample_frame_indices(total_frames, seq_len):
    indices = np.linspace(0, total_frames - 1, seq_len)
    return np.round(indices).astype(int)


def normalize_hand_landmarks(raw_landmarks):
    """
    raw_landmarks: 21 x 3
    returns wrist-centered, palm-scale-normalized landmarks.
    """
    pts = raw_landmarks.astype(np.float32).copy()

    wrist = pts[0].copy()
    pts = pts - wrist

    palm_scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / palm_scale

    return pts


def extract_video_landmarks(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = sample_frame_indices(total_frames, SEQ_LEN)

    raw_sequence = []
    norm_sequence = []
    valid_flags = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret or frame is None:
                raw_sequence.append(np.zeros((21, 3), dtype=np.float32))
                norm_sequence.append(np.zeros((21, 3), dtype=np.float32))
                valid_flags.append(0)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                raw_pts = []
                for lm in hand.landmark:
                    raw_pts.append([lm.x, lm.y, lm.z])

                raw_pts = np.array(raw_pts, dtype=np.float32)
                norm_pts = normalize_hand_landmarks(raw_pts)

                raw_sequence.append(raw_pts)
                norm_sequence.append(norm_pts)
                valid_flags.append(1)

            else:
                raw_sequence.append(np.zeros((21, 3), dtype=np.float32))
                norm_sequence.append(np.zeros((21, 3), dtype=np.float32))
                valid_flags.append(0)

    cap.release()

    raw_sequence = np.array(raw_sequence, dtype=np.float32)
    norm_sequence = np.array(norm_sequence, dtype=np.float32)
    valid_flags = np.array(valid_flags, dtype=np.float32)

    return {
        "raw_landmarks": raw_sequence.reshape(SEQ_LEN, 63),
        "norm_landmarks": norm_sequence.reshape(SEQ_LEN, 63),
        "valid": valid_flags,
        "valid_ratio": float(np.mean(valid_flags)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0

    for cls in CLASSES:
        input_dir = DATASET_DIR / cls
        output_dir = OUTPUT_DIR / cls
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            print(f"[WARNING] Missing folder: {input_dir}")
            continue

        videos = [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        ]

        print(f"\nProcessing class: {cls} | videos: {len(videos)}")

        for video_path in tqdm(videos):
            output_path = output_dir / f"{video_path.stem}.npz"

            if output_path.exists():
                continue

            data = extract_video_landmarks(video_path)

            if data is None:
                total_failed += 1
                continue

            np.savez_compressed(
                output_path,
                raw_landmarks=data["raw_landmarks"],
                norm_landmarks=data["norm_landmarks"],
                valid=data["valid"],
                valid_ratio=data["valid_ratio"],
                label=cls,
                video_name=video_path.name,
            )

            total_processed += 1

    print("\nLandmark extraction V2 complete.")
    print("-" * 50)
    print(f"Processed videos : {total_processed}")
    print(f"Failed videos    : {total_failed}")
    print(f"Output folder    : {OUTPUT_DIR}")
    print("-" * 50)


if __name__ == "__main__":
    main()