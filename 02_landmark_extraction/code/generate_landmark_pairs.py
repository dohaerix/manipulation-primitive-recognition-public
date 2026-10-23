"""
Generates a paired raw-frame + landmark-annotated-frame for every processed
frame from a video (or a folder of images), using MediaPipe's own hand
detector and drawing utilities. Produces clean output: MediaPipe's standard
skeleton style (dots + connections), with no extra text/debug overlays.

Two files are saved per processed frame:
    frame_XXXX_raw.png         (unannotated, exactly as captured)
    frame_XXXX_landmarks.png   (same frame, MediaPipe skeleton overlay)

USAGE
-----
    # From a video file, sampling every 15th frame, stopping after 10 pairs:
    python3 generate_landmark_pairs.py --video my_recording.mp4 \
        --every-n-frames 15 --max-pairs 10 --out-dir landmark_pairs

    # From a folder of existing images instead:
    python3 generate_landmark_pairs.py --image-dir some_folder/ \
        --max-pairs 10 --out-dir landmark_pairs

Only frames where MediaPipe actually detects a hand are saved as pairs.

IMPORTANT -- MediaPipe version note (found during testing):
MediaPipe 0.10.14+ removed the classic `mp.solutions.hands` API entirely,
replacing it with a newer Tasks API that requires a separately-downloaded
.task model file. This script uses the classic API since it's almost
certainly what your existing project environment already has (your whole
pipeline depends on it). If the import below fails, run:
    pip show mediapipe
and if the version is 0.10.14 or newer, either downgrade
(pip install "mediapipe==0.10.13") or ask for a Tasks-API rewrite instead.
"""
from pathlib import Path
import argparse
import sys

import cv2

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # Colors: WHITE joints, GREEN connecting lines (swapped per feedback).
    # NOTE: cv2 uses BGR order, not RGB.
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
except AttributeError:
    print("ERROR: mp.solutions.hands is not available in your installed "
          "MediaPipe version.", file=sys.stderr)
    print("Run 'pip show mediapipe' to check your version. If it's 0.10.14 "
          "or newer, the classic solutions API has been removed -- run "
          "'pip install \"mediapipe==0.10.13\"' to get a version with it, "
          "or ask for a rewrite using the newer Tasks API instead.",
          file=sys.stderr)
    sys.exit(1)


def process_frame(frame_bgr, hands, out_dir, idx, upscale=1.0):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return False

    if upscale != 1.0:
        new_w = round(frame_bgr.shape[1] * upscale)
        new_h = round(frame_bgr.shape[0] * upscale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    raw_path = out_dir / f"frame_{idx:04d}_raw.png"
    cv2.imwrite(str(raw_path), frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # Scale drawing sizes to the actual frame width so joints/lines look
    # proportionate instead of using one fixed pixel size for every
    # resolution (a likely cause of the "thin/low quality" look reported).
    frame_w = frame_bgr.shape[1]
    scale = frame_w / 1280.0
    circle_radius = max(3, round(6 * scale))
    line_thickness = max(2, round(3 * scale))
    landmark_style = mp_drawing.DrawingSpec(color=WHITE, thickness=-1, circle_radius=circle_radius)
    connection_style = mp_drawing.DrawingSpec(color=GREEN, thickness=line_thickness)

    annotated = frame_bgr.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )
    landmark_path = out_dir / f"frame_{idx:04d}_landmarks.png"
    cv2.imwrite(str(landmark_path), annotated, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    print(f"  Saved pair {idx}: {raw_path.name} + {landmark_path.name} "
          f"({frame_w}x{frame_bgr.shape[0]}, radius={circle_radius}, thickness={line_thickness})")
    return True


def run_on_video(video_path, out_dir, every_n_frames, max_pairs, upscale=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                         min_detection_confidence=0.6) as hands:
        while cap.isOpened() and saved < max_pairs:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n_frames == 0:
                if process_frame(frame, hands, out_dir, saved, upscale):
                    saved += 1
            frame_idx += 1
    cap.release()
    return saved


def run_on_image_dir(image_dir, out_dir, max_pairs, upscale=1.0):
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    saved = 0
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.6) as hands:
        for p in paths:
            if saved >= max_pairs:
                break
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            if process_frame(frame, hands, out_dir, saved, upscale):
                saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to a video file")
    src.add_argument("--image-dir", help="Path to a folder of images instead")
    parser.add_argument("--every-n-frames", type=int, default=15,
                         help="For video input: only process every Nth frame (default 15)")
    parser.add_argument("--max-pairs", type=int, default=10,
                         help="Stop after this many successful pairs (default 10)")
    parser.add_argument("--out-dir", default="02_landmark_extraction/outputs/landmark_pairs")
    parser.add_argument("--upscale", type=float, default=1.0,
                         help="Resize factor applied before drawing/saving, e.g. 2.0 "
                              "doubles resolution via cubic interpolation for extra "
                              "sharpness (default 1.0, no resizing)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        saved = run_on_video(args.video, out_dir, args.every_n_frames, args.max_pairs, args.upscale)
    else:
        saved = run_on_image_dir(args.image_dir, out_dir, args.max_pairs, args.upscale)

    print(f"\nDone. Saved {saved} raw+landmark pairs to {out_dir}/")
    print("Open a few pairs and pick whichever raw/landmarks combination looks")
    print("best for the diagram -- no need to use all of them.")


if __name__ == "__main__":
    main()