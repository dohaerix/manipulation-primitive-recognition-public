"""
extract_handcentric_features.py

Next-stage feature extractor for object-independent, hand-centric primitive action labeling.

Input folder:
    processed_landmarks_v2/<class_name>/*.npz

Expected input .npz fields:
    raw_landmarks   : (96, 63)
    norm_landmarks  : (96, 63)
    valid           : (96,)
    valid_ratio     : scalar
    label           : class string
    video_name      : original video filename

Output folder:
    processed_features_v1/<class_name>/*.npz

Output .npz fields:
    features        : (96, D) frame-level hand-centric features
    feature_names   : list of D feature names
    raw_landmarks   : original raw landmarks, (96, 63)
    norm_landmarks  : original normalized landmarks, (96, 63)
    valid           : original valid mask, (96,)
    label           : class string
    video_name      : original video filename
    valid_ratio     : scalar
    interpolated    : (96,) 1 where a missing frame was filled by interpolation

Notes:
    - Missing frames are linearly interpolated for feature computation.
    - The original valid mask is preserved and included as a feature.
    - Do NOT treat valid_ratio as proof of action class; use it for reliability analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# CONFIG
# =========================
INPUT_DIR = Path("02_landmark_extraction/outputs/processed_landmarks_v2")
OUTPUT_DIR = Path("03_feature_extraction/outputs/processed_features_v1")
SUMMARY_CSV = Path("03_feature_extraction/outputs/feature_extraction_summary.csv")
FEATURE_NAMES_JSON = Path("03_feature_extraction/outputs/feature_names_v1.json")

CLASSES = [
    "Reach", "Move", "Pick", "Place", "Retract",
    "Grasp", "Tilt", "Release", "Mop", "Give"
]

SEQ_LEN = 96
N_LANDMARKS = 21
EPS = 1e-6
SMOOTH_WINDOW = 3  # small temporal smoothing after interpolation
LOCAL_WINDOW = 9   # local oscillation window, should be odd

# MediaPipe hand landmark ids
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

PALM_IDS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGERTIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
THUMB_TO_FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

FINGER_CHAINS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# =========================
# NUMERIC UTILITIES
# =========================
def safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def unit_vector(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (safe_norm(x, axis=axis, keepdims=True) + EPS)


def temporal_diff(x: np.ndarray) -> np.ndarray:
    """First-order frame difference with first frame copied as zero diff."""
    d = np.zeros_like(x, dtype=np.float32)
    d[1:] = x[1:] - x[:-1]
    return d


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between vectors u and v in radians. Supports arrays ending with xyz."""
    dot = np.sum(u * v, axis=-1)
    den = safe_norm(u, axis=-1) * safe_norm(v, axis=-1) + EPS
    c = np.clip(dot / den, -1.0, 1.0)
    return np.arccos(c).astype(np.float32)


def polygon_area_xy(points: np.ndarray) -> np.ndarray:
    """Shoelace area over x-y plane. points: T x K x 3."""
    x = points[:, :, 0]
    y = points[:, :, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))


def moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32)
    pad = window // 2
    xpad = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        out[t] = xpad[t:t + window].mean(axis=0)
    return out


def interpolate_missing(seq: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill missing frames by linear interpolation for feature computation.
    seq: T x 21 x 3
    valid: T, 1 for detected frame, 0 for missing
    Returns:
        filled sequence, interpolated flag
    """
    seq = seq.astype(np.float32).copy()
    valid = valid.astype(bool)
    T = seq.shape[0]
    interp_flag = (~valid).astype(np.float32)

    if valid.sum() == 0:
        return np.zeros_like(seq, dtype=np.float32), interp_flag

    if valid.sum() == 1:
        only = seq[valid][0]
        filled = np.repeat(only[None, :, :], T, axis=0).astype(np.float32)
        return filled, interp_flag

    t_all = np.arange(T)
    t_valid = t_all[valid]
    filled = seq.copy()

    flat = seq.reshape(T, -1)
    filled_flat = filled.reshape(T, -1)
    for j in range(flat.shape[1]):
        filled_flat[:, j] = np.interp(t_all, t_valid, flat[valid, j])

    return filled_flat.reshape(seq.shape).astype(np.float32), interp_flag


def missing_run_lengths(valid: np.ndarray) -> np.ndarray:
    valid = valid.astype(bool)
    out = np.zeros(valid.shape[0], dtype=np.float32)
    run = 0
    for i, ok in enumerate(valid):
        if not ok:
            run += 1
            out[i] = run
        else:
            run = 0
    return out


def rolling_count(arr: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(arr.astype(np.float32), (pad, pad), mode="edge")
    return np.array([padded[i:i + window].sum() for i in range(arr.shape[0])], dtype=np.float32)


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    return np.array([padded[i:i + window].std() for i in range(x.shape[0])], dtype=np.float32)


def rolling_range(x: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    return np.array([padded[i:i + window].max() - padded[i:i + window].min() for i in range(x.shape[0])], dtype=np.float32)


# =========================
# FEATURE COMPUTATION
# =========================
def add_features(cols: List[np.ndarray], names: List[str], prefix: str, values: np.ndarray, subnames: List[str] | None = None) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        cols.append(values[:, None])
        names.append(prefix)
    elif values.ndim == 2:
        cols.append(values)
        if subnames is None:
            for i in range(values.shape[1]):
                names.append(f"{prefix}_{i}")
        else:
            assert len(subnames) == values.shape[1], (prefix, values.shape, subnames)
            for s in subnames:
                names.append(f"{prefix}_{s}")
    else:
        raise ValueError(f"Unsupported feature shape for {prefix}: {values.shape}")


def compute_curvature(pos: np.ndarray) -> np.ndarray:
    vel = temporal_diff(pos)
    acc = temporal_diff(vel)
    cross = np.cross(vel, acc)
    return (safe_norm(cross, axis=1) / (safe_norm(vel, axis=1) ** 3 + EPS)).astype(np.float32)


def compute_joint_angles(norm_pts: np.ndarray) -> np.ndarray:
    angles = []
    for a, b, c in ANGLE_TRIPLETS:
        u = norm_pts[:, a] - norm_pts[:, b]
        v = norm_pts[:, c] - norm_pts[:, b]
        angles.append(angle_between(u, v))
    return np.stack(angles, axis=1).astype(np.float32)


def compute_finger_curl(norm_pts: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    curls = []
    names = []
    for finger, chain in FINGER_CHAINS.items():
        pts = norm_pts[:, chain]
        direct = safe_norm(pts[:, -1] - pts[:, 0], axis=1)
        bone_lengths = safe_norm(pts[:, 1:] - pts[:, :-1], axis=2).sum(axis=1)
        curl = 1.0 - direct / (bone_lengths + EPS)
        curl = np.clip(curl, 0.0, 1.0)
        curls.append(curl.astype(np.float32))
        names.append(finger)
    return np.stack(curls, axis=1).astype(np.float32), names


def compute_palm_normal(norm_pts: np.ndarray) -> np.ndarray:
    wrist = norm_pts[:, WRIST]
    index = norm_pts[:, INDEX_MCP]
    pinky = norm_pts[:, PINKY_MCP]
    v1 = index - wrist
    v2 = pinky - wrist
    normal = np.cross(v1, v2)
    return unit_vector(normal, axis=1).astype(np.float32)


def compute_oscillation_features(pos: np.ndarray, speed: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Local, threshold-free oscillation descriptors. These are continuous/count-like signals,
    not class decision rules.
    """
    vel = temporal_diff(pos)
    vel_unit = unit_vector(vel, axis=1)

    cos_adj = np.ones(pos.shape[0], dtype=np.float32)
    cos_adj[1:] = np.sum(vel_unit[1:] * vel_unit[:-1], axis=1)

    # Reversal evidence as a continuous signal: 0 means same direction, high means opposite direction.
    reversal_strength = np.clip((1.0 - cos_adj) / 2.0, 0.0, 1.0).astype(np.float32)
    rolling_reversal_strength = rolling_std(cos_adj, LOCAL_WINDOW)

    # Sign-change counts in velocity coordinates, local window.
    sign_v = np.sign(vel)
    sign_change = np.zeros_like(vel, dtype=np.float32)
    sign_change[1:] = (sign_v[1:] * sign_v[:-1] < 0).astype(np.float32)
    zero_cross_x = rolling_count(sign_change[:, 0], LOCAL_WINDOW)
    zero_cross_y = rolling_count(sign_change[:, 1], LOCAL_WINDOW)
    zero_cross_z = rolling_count(sign_change[:, 2], LOCAL_WINDOW)

    # Local path geometry.
    local_speed_std = rolling_std(speed, LOCAL_WINDOW)
    local_speed_range = rolling_range(speed, LOCAL_WINDOW)

    T = pos.shape[0]
    pad = LOCAL_WINDOW // 2
    pospad = np.pad(pos, ((pad, pad), (0, 0)), mode="edge")
    elongation_ratio = np.zeros(T, dtype=np.float32)
    local_tortuosity = np.zeros(T, dtype=np.float32)

    for t in range(T):
        window_pts = pospad[t:t + LOCAL_WINDOW]
        centered = window_pts - window_pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, LOCAL_WINDOW - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0.0))
        elongation_ratio[t] = eigvals[-1] / (eigvals.sum() + EPS)

        local_steps = safe_norm(window_pts[1:] - window_pts[:-1], axis=1).sum()
        net = safe_norm(window_pts[-1] - window_pts[0], axis=0)
        local_tortuosity[t] = local_steps / (net + EPS)

    feats = np.stack([
        reversal_strength,
        rolling_reversal_strength,
        zero_cross_x,
        zero_cross_y,
        zero_cross_z,
        local_speed_std,
        local_speed_range,
        elongation_ratio,
        local_tortuosity,
    ], axis=1).astype(np.float32)

    names = [
        "reversal_strength",
        "rolling_reversal_strength",
        "zero_cross_x",
        "zero_cross_y",
        "zero_cross_z",
        "local_speed_std",
        "local_speed_range",
        "elongation_ratio",
        "local_tortuosity",
    ]
    return feats, names


def compute_features(raw_flat: np.ndarray, norm_flat: np.ndarray, valid: np.ndarray, valid_ratio: float) -> Tuple[np.ndarray, List[str], np.ndarray]:
    raw = raw_flat.reshape(SEQ_LEN, N_LANDMARKS, 3).astype(np.float32)
    norm = norm_flat.reshape(SEQ_LEN, N_LANDMARKS, 3).astype(np.float32)
    valid = valid.astype(np.float32)

    raw_filled, interp_flag = interpolate_missing(raw, valid)
    norm_filled, _ = interpolate_missing(norm, valid)

    raw_filled = moving_average(raw_filled, SMOOTH_WINDOW)
    norm_filled = moving_average(norm_filled, SMOOTH_WINDOW)

    cols: List[np.ndarray] = []
    names: List[str] = []

    T = raw.shape[0]
    time01 = np.linspace(0.0, 1.0, T, dtype=np.float32)
    phase_early = (time01 < 1 / 3).astype(np.float32)
    phase_middle = ((time01 >= 1 / 3) & (time01 < 2 / 3)).astype(np.float32)
    phase_late = (time01 >= 2 / 3).astype(np.float32)

    # Reliability/mask features.
    add_features(cols, names, "valid", valid)
    add_features(cols, names, "interpolated", interp_flag)
    add_features(cols, names, "valid_ratio", np.full(T, valid_ratio, dtype=np.float32))
    add_features(cols, names, "missing_run_length", missing_run_lengths(valid))
    add_features(cols, names, "time01", time01)
    add_features(cols, names, "phase", np.stack([phase_early, phase_middle, phase_late], axis=1), ["early", "middle", "late"])

    # =========================
    # Raw trajectory features
    # =========================
    wrist = raw_filled[:, WRIST]
    palm_center = raw_filled[:, PALM_IDS].mean(axis=1)
    centroid = raw_filled.mean(axis=1)

    add_features(cols, names, "raw_wrist", wrist, ["x", "y", "z"])
    add_features(cols, names, "raw_palm_center", palm_center, ["x", "y", "z"])
    add_features(cols, names, "raw_centroid", centroid, ["x", "y", "z"])

    xy = raw_filled[:, :, :2]
    bbox_min = xy.min(axis=1)
    bbox_max = xy.max(axis=1)
    bbox_wh = bbox_max - bbox_min
    bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]
    add_features(cols, names, "raw_bbox", bbox_wh, ["width", "height"])
    add_features(cols, names, "raw_bbox_area", bbox_area)

    # =========================
    # Kinematics
    # =========================
    for prefix, pos in [("wrist", wrist), ("palm", palm_center), ("centroid", centroid)]:
        vel = temporal_diff(pos)
        acc = temporal_diff(vel)
        jerk = temporal_diff(acc)
        speed = safe_norm(vel, axis=1)
        acc_mag = safe_norm(acc, axis=1)
        jerk_mag = safe_norm(jerk, axis=1)
        direction = unit_vector(vel, axis=1)
        disp = pos - pos[0:1]
        disp_mag = safe_norm(disp, axis=1)
        path_length = np.cumsum(speed).astype(np.float32)
        straightness = (disp_mag / (path_length + EPS)).astype(np.float32)
        curvature = compute_curvature(pos)

        add_features(cols, names, f"{prefix}_vel", vel, ["x", "y", "z"])
        add_features(cols, names, f"{prefix}_speed", speed)
        add_features(cols, names, f"{prefix}_acc", acc, ["x", "y", "z"])
        add_features(cols, names, f"{prefix}_acc_mag", acc_mag)
        add_features(cols, names, f"{prefix}_jerk", jerk, ["x", "y", "z"])
        add_features(cols, names, f"{prefix}_jerk_mag", jerk_mag)
        add_features(cols, names, f"{prefix}_direction", direction, ["x", "y", "z"])
        add_features(cols, names, f"{prefix}_disp", disp, ["x", "y", "z"])
        add_features(cols, names, f"{prefix}_disp_mag", disp_mag)
        add_features(cols, names, f"{prefix}_path_length", path_length)
        add_features(cols, names, f"{prefix}_straightness", straightness)
        add_features(cols, names, f"{prefix}_curvature", curvature)

    # =========================
    # Normalized hand-shape features
    # =========================
    norm_palm_center = norm_filled[:, PALM_IDS].mean(axis=1)
    fingertips = norm_filled[:, FINGERTIP_IDS]

    tip_to_palm = safe_norm(fingertips - norm_palm_center[:, None, :], axis=2)
    add_features(cols, names, "tip_to_palm", tip_to_palm, ["thumb", "index", "middle", "ring", "pinky"])
    hand_openness = tip_to_palm.mean(axis=1)
    add_features(cols, names, "hand_openness", hand_openness)
    add_features(cols, names, "hand_openness_vel", temporal_diff(hand_openness[:, None])[:, 0])
    add_features(cols, names, "hand_openness_acc", temporal_diff(temporal_diff(hand_openness[:, None]))[:, 0])
    add_features(cols, names, "tip_to_palm_std", tip_to_palm.std(axis=1))

    # Fingertip spread: pairwise distances among fingertips.
    pair_dists = []
    pair_names = []
    for i in range(len(FINGERTIP_IDS)):
        for j in range(i + 1, len(FINGERTIP_IDS)):
            d = safe_norm(fingertips[:, i] - fingertips[:, j], axis=1)
            pair_dists.append(d)
            pair_names.append(f"{i}_{j}")
    fingertip_pair_dists = np.stack(pair_dists, axis=1).astype(np.float32)
    add_features(cols, names, "fingertip_dist", fingertip_pair_dists, pair_names)
    add_features(cols, names, "mean_fingertip_spread", fingertip_pair_dists.mean(axis=1))

    # Thumb-finger distances.
    thumb = norm_filled[:, THUMB_TIP]
    thumb_dists = []
    thumb_names = []
    for fid, fname in zip(THUMB_TO_FINGER_TIPS, ["index", "middle", "ring", "pinky"]):
        d = safe_norm(thumb - norm_filled[:, fid], axis=1)
        thumb_dists.append(d)
        thumb_names.append(fname)
    thumb_dists = np.stack(thumb_dists, axis=1).astype(np.float32)
    add_features(cols, names, "thumb_to_finger", thumb_dists, thumb_names)
    add_features(cols, names, "thumb_to_finger_min", thumb_dists.min(axis=1))
    add_features(cols, names, "thumb_to_finger_mean", thumb_dists.mean(axis=1))
    add_features(cols, names, "thumb_index_vel", temporal_diff(thumb_dists[:, 0:1])[:, 0])
    add_features(cols, names, "thumb_index_acc", temporal_diff(temporal_diff(thumb_dists[:, 0:1]))[:, 0])

    # Palm area and compactness.
    palm_area = polygon_area_xy(norm_filled[:, PALM_IDS])
    add_features(cols, names, "norm_palm_area", palm_area)
    norm_xy = norm_filled[:, :, :2]
    norm_bbox_wh = norm_xy.max(axis=1) - norm_xy.min(axis=1)
    norm_bbox_area = norm_bbox_wh[:, 0] * norm_bbox_wh[:, 1]
    norm_bbox_diag = safe_norm(norm_bbox_wh, axis=1)
    add_features(cols, names, "norm_bbox", norm_bbox_wh, ["width", "height"])
    add_features(cols, names, "norm_bbox_area", norm_bbox_area)
    add_features(cols, names, "hand_compactness", hand_openness / (norm_bbox_diag + EPS))

    # Finger curl and dynamics.
    curls, curl_names = compute_finger_curl(norm_filled)
    add_features(cols, names, "finger_curl", curls, curl_names)
    mean_curl = curls.mean(axis=1)
    add_features(cols, names, "mean_finger_curl", mean_curl)
    add_features(cols, names, "mean_finger_curl_vel", temporal_diff(mean_curl[:, None])[:, 0])
    add_features(cols, names, "mean_finger_curl_acc", temporal_diff(temporal_diff(mean_curl[:, None]))[:, 0])

    # Joint angles and flexion.
    joint_angles = compute_joint_angles(norm_filled)
    angle_names = [f"a{a}_b{b}_c{c}" for a, b, c in ANGLE_TRIPLETS]
    add_features(cols, names, "joint_angle", joint_angles, angle_names)
    mean_flexion = joint_angles.mean(axis=1)
    add_features(cols, names, "mean_joint_angle", mean_flexion)
    add_features(cols, names, "mean_joint_angle_vel", temporal_diff(mean_flexion[:, None])[:, 0])

    # Bone lengths and ratios.
    bone_lengths = []
    bone_names = []
    for a, b in HAND_EDGES:
        bone_lengths.append(safe_norm(norm_filled[:, b] - norm_filled[:, a], axis=1))
        bone_names.append(f"{a}_{b}")
    bone_lengths = np.stack(bone_lengths, axis=1).astype(np.float32)
    add_features(cols, names, "bone_length", bone_lengths, bone_names)
    add_features(cols, names, "bone_length_mean", bone_lengths.mean(axis=1))
    add_features(cols, names, "bone_length_std", bone_lengths.std(axis=1))

    # =========================
    # Palm orientation features
    # =========================
    palm_normal = compute_palm_normal(norm_filled)
    add_features(cols, names, "palm_normal", palm_normal, ["x", "y", "z"])
    normal_vel = temporal_diff(palm_normal)
    normal_acc = temporal_diff(normal_vel)
    add_features(cols, names, "palm_normal_vel", normal_vel, ["x", "y", "z"])
    add_features(cols, names, "palm_angular_speed", safe_norm(normal_vel, axis=1))
    add_features(cols, names, "palm_normal_acc", normal_acc, ["x", "y", "z"])
    add_features(cols, names, "palm_angular_acc_mag", safe_norm(normal_acc, axis=1))

    # Orientation angle change per frame.
    orientation_change = np.zeros(T, dtype=np.float32)
    orientation_change[1:] = angle_between(palm_normal[1:], palm_normal[:-1])
    add_features(cols, names, "palm_orientation_change", orientation_change)

    # =========================
    # Oscillation / local periodicity proxies
    # =========================
    wrist_speed = safe_norm(temporal_diff(wrist), axis=1)
    osc, osc_names = compute_oscillation_features(wrist, wrist_speed)
    add_features(cols, names, "wrist_osc", osc, osc_names)

    palm_speed = safe_norm(temporal_diff(palm_center), axis=1)
    osc_palm, osc_palm_names = compute_oscillation_features(palm_center, palm_speed)
    add_features(cols, names, "palm_osc", osc_palm, osc_palm_names)

    features = np.concatenate(cols, axis=1).astype(np.float32)

    # Replace accidental NaN/inf with finite values.
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    if features.shape[0] != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} frames, got {features.shape[0]}")
    if features.shape[1] != len(names):
        raise ValueError(f"Feature/name mismatch: {features.shape[1]} vs {len(names)}")

    return features, names, interp_flag.astype(np.float32)
def compute_features_from_sequences(raw_landmarks: np.ndarray, norm_landmarks: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Realtime/API wrapper for feature computation.

    This function is used by realtime_best_models.py.

    Parameters
    ----------
    raw_landmarks:
        np.ndarray with shape (96, 63) or (96, 21, 3)
    norm_landmarks:
        np.ndarray with shape (96, 63) or (96, 21, 3)
    valid:
        np.ndarray with shape (96,)

    Returns
    -------
    features:
        np.ndarray with shape (96, 196)

    Important
    ---------
    This calls the SAME compute_features(...) function used during offline
    feature extraction, so realtime inference uses the same feature definitions
    and feature order as training.
    """
    raw_landmarks = np.asarray(raw_landmarks, dtype=np.float32)
    norm_landmarks = np.asarray(norm_landmarks, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32).reshape(-1)

    if raw_landmarks.shape == (SEQ_LEN, N_LANDMARKS, 3):
        raw_flat = raw_landmarks.reshape(SEQ_LEN, 63)
    elif raw_landmarks.shape == (SEQ_LEN, 63):
        raw_flat = raw_landmarks
    else:
        raise ValueError(
            f"raw_landmarks must have shape (96, 63) or (96, 21, 3), got {raw_landmarks.shape}"
        )

    if norm_landmarks.shape == (SEQ_LEN, N_LANDMARKS, 3):
        norm_flat = norm_landmarks.reshape(SEQ_LEN, 63)
    elif norm_landmarks.shape == (SEQ_LEN, 63):
        norm_flat = norm_landmarks
    else:
        raise ValueError(
            f"norm_landmarks must have shape (96, 63) or (96, 21, 3), got {norm_landmarks.shape}"
        )

    if valid.shape != (SEQ_LEN,):
        raise ValueError(f"valid must have shape (96,), got {valid.shape}")

    valid_ratio = float(np.mean(valid))

    features, feature_names, _interpolated = compute_features(
        raw_flat=raw_flat,
        norm_flat=norm_flat,
        valid=valid,
        valid_ratio=valid_ratio,
    )

    if features.shape != (SEQ_LEN, 196):
        raise ValueError(f"Expected realtime features shape (96, 196), got {features.shape}")

    # Safety check: make sure realtime feature order matches training feature order.
    if FEATURE_NAMES_JSON.exists():
        try:
            with open(FEATURE_NAMES_JSON, "r", encoding="utf-8") as fp:
                saved_names = json.load(fp)

            if list(saved_names) != list(feature_names):
                raise ValueError(
                    "Realtime feature-name order does not match feature_names_v1.json. "
                    "Do not run realtime inference until feature order is fixed."
                )

        except Exception as e:
            raise RuntimeError(f"Feature-name consistency check failed: {e}") from e

    return features.astype(np.float32)


# =========================
# MAIN
# =========================
def process_file(input_path: Path, output_path: Path) -> Dict[str, object]:
    data = np.load(input_path, allow_pickle=True)

    raw = data["raw_landmarks"].astype(np.float32)
    norm = data["norm_landmarks"].astype(np.float32)
    valid = data["valid"].astype(np.float32)
    valid_ratio = float(data["valid_ratio"])
    label = str(data["label"])
    video_name = str(data["video_name"])

    features, feature_names, interpolated = compute_features(raw, norm, valid, valid_ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        feature_names=np.array(feature_names, dtype=object),
        raw_landmarks=raw,
        norm_landmarks=norm,
        valid=valid,
        label=label,
        video_name=video_name,
        valid_ratio=valid_ratio,
        interpolated=interpolated,
    )

    return {
        "class": label,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "video_name": video_name,
        "valid_ratio": valid_ratio,
        "feature_shape": str(features.shape),
        "num_features": features.shape[1],
        "interpolated_frames": int(interpolated.sum()),
        "status": "ok",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    first_feature_names: List[str] | None = None

    for cls in CLASSES:
        in_dir = INPUT_DIR / cls
        out_dir = OUTPUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            rows.append({"class": cls, "status": "missing_input_folder", "input_file": str(in_dir)})
            print(f"[WARNING] Missing input folder: {in_dir}")
            continue

        files = sorted(in_dir.glob("*.npz"))
        print(f"\nProcessing class: {cls} | files: {len(files)}")

        for f in tqdm(files):
            out_path = out_dir / f.name
            try:
                row = process_file(f, out_path)
                rows.append(row)

                if first_feature_names is None:
                    with np.load(out_path, allow_pickle=True) as tmp:
                        first_feature_names = list(tmp["feature_names"])

            except Exception as e:
                rows.append({
                    "class": cls,
                    "input_file": str(f),
                    "output_file": str(out_path),
                    "status": f"error: {e}",
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    if first_feature_names is not None:
        with open(FEATURE_NAMES_JSON, "w", encoding="utf-8") as fp:
            json.dump(first_feature_names, fp, indent=2)

    ok = summary[summary["status"] == "ok"] if "status" in summary else pd.DataFrame()

    print("\nFeature extraction complete.")
    print("-" * 60)
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Summary CSV  : {SUMMARY_CSV}")
    print(f"Feature names: {FEATURE_NAMES_JSON}")
    print(f"Processed OK : {len(ok)}")

    if len(ok) > 0:
        print("\nCounts per class:")
        print(ok.groupby("class").size())
        print("\nFeature dimension:")
        print(ok["num_features"].value_counts().sort_index())

    errors = summary[summary["status"] != "ok"] if "status" in summary else pd.DataFrame()
    if len(errors) > 0:
        print("\nErrors / warnings:")
        print(errors[["class", "input_file", "status"]].head(30))
    print("-" * 60)


if __name__ == "__main__":
    main()
