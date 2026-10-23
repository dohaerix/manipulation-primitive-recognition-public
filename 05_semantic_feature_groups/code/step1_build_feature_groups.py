#!/usr/bin/env python3
"""
step1_build_feature_groups.py

Step 1 for the stronger novelty direction:
    Validity-Aware Semantic Feature-Group Masking.

This script DOES NOT rerun:
    - video processing
    - MediaPipe landmark extraction
    - normalization
    - feature extraction
    - grouped train/val/test splitting

It only reuses existing files:
    - feature_names_v1.json
    - processed_features_v1/
    - baseline_grouped_outputs_v1/grouped_splits.csv, if available

Purpose:
    1. Load the existing 196 feature names.
    2. Remove the same 8 non-action/support features to obtain 188 model input features.
    3. Assign each of the 188 action features to a semantic group.
    4. Save a CSV mapping every feature to a group.
    5. Save a group summary.
    6. Audit a few .npz feature files to verify shapes.

Outputs:
    novelty_step1_feature_groups/
        feature_group_mapping_action_only.csv
        feature_group_summary_action_only.csv
        feature_group_mapping_action_only.json
        step1_feature_group_report.txt

Run:
    python step1_build_feature_groups.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


FEATURE_ROOT = Path("03_feature_extraction/outputs/processed_features_v1")
FEATURE_NAMES_JSON = Path("03_feature_extraction/outputs/feature_names_v1.json")
SPLITS_CSV = Path("04_train_val_test_split/outputs/grouped_splits.csv")
OUTPUT_DIR = Path("05_semantic_feature_groups/outputs/novelty_step1_feature_groups")

NON_ACTION_FEATURES = {
    "valid",
    "interpolated",
    "valid_ratio",
    "missing_run_length",
    "time01",
    "phase_early",
    "phase_middle",
    "phase_late",
}


def load_feature_names() -> List[str]:
    if FEATURE_NAMES_JSON.exists():
        names = json.loads(FEATURE_NAMES_JSON.read_text())
        return [str(x) for x in names]

    first = next(FEATURE_ROOT.glob("*/*.npz"), None)
    if first is None:
        raise FileNotFoundError(
            "Could not find feature_names_v1.json or any .npz files under processed_features_v1/"
        )

    data = np.load(first, allow_pickle=True)
    if "feature_names" not in data.files:
        raise KeyError(
            f"{first} does not contain feature_names. Please provide feature_names_v1.json."
        )

    return [str(x) for x in data["feature_names"]]


def infer_feature_group(name: str) -> str:
    n = name.lower()

    if n in {"valid", "interpolated", "valid_ratio", "missing_run_length"}:
        return "support_validity"
    if n == "time01" or n.startswith("phase_"):
        return "support_temporal_metadata"

    if "curl" in n:
        return "finger_curl"
    if "openness" in n or "open" in n or "aperture" in n:
        return "hand_openness"
    if "thumb" in n or "tip" in n or "finger" in n or "fingertip" in n:
        return "finger_distance"
    if "distance" in n or "dist" in n or "pinch" in n:
        return "finger_distance"
    if "palm" in n or "normal" in n or "orientation" in n:
        return "palm_orientation"
    if "velocity" in n or "speed" in n or "vel" in n:
        return "velocity_speed"
    if "acceleration" in n or "accel" in n or "acc_" in n or n.startswith("acc"):
        return "acceleration"
    if "jerk" in n:
        return "jerk"
    if "osc" in n or "zero_cross" in n or "zerocross" in n or "reversal" in n:
        return "oscillation_reversal"
    if "tortuosity" in n or "curvature" in n:
        return "oscillation_curvature"
    if "trajectory" in n or "displacement" in n or "path" in n or "travel" in n:
        return "trajectory_displacement"
    if "angle" in n:
        return "joint_angle"
    if "bone" in n or "length" in n:
        return "bone_length"
    if "raw" in n:
        return "raw_landmark_shape"
    if "norm" in n or "landmark" in n or "coord" in n:
        return "normalized_landmark_shape"
    if n.endswith("_x") or n.endswith("_y") or n.endswith("_z"):
        return "normalized_landmark_shape"

    return "other"


def build_mapping(feature_names: List[str], include_all_features: bool = False) -> pd.DataFrame:
    rows = []
    model_input_idx = 0

    for original_idx, name in enumerate(feature_names):
        is_support = name in NON_ACTION_FEATURES
        used_in_mapping = include_all_features or not is_support

        if used_in_mapping:
            current_model_idx = model_input_idx
            model_input_idx += 1
        else:
            current_model_idx = None

        rows.append({
            "model_input_index_0based": current_model_idx,
            "original_feature_index_0based": original_idx,
            "feature_name": name,
            "feature_group": infer_feature_group(name),
            "is_support_feature_removed_from_model_input": bool(is_support),
            "used_in_this_mapping": bool(used_in_mapping),
        })

    df = pd.DataFrame(rows)
    if not include_all_features:
        df = df[df["used_in_this_mapping"]].copy().reset_index(drop=True)
        df["model_input_index_0based"] = df["model_input_index_0based"].astype(int)

    return df


def audit_feature_files(max_files: int = 5) -> List[str]:
    lines = []
    files = sorted(FEATURE_ROOT.glob("*/*.npz"))
    lines.append(f"Feature root exists: {FEATURE_ROOT.exists()}")
    lines.append(f"Number of .npz feature files found: {len(files)}")

    if not files:
        return lines

    for path in files[:max_files]:
        try:
            data = np.load(path, allow_pickle=True)
            keys = list(data.files)
            shape = tuple(data["features"].shape) if "features" in data.files else None
            label = str(data["label"]) if "label" in data.files else "NA"
            valid_ratio = float(data["valid_ratio"]) if "valid_ratio" in data.files else float("nan")
            lines.append(
                f"Sample file: {path} | keys={keys} | features_shape={shape} | "
                f"label={label} | valid_ratio={valid_ratio:.4f}"
            )
        except Exception as exc:
            lines.append(f"Could not read {path}: {repr(exc)}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-all-features", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names()
    mode = "all_features" if args.include_all_features else "action_only"
    mapping = build_mapping(feature_names, include_all_features=args.include_all_features)

    mapping_csv = OUTPUT_DIR / f"feature_group_mapping_{mode}.csv"
    mapping_json = OUTPUT_DIR / f"feature_group_mapping_{mode}.json"
    summary_csv = OUTPUT_DIR / f"feature_group_summary_{mode}.csv"
    report_txt = OUTPUT_DIR / "step1_feature_group_report.txt"

    mapping.to_csv(mapping_csv, index=False)
    mapping_json.write_text(json.dumps(mapping.to_dict(orient="records"), indent=2))

    summary = (
        mapping.groupby("feature_group")
        .agg(
            n_features=("feature_name", "count"),
            example_features=("feature_name", lambda x: ", ".join(list(x)[:5])),
        )
        .reset_index()
        .sort_values(["n_features", "feature_group"], ascending=[False, True])
    )
    summary.to_csv(summary_csv, index=False)

    lines = []
    lines.append("Step 1 Feature Grouping Report")
    lines.append("=" * 80)
    lines.append(f"Mode: {mode}")
    lines.append(f"Total features in feature_names_v1.json: {len(feature_names)}")
    lines.append(f"Support features expected to remove: {sorted(NON_ACTION_FEATURES)}")
    lines.append(f"Features in mapping: {len(mapping)}")
    lines.append("")
    lines.append("Group summary:")
    lines.append(summary.to_string(index=False))
    lines.append("")

    if SPLITS_CSV.exists():
        splits = pd.read_csv(SPLITS_CSV)
        lines.append(f"Grouped split file found: {SPLITS_CSV}")
        lines.append(f"Split rows: {len(splits)}")
        if "split" in splits.columns:
            lines.append("Split counts:")
            lines.append(str(splits["split"].value_counts()))
        if "class" in splits.columns:
            lines.append("Class counts:")
            lines.append(str(splits["class"].value_counts().sort_index()))
    else:
        lines.append(f"Grouped split file not found: {SPLITS_CSV}")
    lines.append("")

    lines.extend(audit_feature_files(max_files=5))
    lines.append("")
    lines.append("Output files:")
    lines.append(f"- {mapping_csv}")
    lines.append(f"- {mapping_json}")
    lines.append(f"- {summary_csv}")
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "Open feature_group_mapping_action_only.csv and check whether many features are mapped to 'other'. "
        "If yes, share the CSV or group summary so the grouping can be refined before training VAGM-TCN."
    )

    report_txt.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
