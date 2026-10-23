#!/usr/bin/env python3
"""
step1b_refine_feature_groups.py

Refines the Step-1 semantic feature grouping.

Why this exists:
    The first automatic grouping is useful, but it may over-group features.
    Example:
        palm_vel_x, palm_acc_x, palm_osc_zero_cross_x
    should not all become "palm_orientation" only because they contain "palm".

This refined version uses more specific priority rules:
    1. support/metadata
    2. oscillation/reversal/curvature
    3. jerk
    4. acceleration
    5. velocity/speed
    6. trajectory/displacement/path/straightness
    7. palm orientation/normal/angular
    8. hand shape/curl/openness/distances/angles/bones
    9. raw/global shape
    10. other

Inputs reused:
    feature_names_v1.json

Outputs:
    novelty_step1_feature_groups_v2/
        feature_group_mapping_action_only_v2.csv
        feature_group_summary_action_only_v2.csv
        feature_group_mapping_action_only_v2.json
        step1b_refined_group_report.txt

Run:
    python step1b_refine_feature_groups.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


FEATURE_NAMES_JSON = Path("03_feature_extraction/outputs/feature_names_v1.json")
FEATURE_ROOT = Path("03_feature_extraction/outputs/processed_features_v1")
OUTPUT_DIR = Path("05_semantic_feature_groups/outputs/novelty_step1_feature_groups_v2")

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
        return [str(x) for x in json.loads(FEATURE_NAMES_JSON.read_text())]

    first = next(FEATURE_ROOT.glob("*/*.npz"), None)
    if first is None:
        raise FileNotFoundError(
            "Could not find feature_names_v1.json or any .npz files under processed_features_v1/"
        )

    data = np.load(first, allow_pickle=True)
    if "feature_names" not in data.files:
        raise KeyError(f"{first} does not contain feature_names.")

    return [str(x) for x in data["feature_names"]]


def refined_group(name: str) -> str:
    n = name.lower()

    # 0. Support/meta features
    if n in {"valid", "interpolated", "valid_ratio", "missing_run_length"}:
        return "support_validity"
    if n == "time01" or n.startswith("phase_"):
        return "support_temporal_metadata"

    # 1. Oscillation/reversal/curvature first because names may also contain palm/wrist/centroid.
    if "osc" in n or "zero_cross" in n or "zerocross" in n or "reversal" in n:
        return "oscillation_reversal"
    if "tortuosity" in n or "curvature" in n:
        return "oscillation_curvature"

    # 2. Higher-order temporal derivatives.
    if "jerk" in n:
        return "jerk"
    if "acceleration" in n or "accel" in n or "acc_" in n or n.startswith("acc") or "_acc_" in n or n.endswith("_acc") or "angular_acc" in n:
        return "acceleration"
    if "velocity" in n or "speed" in n or "vel" in n or "angular_speed" in n:
        return "velocity_speed"

    # 3. Trajectory / displacement / path geometry.
    if (
        "trajectory" in n or "displacement" in n or "disp" in n or
        "path" in n or "travel" in n or "straightness" in n or
        "direction" in n
    ):
        return "trajectory_displacement"

    # 4. Palm orientation / palm geometry.
    if (
        "palm_normal" in n or "palm_orientation" in n or
        "orientation_change" in n or "normal_" in n or
        n in {"palm_normal_x", "palm_normal_y", "palm_normal_z"}
    ):
        return "palm_orientation"
    if n == "norm_palm_area":
        return "palm_shape_area"

    # 5. Finger/hand configuration.
    if "curl" in n:
        return "finger_curl"
    if "openness" in n or "open" in n or "aperture" in n:
        return "hand_openness"
    if "thumb" in n or "tip" in n or "finger" in n or "fingertip" in n:
        return "finger_distance"
    if "distance" in n or "dist" in n or "pinch" in n:
        return "finger_distance"
    if "angle" in n:
        return "joint_angle"
    if "bone" in n or "length" in n:
        return "bone_length"
    if "compactness" in n:
        return "hand_shape_compactness"

    # 6. Raw/global measurements.
    if n.startswith("raw_"):
        return "raw_global_shape"

    # 7. Normalized bounding box / hand shape measurements.
    if "bbox" in n:
        return "hand_shape_bbox"
    if "norm" in n or "landmark" in n or "coord" in n:
        return "normalized_landmark_shape"
    if n.endswith("_x") or n.endswith("_y") or n.endswith("_z"):
        return "coordinate_shape"

    return "other"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names()

    rows = []
    model_idx = 0
    for original_idx, name in enumerate(feature_names):
        if name in NON_ACTION_FEATURES:
            continue

        rows.append({
            "model_input_index_0based": model_idx,
            "original_feature_index_0based": original_idx,
            "feature_name": name,
            "feature_group": refined_group(name),
        })
        model_idx += 1

    mapping = pd.DataFrame(rows)

    summary = (
        mapping.groupby("feature_group")
        .agg(
            n_features=("feature_name", "count"),
            example_features=("feature_name", lambda x: ", ".join(list(x)[:6])),
        )
        .reset_index()
        .sort_values(["n_features", "feature_group"], ascending=[False, True])
    )

    mapping_csv = OUTPUT_DIR / "feature_group_mapping_action_only_v2.csv"
    mapping_json = OUTPUT_DIR / "feature_group_mapping_action_only_v2.json"
    summary_csv = OUTPUT_DIR / "feature_group_summary_action_only_v2.csv"
    report_txt = OUTPUT_DIR / "step1b_refined_group_report.txt"

    mapping.to_csv(mapping_csv, index=False)
    mapping_json.write_text(json.dumps(mapping.to_dict(orient="records"), indent=2))
    summary.to_csv(summary_csv, index=False)

    lines = []
    lines.append("Step 1B Refined Feature Grouping Report")
    lines.append("=" * 80)
    lines.append(f"Total original features: {len(feature_names)}")
    lines.append(f"Action features mapped: {len(mapping)}")
    lines.append(f"Support features removed: {sorted(NON_ACTION_FEATURES)}")
    lines.append("")
    lines.append("Refined group summary:")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("Output files:")
    lines.append(f"- {mapping_csv}")
    lines.append(f"- {mapping_json}")
    lines.append(f"- {summary_csv}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "This refined grouping avoids putting all palm-related motion features into palm_orientation. "
        "It separates palm velocity/acceleration/jerk/oscillation into temporal-motion groups, which is "
        "more appropriate for semantic feature-group masking."
    )

    report_txt.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
