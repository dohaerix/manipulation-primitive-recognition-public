# Step 5: semantic feature groups

This step sorts the feature columns into named groups, so related features can be handled together (for example, all fingertip-distance features at once).

| Path | What it is |
|---|---|
| `code/step1b_refine_feature_groups.py` | builds the groups used in the paper |
| `code/step1_build_feature_groups.py` | an earlier grouping by simple name matching, kept for reference. It over-grouped (for example it put `palm_vel_x` under palm orientation because the name contains "palm"). Its output is not used |
| `outputs/novelty_step1_feature_groups_v2/feature_group_mapping_action_only_v2.csv` / `.json` | each feature's index, name and group |
| `outputs/novelty_step1_feature_groups_v2/feature_group_summary_action_only_v2.csv` | feature count per group, with examples |
| `outputs/novelty_step1_feature_groups_v2/step1b_refined_group_report.txt` | readable report |

## The 16 groups (188 features)

| Group | Features | What it describes |
|---|---|---|
| trajectory_displacement | 27 | direction of travel, distance from the start, path length, straightness |
| finger_distance | 23 | fingertip-to-palm and fingertip-to-fingertip distances |
| bone_length | 22 | length of each finger bone after normalization |
| velocity_speed | 20 | speed of wrist, palm and hand centre |
| acceleration | 19 | change in speed |
| oscillation_reversal | 18 | back-and-forth motion, direction reversals |
| joint_angle | 16 | bend at each finger joint |
| jerk | 12 | change in acceleration |
| raw_global_shape | 12 | absolute positions and image-space bounding box |
| finger_curl | 6 | how curled each finger is |
| palm_orientation | 4 | palm direction and how much it turns |
| hand_shape_bbox | 3 | normalized bounding-box size |
| oscillation_curvature | 3 | path curvature |
| hand_openness, hand_shape_compactness, palm_shape_area | 1 each | overall hand shape |

The remaining 8 of the 196 features describe tracking quality or time position, and are not put in any group.

## How step1b decides the group

Each feature name is checked against rules in a fixed order, and the first rule that matches wins: support/metadata, then oscillation/reversal/curvature, jerk, acceleration, velocity/speed, trajectory/displacement, palm orientation, hand shape (curl, openness, distances, angles, bones), and finally raw/global shape. Because motion rules come first, `palm_vel_x` goes to velocity, not to palm orientation.

## What happens to the groups next (described in the main README)

1. **Masking during training.** During training, VAGM-TCN switches off whole groups at random.
2. **Choosing groups.** Each group was removed in turn to measure how much it matters (paper Fig. 2), and 7 groups (122 features) were kept for the final model: raw_global_shape, oscillation_reversal, trajectory_displacement, joint_angle, palm_orientation, finger_distance, bone_length.

## Run

From the repository root: `python 05_semantic_feature_groups/code/step1b_refine_feature_groups.py`
