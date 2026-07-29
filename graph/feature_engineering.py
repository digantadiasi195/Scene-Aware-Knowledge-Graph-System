# feature_engineering.py
"""
Turns raw per-frame observations (video_id, frame, hand_xyz, object_xyz, ...)
into:
  1. per-frame geometry (d_Ho, theta, e) -- via geometry.geometry_from_coords
  2. per-action-instance summaries (one row per action_instance_id) with the
     mean d_Ho/theta/e across its frames -- this is what gets written onto
     each ActionInstance node in Neo4j.

Nothing here is hardcoded: every number is computed from the input CSV.
Swap `sample_dataset_generator.py`'s output for a CSV produced by your real
video pipeline and this file (and everything downstream) works unchanged,
as long as the column names match.
"""

import pandas as pd
from geometry import geometry_from_coords


REQUIRED_COLUMNS = [
    "video_id", "human_id", "scene_id", "activity_id", "activity_instance_id",
    "action_instance_id", "step_order", "action_id", "object_id", "frame",
    "timestamp", "hand_x", "hand_y", "hand_z", "object_x", "object_y", "object_z",
]


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df


def compute_frame_geometry(df, camera_xyz=(0.0, 0.0, 0.0)):
    """Adds d_Ho, d_H, theta, e columns computed per frame."""
    geoms = df.apply(
        lambda row: geometry_from_coords(
            (row.hand_x, row.hand_y, row.hand_z),
            (row.object_x, row.object_y, row.object_z),
            camera_xyz,
        ),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df, geoms], axis=1)


def summarize_action_instances(df_with_geometry):
    """
    One row per action_instance_id, with the mean geometry across its
    frames plus the metadata needed to create graph relationships
    (human, scene, object, action type, activity instance, step order,
    and the video/frame span for provenance).
    """
    agg = df_with_geometry.groupby("action_instance_id").agg(
        video_id=("video_id", "first"),
        human_id=("human_id", "first"),
        scene_id=("scene_id", "first"),
        activity_id=("activity_id", "first"),
        activity_instance_id=("activity_instance_id", "first"),
        step_order=("step_order", "first"),
        action_id=("action_id", "first"),
        object_id=("object_id", "first"),
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        d_Ho=("d_Ho", "mean"),
        theta=("theta", "mean"),
        e=("e", "mean"),
        n_frames=("frame", "count"),
    ).reset_index().rename(columns={"action_instance_id": "instance_id"})
    return agg


def summarize_activity_instances(action_instance_summary):
    """One row per activity_instance_id, spanning all its action instances."""
    agg = action_instance_summary.groupby("activity_instance_id").agg(
        video_id=("video_id", "first"),
        human_id=("human_id", "first"),
        scene_id=("scene_id", "first"),
        activity_id=("activity_id", "first"),
        start_frame=("start_frame", "min"),
        end_frame=("end_frame", "max"),
        n_steps=("instance_id", "count"),
    ).reset_index().rename(columns={"activity_instance_id": "instance_id"})
    return agg


def build_observation_rows(df_with_geometry):
    """
    One row per raw frame -- becomes an Observation node (frame-level
    sensor reading: hand position + nearest object + its distance).
    """
    obs = df_with_geometry[[
        "video_id", "human_id", "scene_id", "frame", "timestamp",
        "hand_x", "hand_y", "hand_z", "object_id", "d_Ho",
    ]].copy()
    obs["observation_id"] = (
        obs["video_id"] + "_f" + obs["frame"].astype(str) + "_" + obs["human_id"]
    )
    return obs


def run_feature_engineering(csv_path, camera_xyz=(0.0, 0.0, 0.0)):
    """Convenience entry point used by etl_pipeline.py."""
    raw = load_raw(csv_path)
    with_geom = compute_frame_geometry(raw, camera_xyz)
    action_instances = summarize_action_instances(with_geom)
    activity_instances = summarize_activity_instances(action_instances)
    observations = build_observation_rows(with_geom)
    return {
        "raw_with_geometry": with_geom,
        "action_instances": action_instances,
        "activity_instances": activity_instances,
        "observations": observations,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "raw_observations.csv"
    result = run_feature_engineering(csv_path)
    print("Action instances:", len(result["action_instances"]))
    print("Activity instances:", len(result["activity_instances"]))
    print("Observations:", len(result["observations"]))
    print(result["action_instances"].head())