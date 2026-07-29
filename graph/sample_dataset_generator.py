# graph/sample_dataset_generator.py
"""
Generates a synthetic raw per-frame observation dataset that stands in for
what your future vision pipeline (Mask2Former/YOLO for objects, a pose
model for the hand/skeleton, an action recognizer for action labels) would
actually produce. Use this to test/demo the ETL pipeline end-to-end today;
replace it with real extraction code once your CV pipeline is wired up --
everything downstream (feature_engineering.py, etl_pipeline.py,
stats_learning.py) consumes the SAME csv schema either way.

Output CSV columns (one row = one detected frame of one action instance):
    video_id, human_id, scene_id, activity_id, activity_instance_id,
    action_instance_id, step_order, action_id, object_id, frame, timestamp,
    hand_x, hand_y, hand_z, object_x, object_y, object_z
"""

# graph/sample_dataset_generator.py
"""
Generates a synthetic raw per-frame observation dataset...
[Docstring remains the same]
"""

import csv
import random
from knowledge_seed import ACTIVITIES

random.seed(42)

HUMANS = ["H1", "H2", "H3", "H4"]
FRAMES_PER_ACTION = 12          
FRAME_RATE_HZ = 30
N_REPEATS_PER_ACTIVITY_SCENE = 6  

def _object_baseline_position(object_id):
    """A fixed nominal 3D position per object type (arbitrary demo units, cm)."""
    rnd = random.Random(hash(object_id) % (2 ** 32))
    return (rnd.uniform(40, 90), rnd.uniform(-20, 20), rnd.uniform(60, 110))

def _simulate_action_frames(object_xyz, n_frames=FRAMES_PER_ACTION):
    """Simulates a reach-and-use trajectory with noise."""
    start = (
        object_xyz[0] + random.uniform(60, 100),
        object_xyz[1] + random.uniform(-40, 40),
        object_xyz[2] + random.uniform(-20, 20),
    )
    rows = []
    for f in range(n_frames):
        t = f / (n_frames - 1)
        hand = tuple(
            start[i] + (object_xyz[i] - start[i]) * t + random.gauss(0, 2.0)
            for i in range(3)
        )
        obj_noisy = tuple(object_xyz[i] + random.gauss(0, 1.0) for i in range(3))
        rows.append((hand, obj_noisy))
    return rows

def generate(output_csv="raw_observations.csv"):
    rows = []
    video_counter = 0
    action_inst_counter = 0
    activity_inst_counter = 0
    
    # FIX: We will use a global frame counter to guarantee unique observation IDs
    frame_global = 0 

    for act in ACTIVITIES:
        for scene_id in act["possible_scenes"]:
            for _rep in range(N_REPEATS_PER_ACTIVITY_SCENE):
                video_counter += 1
                video_id = f"V{video_counter:04d}"
                human_id = random.choice(HUMANS)
                activity_inst_counter += 1
                activity_instance_id = f"ACI{activity_inst_counter:05d}"

                sequence = random.choice(act["sequences"])
                for step_order, (action_id, object_id) in enumerate(sequence):
                    action_inst_counter += 1
                    action_instance_id = f"AII{action_inst_counter:06d}"
                    object_xyz = _object_baseline_position(object_id)
                    frames = _simulate_action_frames(object_xyz)

                    for f_idx, (hand_xyz, obj_xyz) in enumerate(frames):
                        # FIX: Use frame_global instead of f_idx to prevent ID collisions
                        # in feature_engineering.py's observation_id generation.
                        
                        rows.append({
                            "video_id": video_id,
                            "human_id": human_id,
                            "scene_id": scene_id,
                            "activity_id": act["activity_id"],
                            "activity_instance_id": activity_instance_id,
                            "action_instance_id": action_instance_id,
                            "step_order": step_order,
                            "action_id": action_id,
                            "object_id": object_id,
                            "frame": frame_global,  # <--- CRITICAL FIX
                            "timestamp": round(frame_global / FRAME_RATE_HZ, 4),
                            "hand_x": round(hand_xyz[0], 3),
                            "hand_y": round(hand_xyz[1], 3),
                            "hand_z": round(hand_xyz[2], 3),
                            "object_x": round(obj_xyz[0], 3),
                            "object_y": round(obj_xyz[1], 3),
                            "object_z": round(obj_xyz[2], 3),
                        })
                        frame_global += 1 # Increment global frame

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} raw frame rows across {video_counter} videos -> {output_csv}")
    return output_csv

if __name__ == "__main__":
    generate()