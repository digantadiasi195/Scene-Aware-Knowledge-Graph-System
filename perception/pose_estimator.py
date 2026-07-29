import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_name="yolo11n-pose.pt"):
        print(" Loading YOLO-Pose...")
        self.model = YOLO(model_name)
        self.RIGHT_WRIST = 10
        print("✅ Pose estimator ready")

    def get_hand_xyz(self, frame_bgr, depth_frame=None, conf_threshold=0.5):
        results = self.model(frame_bgr, verbose=False)[0]
        h, w = frame_bgr.shape[:2]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return None

        keypoints_xy = results.keypoints.xy[0]
        keypoints_conf = results.keypoints.conf[0]

        if keypoints_xy.shape[0] <= self.RIGHT_WRIST:
            return None

        if keypoints_conf[self.RIGHT_WRIST] < conf_threshold:
            return None

        rw = keypoints_xy[self.RIGHT_WRIST]
        x2d, y2d = float(rw[0]), float(rw[1])

        if x2d <= 0 or y2d <= 0 or x2d >= w or y2d >= h:
            return None

        if depth_frame is not None:
            ix = max(0, min(int(x2d), w - 1))
            iy = max(0, min(int(y2d), h - 1))
            z = float(depth_frame[iy, ix])
            if z > 10.0:
                z = z / 1000.0
            if z <= 0 or np.isnan(z) or np.isinf(z):
                z = 1.5
        else:
            z = 1.5

        fx = fy = 500.0
        x_3d = (x2d - w / 2.0) * z / fx
        y_3d = (y2d - h / 2.0) * z / fy

        return (float(x_3d), float(y_3d), float(z))

    def get_full_skeleton(self, frame_bgr, conf_threshold=0.3):
        """Returns normalized 2D skeleton joints and connections for ALL detected humans"""
        results = self.model(frame_bgr, verbose=False)[0]
        h, w = frame_bgr.shape[:2]

        print(f"🔍 Pose results type: {type(results)}")
        print(f"🔍 Has keypoints: {results.keypoints is not None}")
        
        if results.keypoints is not None:
            print(f" Keypoints shape: {results.keypoints.xy.shape}")
            print(f"🔍 Number of humans detected: {len(results.keypoints.xy)}")
        else:
            print("❌ No keypoints detected!")
            return {"humans": []}

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return {"humans": []}

        humans_skeletons = []
        
        # Process ALL detected humans
        for person_idx in range(len(results.keypoints.xy)):
            keypoints_xy = results.keypoints.xy[person_idx]
            keypoints_conf = results.keypoints.conf[person_idx]

            print(f"👤 Person {person_idx}: {len(keypoints_xy)} keypoints, conf shape: {keypoints_conf.shape}")

            joints = []
            valid_joint_count = 0
            
            for i in range(len(keypoints_xy)):
                if i < len(keypoints_conf) and keypoints_conf[i] > conf_threshold:
                    x, y = float(keypoints_xy[i][0]) / w, float(keypoints_xy[i][1]) / h
                    joints.append({
                        "x": x, 
                        "y": y, 
                        "conf": float(keypoints_conf[i]),
                        "id": i
                    })
                    valid_joint_count += 1
                else:
                    joints.append(None)

            print(f"   Valid joints: {valid_joint_count}/17")

            # Only include humans with at least 3 valid joints
            if valid_joint_count < 3:
                print(f"   ️ Skipping person {person_idx} (too few valid joints)")
                continue

            # YOLO COCO pose connections (17 keypoints)
            connections = [
                (0, 1), (0, 2),      # Nose to eyes
                (1, 3), (2, 4),      # Eyes to ears
                (5, 6),              # Shoulders
                (5, 7), (7, 9),      # Right arm
                (6, 8), (8, 10),     # Left arm
                (5, 11), (6, 12),    # Shoulders to hips
                (11, 12),            # Hips
                (11, 13), (13, 15),  # Right leg
                (12, 14), (14, 16)   # Left leg
            ]

            valid_connections = []
            for start, end in connections:
                if joints[start] is not None and joints[end] is not None:
                    valid_connections.append((start, end))

            humans_skeletons.append({
                "joints": joints,
                "connections": valid_connections,
                "person_id": person_idx,
                "valid_joints": valid_joint_count
            })

        print(f"✅ Returning {len(humans_skeletons)} skeleton(s)")
        return {"humans": humans_skeletons}

    def close(self):
        pass