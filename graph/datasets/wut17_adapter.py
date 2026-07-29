# # # graph/datasets/wut17_adapter.py
# # """
# # WUT-17 Dataset Adapter V13 (Final Safe Version)
# # - Processes ONLY Activity 1 (for sanity testing).
# # - Explicitly loops through all 12 Persons and 2 Cameras.
# # - Correctly reads the header row in 3D Skeleton ('W') files.
# # - Fixes the PersonCount filter (filters for 0).
# # - Fixes the Object Frame offset (adds +1 to index).
# # - Uses Real 3D Hand Coordinates from the 'W' file.
# # - Uses Depth Images for Object 3D coordinates.
# # - FIX 1: Corrects negative Z-axis in skeleton data.
# # - FIX 2: Checks all 3 object labels safely (handles missing columns).
# # """

# # import os
# # import cv2
# # import csv
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # from difflib import get_close_matches

# # # ==================================================================== #
# # # 1. CONFIGURATION (Activity 1 Only)
# # # ==================================================================== #
# # # TEMPORARY: Only process Activity 1 for testing
# # ACTIVITY_FOLDERS = ["Activity1"] 

# # ACTIVITY_TO_SCENE = {"Activity1": "SC1"}
# # ACTIVITY_MAP = {"Activity1": "AC2"} 

# # # ==================================================================== #
# # # 2. MAPPINGS (Actions & Objects)
# # # ==================================================================== #
# # ACTION_GROUND_TRUTH = {
# #     # IGNORED (Background)
# #     "standing still": None, "standing  still": None, "sitting": None, 
# #     "sitting, reading": None, "sitting, eating": None, "stitting, eating": None,
# #     "standing, hold cart": None, "not specified": None,
    
# #     # a1: Approaching / Moving
# #     "standing up": "a1", "standing, up": "a1",
# #     "rotation": "a1", "rotaion": "a1", "rotaiton": "a1",
# #     "walking forward": "a1", "walking, forward": "a1", "walking, backwards": "a1", "walking, towards observer": "a1",
    
# #     # a2: Reaching / Collecting
# #     "standing, collect object": "a2", "extend arms, collect object": "a2", "extending arms, collect object": "a2",
# #     "crouch, collect object": "a2", "crouching, collect object": "a2", "bend, retrieve object": "a2",
# #     "standing, pick up phone": "a2",
    
# #     # a4: Pouring
# #     "standing, move hand vertically": "a4",
    
# #     # a5: Placing
# #     "standing, place object": "a5", "stading, place object": "a5", "standind, place object": "a5",
# #     "standing , place object": "a5", "standing, place pbject": "a5", "extend arms, place object": "a5",
# #     "bend, place object": "a5", "bend,place object": "a5",
    
# #     # a6: Drinking
# #     "sitting, drinking": "a6",
    
# #     # a9: Passing
# #     "standing, move object to other hand": "a9"
# # }

# # def get_action_id(action_str):
# #     action_str = str(action_str).strip().lower()
# #     if action_str in ACTION_GROUND_TRUTH:
# #         return ACTION_GROUND_TRUTH[action_str]
# #     valid_keys = [k for k in ACTION_GROUND_TRUTH.keys() if ACTION_GROUND_TRUTH[k] is not None]
# #     matches = get_close_matches(action_str, valid_keys, n=1, cutoff=0.7)
# #     if matches:
# #         return ACTION_GROUND_TRUTH[matches[0]]
# #     return None

# # OBJECT_MAP = {
# #     "table": "o_table", "high_table": "o_table", "chair": "o_object", "trolley": "o_object",
# #     "door": "o_door", "book": "o_book", 
# #     "bottle": "o_bottle", "kettle": "o_bottle", "spray": "o_bottle",
# #     "glass": "o_glass", "cup": "o_glass", 
# #     "plate": "o_bowl", "phone": "o_phone", "etui": "o_phone",
# #     "cloth": "o_object", "box": "o_object", "package": "o_object",
# #     "suitcase": "o_object", "broom": "o_object", "comb": "o_object"
# # }

# # # ==================================================================== #
# # # 3. 3D GEOMETRY HELPERS
# # # ==================================================================== #
# # FX, FY, CX, CY = 525.0, 525.0, 320.0, 240.0
# # DEPTH_SCALE = 1000.0 

# # def get_depth_z(depth_img, u, v):
# #     u, v = int(round(u)), int(round(v))
# #     h, w = depth_img.shape
# #     if 0 <= u < w and 0 <= v < h:
# #         z_mm = depth_img[v, u]
# #         z_m = float(z_mm) / DEPTH_SCALE
# #         if 0.1 < z_m < 4.0: 
# #             return z_m
# #     return None

# # def pixel_to_real_3d(u, v, z):
# #     x = (u - CX) * z / FX
# #     y = (v - CY) * z / FY
# #     return round(x, 4), round(y, 4), round(z, 4)

# # def find_depth_path(root_dir, activity_name, rec_name, frame_num):
# #     depth_dir = Path(root_dir) / "Depth_Images" / activity_name / rec_name
# #     for fmt in [f"frame_{frame_num:04d}.png", f"depth_{frame_num:04d}.png", f"{frame_num:04d}.png"]:
# #         p = depth_dir / fmt
# #         if p.exists():
# #             return p
# #     return None

# # # ==================================================================== #
# # # 4. EXTRACTION LOGIC
# # # ==================================================================== #
# # def process_recording(root_dir, activity_name, rec_name, out_rows):
# #     parts = rec_name.replace("R", "").split("C")
# #     person_num = int(parts[0])
# #     person_idx = person_num - 1 
# #     human_id = f"H{person_num}"
    
# #     video_id = f"WUT_{activity_name}_{rec_name}"
# #     activity_id = ACTIVITY_MAP.get(activity_name, "AC2")
# #     scene_id = ACTIVITY_TO_SCENE.get(activity_name, "SC1")
# #     activity_instance_id = f"WUT_ACI_{activity_name}_{rec_name}"

# #     # Paths
# #     annot_path = Path(root_dir) / "Annotation" / activity_name / f"P{person_num}C{parts[1]}.xlsx"
# #     obj_path = Path(root_dir) / "Object_Information" / activity_name / f"{rec_name}.csv"
# #     skel_dir = Path(root_dir) / "Skeleton_Data" / activity_name / rec_name

# #     if not annot_path.exists():
# #         return
# #     if not obj_path.exists():
# #         return
# #     if not skel_dir.exists():
# #         return

# #     # DYNAMIC FILE FINDING: Find 3D ('W') file
# #     skel_3d_files = list(skel_dir.glob("*W*.csv"))
# #     if not skel_3d_files:
# #         return

# #     skel_3d_path = skel_3d_files[0] 

# #     try:
# #         df_annot = pd.read_excel(annot_path)
# #         action_col = [c for c in df_annot.columns if c.lower() != 'frame'][0]
# #         df_annot.rename(columns={action_col: 'Action'}, inplace=True)
# #         df_annot['Frame'] = df_annot['Frame'].astype(int)

# #         # FIX: Object CSV uses 0-based index, Annotation uses 1-based frames
# #         df_obj = pd.read_csv(obj_path)
# #         df_obj['Frame'] = df_obj.index + 1  

# #         # Read 3D Skeleton WITH HEADER
# #         df_skel = pd.read_csv(skel_3d_path, header=0)
# #         df_skel.columns = [c.strip() for c in df_skel.columns]
        
# #         if 'Frame number' in df_skel.columns:
# #             df_skel.rename(columns={'Frame number': 'Frame', 'person count': 'PersonCount'}, inplace=True)
            
# #         df_skel['Frame'] = df_skel['Frame'].astype(int)
        
# #         # FIX: WUT-17 skeleton files only contain data for ONE person, labeled as '0'
# #         df_skel = df_skel[df_skel['PersonCount'] == 0]

# #     except Exception as e:
# #         print(f"  [ERROR] Failed to load data for {video_id}: {e}")
# #         return

# #     # Frame Alignment (Just in case)
# #     min_ann_frame = df_annot['Frame'].min()
# #     min_skel_frame = df_skel['Frame'].min()
# #     if min_ann_frame != min_skel_frame:
# #         offset = min_ann_frame - min_skel_frame
# #         df_skel['Frame'] = df_skel['Frame'] + offset
        
# #     df_merged = pd.merge(df_annot, df_obj, on='Frame', how='inner')
# #     df_merged = pd.merge(df_merged, df_skel, on='Frame', how='inner')
    
# #     if df_merged.empty:
# #         return

# #     # Action Segmentation
# #     segments = []
# #     current_action = None
# #     start_frame = 0
# #     for idx, row in df_merged.iterrows():
# #         frame = int(row['Frame'])
# #         action_str = str(row['Action']).strip().lower()
# #         action_id = get_action_id(action_str)
        
# #         if action_id != current_action:
# #             if current_action is not None:
# #                 segments.append({"start_frame": start_frame, "end_frame": frame - 1, "action_id": current_action})
# #             current_action = action_id
# #             start_frame = frame
# #     if current_action is not None:
# #         segments.append({"start_frame": start_frame, "end_frame": int(df_merged['Frame'].max()), "action_id": current_action})

# #     rows_extracted = 0
# #     for step_order, seg in enumerate(segments):
# #         if seg['action_id'] is None: continue 
        
# #         action_instance_id = f"WUT_AI_{activity_name}_{rec_name}_{step_order}"
# #         df_segment = df_merged[(df_merged['Frame'] >= seg['start_frame']) & (df_merged['Frame'] <= seg['end_frame'])]

# #         for _, row in df_segment.iterrows():
# #             frame_num = int(row['Frame'])
            
# #             # 1. Get Hand 3D directly from the 'W' (3D) file! (Joint 10 = Right Hand)
# #             hand_x, hand_y, hand_z = row['x10'], row['y10'], row['z10']
            
# #             # CRITICAL FIX 1: Invert the Z-axis if it's negative
# #             hand_z = abs(hand_z)
            
# #             # Auto-convert mm to meters if Z is > 10
# #             if abs(hand_z) > 10.0: 
# #                 hand_x, hand_y, hand_z = hand_x/1000.0, hand_y/1000.0, hand_z/1000.0

# #             # 2. Get Object 3D (Needs depth image for Z)
# #             obj_id = "o_object"
# #             obj_xyz = (0.0, 0.0, 1.5) # Fallback
# #             obj_u, obj_v = 0.0, 0.0

# #             # CRITICAL FIX 2: Check label1, label2, and label3 safely
# #             for i in range(1, 4):
# #                 label_col = f'label{i}'
# #                 # Check if label column exists and is not empty
# #                 if label_col in row.index and pd.notna(row[label_col]):
# #                     raw_label = str(row[label_col]).strip().lower()
# #                     mapped_obj = OBJECT_MAP.get(raw_label, "o_object")
                    
# #                     # If it maps to a specific object (not the generic o_object), use it!
# #                     if mapped_obj != "o_object":
# #                         obj_id = mapped_obj
                        
# #                         # Safely check if coordinate columns exist before reading them
# #                         col_x1, col_y1 = f'x{i*2-1}', f'y{i*2-1}'
# #                         col_x2, col_y2 = f'x{i*2}', f'y{i*2}'
                        
# #                         if col_x1 in row.index and col_y1 in row.index and col_x2 in row.index and col_y2 in row.index:
# #                             x1, y1 = row[col_x1], row[col_y1]
# #                             x2, y2 = row[col_x2], row[col_y2]
# #                             obj_u = (x1 + x2) / 2 * 640 
# #                             obj_v = (y1 + y2) / 2 * 480
# #                             break

# #             # Get Depth for Object
# #             depth_path = find_depth_path(root_dir, activity_name, rec_name, frame_num)
# #             if depth_path:
# #                 depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
# #                 if depth_img is not None:
# #                     obj_z = get_depth_z(depth_img, obj_u, obj_v)
# #                     if obj_z: 
# #                         obj_xyz = pixel_to_real_3d(obj_u, obj_v, obj_z)
# #                     else:
# #                         # Fallback: If depth fails, assume object is at the same depth as the hand
# #                         obj_xyz = (obj_u - 320.0) * hand_z / 525.0, (obj_v - 240.0) * hand_z / 525.0, hand_z

# #             out_rows.append({
# #                 "video_id": video_id, "human_id": human_id, "scene_id": scene_id,
# #                 "activity_id": activity_id, "activity_instance_id": activity_instance_id,
# #                 "action_instance_id": action_instance_id, "step_order": step_order,
# #                 "action_id": seg['action_id'], "object_id": obj_id, "frame": frame_num,
# #                 "timestamp": round(frame_num / 60.0, 4),
# #                 "hand_x": round(float(hand_x), 4), "hand_y": round(float(hand_y), 4), "hand_z": round(float(hand_z), 4),
# #                 "object_x": round(float(obj_xyz[0]), 4), "object_y": round(float(obj_xyz[1]), 4), "object_z": round(float(obj_xyz[2]), 4),
# #             })
# #             rows_extracted += 1
            
# #     print(f"  [SUCCESS] Extracted {rows_extracted} rows for {video_id}")

# # def run_wut17_extraction(root_dir, output_csv="wut17_test_activity1.csv"):
# #     print(f"Starting WUT-17 V13 Extraction (Activity 1 - Explicit Loop)...")
# #     out_rows = []
# #     base_path = Path(root_dir)
# #     annot_dir = base_path / "Annotation"
    
# #     if not annot_dir.exists():
# #         print(f"ERROR: Annotation directory not found at {annot_dir}")
# #         return

# #     for activity in ACTIVITY_FOLDERS:
# #         if not (annot_dir / activity).exists():
# #             print(f"Warning: {activity} not found.")
# #             continue
            
# #         # EXPLICIT LOOP: Force processing of all 12 Persons and 2 Cameras
# #         for person_num in range(1, 13):
# #             for cam_num in range(1, 3):
# #                 rec_name = f"R{person_num}C{cam_num}"
# #                 process_recording(base_path, activity, rec_name, out_rows)
            
# #     if out_rows:
# #         fieldnames = [
# #             "video_id", "human_id", "scene_id", "activity_id", "activity_instance_id",
# #             "action_instance_id", "step_order", "action_id", "object_id", "frame",
# #             "timestamp", "hand_x", "hand_y", "hand_z", "object_x", "object_y", "object_z"
# #         ]
# #         with open(output_csv, 'w', newline='') as f:
# #             writer = csv.DictWriter(f, fieldnames=fieldnames)
# #             writer.writeheader()
# #             writer.writerows(out_rows)
# #         print(f"\nSuccess! Extracted {len(out_rows)} frames -> {output_csv}")
# #     else:
# #         print("\nNo data extracted.")

# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser(description="Extract WUT-17 Activity 1 to CSV")
# #     parser.add_argument("--root", type=str, required=True, help="Path to WUT-17 root folder")
# #     parser.add_argument("--output", type=str, default="wut17_test_activity1.csv", help="Output CSV filename")
# #     args = parser.parse_args()
# #     run_wut17_extraction(args.root, args.output)


# ######################################################################
# # graph/datasets/wut17_adapter.py
# """
# WUT-17 Dataset Adapter V21 (Final Mathematically Correct Version)
# - FIX: Changed hand joint from x10 (Left Thumb) to x15 (Right Wrist).
#        Paper Section IV.C explicitly requires the "hand wrist position".
# - FIX: Replaced hardcoded magic numbers in depth fallback with constants.
# - Handles P->R mapping and B1/B2 batches.
# """
# # graph/datasets/wut17_adapter.py
# import os
# import cv2
# import csv
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from difflib import get_close_matches

# # ==================================================================== #
# # 1. CONFIGURATION
# # ==================================================================== #
# ACTIVITY_FOLDERS = [f"Activity{i}" for i in range(1, 8)]

# ACTIVITY_TO_SCENE = {
#     "Activity1": "SC1", "Activity2": "SC1", "Activity3": "SC1",
#     "Activity4": "SC2", "Activity5": "SC2", 
#     "Activity6": "SC3", "Activity7": "SC3"
# }

# ACTIVITY_MAP = {
#     "Activity1": "AC1", "Activity2": "AC2", "Activity3": "AC3",
#     "Activity4": "AC4", "Activity5": "AC5", 
#     "Activity6": "AC6", "Activity7": "AC7"
# }

# # ==================================================================== #
# # 2. MAPPINGS
# # ==================================================================== #
# ACTION_GROUND_TRUTH = {
#     # IGNORED (Background)
#     "standing still": None, "standing  still": None, "sitting": None, 
#     "sitting, reading": None, "sitting, eating": None, "stitting, eating": None,
#     "standing, hold cart": None, "not specified": None,
    
#     # a1: Approaching / Moving
#     "standing up": "a1", "standing, up": "a1",
#     "rotation": "a1", "rotaion": "a1", "rotaiton": "a1",
#     "walking forward": "a1", "walking, forward": "a1", "walking, backwards": "a1", "walking, towards observer": "a1",
    
#     # a2: Reaching / Collecting / Receiving
#     "standing, collect object": "a2", "extend arms, collect object": "a2", "extending arms, collect object": "a2",
#     "crouch, collect object": "a2", "crouching, collect object": "a2", "bend, retrieve object": "a2",
#     "standing, pick up phone": "a2", "standing pick up phone": "a2",
#     "sitting, receive object": "a2", "sitting receive object": "a2",
    
#     # a4: Pouring
#     "standing, move hand vertically": "a4",
    
#     # a5: Placing
#     "standing, place object": "a5", "standing place object": "a5",
#     "stading, place object": "a5", "standind, place object": "a5",
#     "standing , place object": "a5", "standing, place pbject": "a5", 
#     "extend arms, place object": "a5",
#     "bend, place object": "a5", "bend,place object": "a5",
    
#     # a6: Drinking
#     "sitting, drinking": "a6",
    
#     # a9: Passing
#     "standing, move object to other hand": "a9"
# }

# def get_action_id(action_str):
#     action_str = str(action_str).strip().lower()
#     action_str = " ".join(action_str.split())
#     if action_str in ACTION_GROUND_TRUTH:
#         return ACTION_GROUND_TRUTH[action_str]
#     valid_keys = [k.strip().lower() for k in ACTION_GROUND_TRUTH.keys() if ACTION_GROUND_TRUTH[k] is not None]
#     matches = get_close_matches(action_str, valid_keys, n=1, cutoff=0.85)
#     if matches:
#         original_key = next(k for k in ACTION_GROUND_TRUTH.keys() if k.strip().lower() == matches[0])
#         return ACTION_GROUND_TRUTH[original_key]
#     return None

# OBJECT_MAP = {
#     "table": "o_table", "high_table": "o_table", "chair": "o_object", "trolley": "o_object",
#     "door": "o_door", "book": "o_book", 
#     "bottle": "o_bottle", "kettle": "o_bottle", "spray": "o_bottle",
#     "glass": "o_glass", "cup": "o_glass", 
#     "plate": "o_bowl", "phone": "o_phone", "etui": "o_phone",
#     "cloth": "o_object", "box": "o_object", "package": "o_object",
#     "suitcase": "o_object", "broom": "o_object", "comb": "o_object"
# }

# # ==================================================================== #
# # 3. HELPERS & CAMERA INTRINSICS
# # ==================================================================== #
# FX, FY, CX, CY = 1050.0, 1050.0, 960.0, 540.0  # Doubled for 1920x1080
# DEPTH_SCALE = 1000.0  # Keep this, but we'll handle 8-bit encoding below

# # CRITICAL FIX: Define the exact column names for the RIGHT WRIST (Joint 15 in 1-indexed 34-joint format).
# # Paper Section IV.C: "we store the distance dHo between the hand wrist position..."
# WRIST_X, WRIST_Y, WRIST_Z = 'x15', 'y15', 'z15'

# def get_depth_z(depth_img, u, v):
#     u, v = int(round(u)), int(round(v))
#     h, w = depth_img.shape
    
#     # Safety check
#     if not (0 <= u < w and 0 <= v < h):
#         return None
    
#     depth_val = depth_img[v, u]
    
#     # Handle 8-bit encoded depth (0-255)
#     if depth_img.dtype == np.uint8 or depth_img.max() <= 255:
#         # CORRECT MAPPING: Senz3D depth range is 1-3 meters
#         # 0 → 1.0m, 255 → 3.0m
#         z_m = 1.0 + (float(depth_val) / 255.0) * 2.0
#     else:
#         # 16-bit depth in millimeters (fallback)
#         z_m = float(depth_val) / DEPTH_SCALE
    
#     # Valid range check (1-3 meters as per paper)
#     if 0.5 < z_m < 4.0:  # Slightly relaxed to handle noise
#         return z_m
    
#     return None

# def pixel_to_real_3d(u, v, z):
#     x = (u - CX) * z / FX
#     y = (v - CY) * z / FY
#     return round(x, 4), round(y, 4), round(z, 4)

# def find_depth_path(root_dir, activity_name, rec_name, frame_num):
#     clean_rec = rec_name.split('_')[0]
#     depth_dir = Path(root_dir) / "Depth_Images" / activity_name / clean_rec
    
#     # FIX: Try the CORRECT naming convention first (6 digits, no underscore)
#     # Based on diagnostic: files are named like "depth000026.png"
#     for fmt in [
#         f"depth{frame_num:06d}.png",        # <-- CORRECT format (matches your files)
#         f"depth_{frame_num:06d}.png",       # fallback with underscore
#         f"frame_{frame_num:06d}.png",       # fallback with "frame_" prefix
#         f"{frame_num:06d}.png",             # fallback bare number
#     ]:
#         p = depth_dir / fmt
#         if p.exists(): 
#             return p
#     return None

# # ==================================================================== #
# # 4. FILE DISCOVERY
# # ==================================================================== #
# def find_files_for_recording(root_dir, activity_name, person_num, cam_num):
#     p, c = person_num, cam_num
#     annot_dir = Path(root_dir) / "Annotation" / activity_name
#     obj_dir = Path(root_dir) / "Object_Information" / activity_name
#     skel_dir = Path(root_dir) / "Skeleton_Data" / activity_name / f"R{p}C{c}"
    
#     ann_files = []
#     std_ann = annot_dir / f"P{p}C{c}.xlsx"
#     if std_ann.exists():
#         ann_files.append(std_ann)
#     else:
#         for f in sorted(annot_dir.glob(f"P{p}C{c}B*.xlsx")):
#             ann_files.append(f)
            
#     if not ann_files or not skel_dir.exists(): return []

#     skel_files = list(skel_dir.glob("*W*.csv"))
#     if not skel_files: return []

#     results = []
#     for ann_file in ann_files:
#         stem = ann_file.stem 
#         suffix = stem.replace(f"P{p}C{c}", "") 
        
#         obj_file = None
#         if suffix:
#             obj_candidate = obj_dir / f"R{p}C{c}{suffix}.csv"
#             if obj_candidate.exists(): obj_file = obj_candidate
#             else:
#                 obj_candidate = obj_dir / f"R{p}C{c}.csv"
#                 if obj_candidate.exists(): obj_file = obj_candidate
#         else:
#             obj_candidate = obj_dir / f"R{p}C{c}.csv"
#             if obj_candidate.exists(): obj_file = obj_candidate
            
#         results.append({
#             "ann_file": ann_file,
#             "obj_file": obj_file,
#             "skel_file": skel_files[0],
#             "suffix": suffix,
#             "rec_name": f"R{p}C{c}{suffix}"
#         })
#     return results

# # ==================================================================== #
# # 5. EXTRACTION LOGIC
# # ==================================================================== #
# def process_recording(root_dir, activity_name, file_info, out_rows):
#     ann_path = file_info["ann_file"]
#     obj_path = file_info["obj_file"]
#     skel_3d_path = file_info["skel_file"]
#     rec_name = file_info["rec_name"]
    
#     parts = rec_name.replace("R", "").split("C")
#     person_num = int(parts[0].split('_')[0])
#     human_id = f"H{person_num}"
    
#     video_id = f"WUT_{activity_name}_{rec_name}"
#     activity_id = ACTIVITY_MAP.get(activity_name, "AC1")
#     scene_id = ACTIVITY_TO_SCENE.get(activity_name, "SC1")
#     activity_instance_id = f"WUT_ACI_{activity_name}_{rec_name}"

#     if not obj_path:
#         print(f"  [SKIP] Missing Object file for {ann_path.name}")
#         return

#     try:
#         df_annot = pd.read_excel(ann_path)
#         action_col = [c for c in df_annot.columns if c.lower() != 'frame'][0]
#         df_annot.rename(columns={action_col: 'Action'}, inplace=True)
#         df_annot['Frame'] = df_annot['Frame'].astype(int)

#         df_obj = pd.read_csv(obj_path)
#         df_obj['Frame'] = df_obj.index + 1  

#         df_skel = pd.read_csv(skel_3d_path, header=0)
#         df_skel.columns = [c.strip().strip("'").strip('"') for c in df_skel.columns]
        
#         if 'Frame number' in df_skel.columns:
#             df_skel.rename(columns={'Frame number': 'Frame', 'person count': 'PersonCount'}, inplace=True)
            
#         df_skel['Frame'] = df_skel['Frame'].astype(int)
        
#         # CRITICAL FIX: Check for the correct RIGHT WRIST columns (x15, y15, z15)
#         if WRIST_X not in df_skel.columns or WRIST_Y not in df_skel.columns or WRIST_Z not in df_skel.columns:
#             print(f"  [SKIP] Missing {WRIST_X}/{WRIST_Y}/{WRIST_Z} (Right Wrist) in {skel_3d_path.name}")
#             return
            
#         df_skel = df_skel[df_skel['PersonCount'] == 0]

#     except Exception as e:
#         print(f"  [ERROR] Failed to load data for {video_id}: {e}")
#         return

#     min_ann_frame = df_annot['Frame'].min()
#     min_skel_frame = df_skel['Frame'].min()
#     if min_ann_frame != min_skel_frame:
#         offset = min_ann_frame - min_skel_frame
#         df_skel['Frame'] = df_skel['Frame'] + offset
        
#     df_merged = pd.merge(df_annot, df_obj, on='Frame', how='inner')
#     df_merged = pd.merge(df_merged, df_skel, on='Frame', how='inner')
    
#     if df_merged.empty:
#         print(f"  [SKIP] Merge failed for {video_id}")
#         return

#     segments = []
#     current_action = None
#     start_frame = 0
#     for idx, row in df_merged.iterrows():
#         frame = int(row['Frame'])
#         action_str = str(row['Action']).strip().lower()
#         action_id = get_action_id(action_str)
        
#         if action_id != current_action:
#             if current_action is not None:
#                 segments.append({"start_frame": start_frame, "end_frame": frame - 1, "action_id": current_action})
#             current_action = action_id
#             start_frame = frame
#     if current_action is not None:
#         segments.append({"start_frame": start_frame, "end_frame": int(df_merged['Frame'].max()), "action_id": current_action})

#     rows_extracted = 0
#     for step_order, seg in enumerate(segments):
#         if seg['action_id'] is None: continue 
        
#         action_instance_id = f"WUT_AI_{activity_name}_{rec_name}_{step_order}"
#         df_segment = df_merged[(df_merged['Frame'] >= seg['start_frame']) & (df_merged['Frame'] <= seg['end_frame'])]

#         for _, row in df_segment.iterrows():
#             if WRIST_X not in row or WRIST_Y not in row or WRIST_Z not in row: continue
#             try:
#                 frame_num = int(row['Frame'])
                
#                 # Extract RIGHT WRIST coordinates
#                 hand_x, hand_y, hand_z = row[WRIST_X], row[WRIST_Y], row[WRIST_Z]
                
#                 # Skip if wrist coordinates are invalid/NaN
#                 if pd.isna(hand_x) or pd.isna(hand_y) or pd.isna(hand_z): continue
                
#                 hand_z = abs(hand_z)
                
#                 # Auto-detect if data is in mm or meters and convert to meters
#                 if abs(hand_z) > 10.0: 
#                     hand_x, hand_y, hand_z = hand_x/1000.0, hand_y/1000.0, hand_z/1000.0

#                 obj_id = "o_object"
#                 obj_xyz = (0.0, 0.0, 1.5)
#                 obj_u, obj_v = 0.0, 0.0

#                 # Inside process_recording(), find the object bounding box loop and replace it with this:

#                 for i in range(1, 4):
#                     label_col = f'label{i}'
#                     if label_col in row.index and pd.notna(row[label_col]):
#                         raw_label = str(row[label_col]).strip().lower()
#                         mapped_obj = OBJECT_MAP.get(raw_label, "o_object")
#                         if mapped_obj != "o_object":
#                             obj_id = mapped_obj
#                             col_x1, col_y1 = f'x{i*2-1}', f'y{i*2-1}'
#                             col_x2, col_y2 = f'x{i*2}', f'y{i*2}'
#                             if col_x1 in row.index and col_y1 in row.index and col_x2 in row.index and col_y2 in row.index:
#                                 x1, y1 = row[col_x1], row[col_y1]
#                                 x2, y2 = row[col_x2], row[col_y2]
                                
#                                 if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
#                                     x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
                                    
#                                     # CRITICAL FIX: Auto-detect if coordinates are normalized (0-1) or pixels
#                                     if x1_f <= 1.0 and x2_f <= 1.0 and y1_f <= 1.0 and y2_f <= 1.0:
#                                         # They are normalized (0 to 1), scale to ACTUAL depth image resolution (1920x1080)
#                                         obj_u = (x1_f + x2_f) / 2 * 1920  # <-- CHANGED from 640
#                                         obj_v = (y1_f + y2_f) / 2 * 1080  # <-- CHANGED from 480
#                                     else:
#                                         # They are already in pixels, but we need to check if they match 640x480 or 1920x1080
#                                         # If max value is around 640, scale up to 1920
#                                         if x2_f < 800:  # Likely 640x480 coordinates
#                                             obj_u = (x1_f + x2_f) / 2 * (1920 / 640)
#                                             obj_v = (y1_f + y2_f) / 2 * (1080 / 480)
#                                         else:  # Already 1920x1080
#                                             obj_u = (x1_f + x2_f) / 2
#                                             obj_v = (y1_f + y2_f) / 2
#                                     break

#                 depth_path = find_depth_path(root_dir, activity_name, rec_name, frame_num)
#                 if depth_path:
#                     depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
#                     if depth_img is not None:
#                         obj_z = get_depth_z(depth_img, obj_u, obj_v)
#                         if obj_z: 
#                             obj_xyz = pixel_to_real_3d(obj_u, obj_v, obj_z)
#                         else: 
#                             # FIX: Use defined constants and pixel_to_real_3d function instead of hardcoded numbers
#                             obj_xyz = pixel_to_real_3d(obj_u, obj_v, hand_z)

#                 out_rows.append({
#                     "video_id": video_id, "human_id": human_id, "scene_id": scene_id,
#                     "activity_id": activity_id, "activity_instance_id": activity_instance_id,
#                     "action_instance_id": action_instance_id, "step_order": step_order,
#                     "action_id": seg['action_id'], "object_id": obj_id, "frame": frame_num,
#                     "timestamp": round(frame_num / 60.0, 4),
#                     "hand_x": round(float(hand_x), 4), "hand_y": round(float(hand_y), 4), "hand_z": round(float(hand_z), 4),
#                     "object_x": round(float(obj_xyz[0]), 4), "object_y": round(float(obj_xyz[1]), 4), "object_z": round(float(obj_xyz[2]), 4),
#                 })
#                 rows_extracted += 1
#             except Exception: continue
            
#     print(f"  [SUCCESS] Extracted {rows_extracted} rows for {video_id}")

# def run_wut17_extraction(root_dir, output_csv="wut17_activities_1_to_7.csv"):
#     print(f"Starting WUT-17 V21 Extraction (Mathematically Correct)...")
#     out_rows = []
#     base_path = Path(root_dir)
#     annot_dir = base_path / "Annotation"
    
#     if not annot_dir.exists():
#         print(f"ERROR: Annotation directory not found at {annot_dir}")
#         return

#     for activity in ACTIVITY_FOLDERS:
#         if not (annot_dir / activity).exists():
#             print(f"[WARNING] Activity folder not found: {activity}.")
#             continue
            
#         print(f"\nProcessing {activity}...")
#         for person_num in range(1, 13):
#             for cam_num in range(1, 3):
#                 files = find_files_for_recording(base_path, activity, person_num, cam_num)
#                 if not files:
#                     print(f"  [SKIP] No matching files for P{person_num}C{cam_num} -> R{person_num}C{cam_num}")
#                     continue
#                 for file_info in files:
#                     process_recording(base_path, activity, file_info, out_rows)
            
#     if out_rows:
#         fieldnames = [
#             "video_id", "human_id", "scene_id", "activity_id", "activity_instance_id",
#             "action_instance_id", "step_order", "action_id", "object_id", "frame",
#             "timestamp", "hand_x", "hand_y", "hand_z", "object_x", "object_y", "object_z"
#         ]
#         with open(output_csv, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(out_rows)
#         print(f"\nSuccess! Extracted {len(out_rows)} frames -> {output_csv}")
#     else:
#         print("\nNo data extracted.")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Extract WUT-17 Activities 1-7 to CSV")
#     parser.add_argument("--root", type=str, required=True, help="Path to WUT-17 root folder")
#     parser.add_argument("--output", type=str, default="wut17_activities_1_to_7.csv", help="Output CSV filename")
#     args = parser.parse_args()
#     run_wut17_extraction(args.root, args.output)




# graph/datasets/wut17_adapter.py
"""
WUT-17 Dataset Adapter V22 (20-Joint Version)
- Extracts ALL 20 joints for GNN training (60 dimensions)
- Uses Right Wrist (Joint 15) for hand_x/y/z (backward compatibility)
- Handles 8-bit depth encoding correctly
- Auto-detects normalized vs pixel coordinates
"""

import os
import cv2
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches

# ==================================================================== #
# 1. CONFIGURATION
# ==================================================================== #
ACTIVITY_FOLDERS = [f"Activity{i}" for i in range(1, 8)]

ACTIVITY_TO_SCENE = {
    "Activity1": "SC1", "Activity2": "SC1", "Activity3": "SC1",
    "Activity4": "SC2", "Activity5": "SC2", 
    "Activity6": "SC3", "Activity7": "SC3"
}

ACTIVITY_MAP = {
    "Activity1": "AC1", "Activity2": "AC2", "Activity3": "AC3",
    "Activity4": "AC4", "Activity5": "AC5", 
    "Activity6": "AC6", "Activity7": "AC7"
}

# ==================================================================== #
# 2. MAPPINGS
# ==================================================================== #
ACTION_GROUND_TRUTH = {
    "standing still": None, "standing  still": None, "sitting": None, 
    "sitting, reading": None, "sitting, eating": None, "stitting, eating": None,
    "standing, hold cart": None, "not specified": None,
    
    "standing up": "a1", "standing, up": "a1",
    "rotation": "a1", "rotaion": "a1", "rotaiton": "a1",
    "walking forward": "a1", "walking, forward": "a1", "walking, backwards": "a1", "walking, towards observer": "a1",
    
    "standing, collect object": "a2", "extend arms, collect object": "a2", "extending arms, collect object": "a2",
    "crouch, collect object": "a2", "crouching, collect object": "a2", "bend, retrieve object": "a2",
    "standing, pick up phone": "a2", "standing pick up phone": "a2",
    "sitting, receive object": "a2", "sitting receive object": "a2",
    
    "standing, move hand vertically": "a4",
    
    "standing, place object": "a5", "standing place object": "a5",
    "stading, place object": "a5", "standind, place object": "a5",
    "standing , place object": "a5", "standing, place pbject": "a5", 
    "extend arms, place object": "a5",
    "bend, place object": "a5", "bend,place object": "a5",
    
    "sitting, drinking": "a6",
    
    "standing, move object to other hand": "a9"
}

def get_action_id(action_str):
    action_str = str(action_str).strip().lower()
    action_str = " ".join(action_str.split())
    if action_str in ACTION_GROUND_TRUTH:
        return ACTION_GROUND_TRUTH[action_str]
    valid_keys = [k.strip().lower() for k in ACTION_GROUND_TRUTH.keys() if ACTION_GROUND_TRUTH[k] is not None]
    matches = get_close_matches(action_str, valid_keys, n=1, cutoff=0.85)
    if matches:
        original_key = next(k for k in ACTION_GROUND_TRUTH.keys() if k.strip().lower() == matches[0])
        return ACTION_GROUND_TRUTH[original_key]
    return None

OBJECT_MAP = {
    "table": "o_table", "high_table": "o_table", "chair": "o_object", "trolley": "o_object",
    "door": "o_door", "book": "o_book", 
    "bottle": "o_bottle", "kettle": "o_bottle", "spray": "o_bottle",
    "glass": "o_glass", "cup": "o_glass", 
    "plate": "o_bowl", "phone": "o_phone", "etui": "o_phone",
    "cloth": "o_object", "box": "o_object", "package": "o_object",
    "suitcase": "o_object", "broom": "o_object", "comb": "o_object"
}

# ==================================================================== #
# 3. 20-JOINT CONFIGURATION
# ==================================================================== #
# Extract these 20 joints (excluding: 8, 9, 10, 15, 16, 17, 21, 25, 28, 29, 30, 31, 32, 33)
INCLUDED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26, 27]

# Right Wrist is Joint 15 (1-indexed in WUT-17 skeleton files)
WRIST_X, WRIST_Y, WRIST_Z = 'x15', 'y15', 'z15'

# ==================================================================== #
# 4. CAMERA INTRINSICS & DEPTH
# ==================================================================== #
FX, FY, CX, CY = 1050.0, 1050.0, 960.0, 540.0  # For 1920x1080 depth images
DEPTH_SCALE = 1000.0

def get_depth_z(depth_img, u, v):
    u, v = int(round(u)), int(round(v))
    h, w = depth_img.shape
    
    if not (0 <= u < w and 0 <= v < h):
        return None
    
    depth_val = depth_img[v, u]
    
    # Handle 8-bit encoded depth (0-255)
    if depth_img.dtype == np.uint8 or depth_img.max() <= 255:
        # Senz3D depth range is 1-3 meters
        z_m = 1.0 + (float(depth_val) / 255.0) * 2.0
    else:
        # 16-bit depth in millimeters
        z_m = float(depth_val) / DEPTH_SCALE
    
    if 0.5 < z_m < 4.0:
        return z_m
    
    return None

def pixel_to_real_3d(u, v, z):
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return round(x, 4), round(y, 4), round(z, 4)

def find_depth_path(root_dir, activity_name, rec_name, frame_num):
    clean_rec = rec_name.split('_')[0]
    depth_dir = Path(root_dir) / "Depth_Images" / activity_name / clean_rec
    
    for fmt in [
        f"depth{frame_num:06d}.png",
        f"depth_{frame_num:06d}.png",
        f"frame_{frame_num:06d}.png",
        f"{frame_num:06d}.png",
    ]:
        p = depth_dir / fmt
        if p.exists(): 
            return p
    return None

# ==================================================================== #
# 5. FILE DISCOVERY
# ==================================================================== #
def find_files_for_recording(root_dir, activity_name, person_num, cam_num):
    p, c = person_num, cam_num
    annot_dir = Path(root_dir) / "Annotation" / activity_name
    obj_dir = Path(root_dir) / "Object_Information" / activity_name
    skel_dir = Path(root_dir) / "Skeleton_Data" / activity_name / f"R{p}C{c}"
    
    ann_files = []
    std_ann = annot_dir / f"P{p}C{c}.xlsx"
    if std_ann.exists():
        ann_files.append(std_ann)
    else:
        for f in sorted(annot_dir.glob(f"P{p}C{c}B*.xlsx")):
            ann_files.append(f)
            
    if not ann_files or not skel_dir.exists(): return []

    skel_files = list(skel_dir.glob("*W*.csv"))
    if not skel_files: return []

    results = []
    for ann_file in ann_files:
        stem = ann_file.stem 
        suffix = stem.replace(f"P{p}C{c}", "") 
        
        obj_file = None
        if suffix:
            obj_candidate = obj_dir / f"R{p}C{c}{suffix}.csv"
            if obj_candidate.exists(): obj_file = obj_candidate
            else:
                obj_candidate = obj_dir / f"R{p}C{c}.csv"
                if obj_candidate.exists(): obj_file = obj_candidate
        else:
            obj_candidate = obj_dir / f"R{p}C{c}.csv"
            if obj_candidate.exists(): obj_file = obj_candidate
            
        results.append({
            "ann_file": ann_file,
            "obj_file": obj_file,
            "skel_file": skel_files[0],
            "suffix": suffix,
            "rec_name": f"R{p}C{c}{suffix}"
        })
    return results

# ==================================================================== #
# 6. EXTRACTION LOGIC (20 JOINTS)
# ==================================================================== #
def process_recording(root_dir, activity_name, file_info, out_rows):
    ann_path = file_info["ann_file"]
    obj_path = file_info["obj_file"]
    skel_3d_path = file_info["skel_file"]
    rec_name = file_info["rec_name"]
    
    parts = rec_name.replace("R", "").split("C")
    person_num = int(parts[0].split('_')[0])
    human_id = f"H{person_num}"
    
    video_id = f"WUT_{activity_name}_{rec_name}"
    activity_id = ACTIVITY_MAP.get(activity_name, "AC1")
    scene_id = ACTIVITY_TO_SCENE.get(activity_name, "SC1")
    activity_instance_id = f"WUT_ACI_{activity_name}_{rec_name}"

    if not obj_path:
        print(f"  [SKIP] Missing Object file for {ann_path.name}")
        return

    try:
        df_annot = pd.read_excel(ann_path)
        action_col = [c for c in df_annot.columns if c.lower() != 'frame'][0]
        df_annot.rename(columns={action_col: 'Action'}, inplace=True)
        df_annot['Frame'] = df_annot['Frame'].astype(int)

        df_obj = pd.read_csv(obj_path)
        df_obj['Frame'] = df_obj.index + 1  

        df_skel = pd.read_csv(skel_3d_path, header=0)
        df_skel.columns = [c.strip().strip("'").strip('"') for c in df_skel.columns]
        
        if 'Frame number' in df_skel.columns:
            df_skel.rename(columns={'Frame number': 'Frame', 'person count': 'PersonCount'}, inplace=True)
            
        df_skel['Frame'] = df_skel['Frame'].astype(int)
        
        # Check for Right Wrist columns
        if WRIST_X not in df_skel.columns or WRIST_Y not in df_skel.columns or WRIST_Z not in df_skel.columns:
            print(f"  [SKIP] Missing {WRIST_X}/{WRIST_Y}/{WRIST_Z} in {skel_3d_path.name}")
            return
            
        df_skel = df_skel[df_skel['PersonCount'] == 0]

    except Exception as e:
        print(f"  [ERROR] Failed to load data for {video_id}: {e}")
        return

    min_ann_frame = df_annot['Frame'].min()
    min_skel_frame = df_skel['Frame'].min()
    if min_ann_frame != min_skel_frame:
        offset = min_ann_frame - min_skel_frame
        df_skel['Frame'] = df_skel['Frame'] + offset
        
    df_merged = pd.merge(df_annot, df_obj, on='Frame', how='inner')
    df_merged = pd.merge(df_merged, df_skel, on='Frame', how='inner')
    
    if df_merged.empty:
        print(f"  [SKIP] Merge failed for {video_id}")
        return

    segments = []
    current_action = None
    start_frame = 0
    for idx, row in df_merged.iterrows():
        frame = int(row['Frame'])
        action_str = str(row['Action']).strip().lower()
        action_id = get_action_id(action_str)
        
        if action_id != current_action:
            if current_action is not None:
                segments.append({"start_frame": start_frame, "end_frame": frame - 1, "action_id": current_action})
            current_action = action_id
            start_frame = frame
    if current_action is not None:
        segments.append({"start_frame": start_frame, "end_frame": int(df_merged['Frame'].max()), "action_id": current_action})

    rows_extracted = 0
    for step_order, seg in enumerate(segments):
        if seg['action_id'] is None: continue 
        
        action_instance_id = f"WUT_AI_{activity_name}_{rec_name}_{step_order}"
        df_segment = df_merged[(df_merged['Frame'] >= seg['start_frame']) & (df_merged['Frame'] <= seg['end_frame'])]

        for _, row in df_segment.iterrows():
            if WRIST_X not in row or WRIST_Y not in row or WRIST_Z not in row: continue
            try:
                frame_num = int(row['Frame'])
                
                # Extract RIGHT WRIST coordinates (Joint 15)
                hand_x, hand_y, hand_z = row[WRIST_X], row[WRIST_Y], row[WRIST_Z]
                
                if pd.isna(hand_x) or pd.isna(hand_y) or pd.isna(hand_z): continue
                
                hand_z = abs(hand_z)
                
                # Auto-detect if data is in mm or meters
                if abs(hand_z) > 10.0: 
                    hand_x, hand_y, hand_z = hand_x/1000.0, hand_y/1000.0, hand_z/1000.0

                # ==================================================================== #
                # NEW: Extract ALL 20 JOINTS for GNN training
                # ==================================================================== #
                joint_data = {}
                for joint_id in INCLUDED_JOINTS:
                    x_col = f'x{joint_id}'
                    y_col = f'y{joint_id}'
                    z_col = f'z{joint_id}'
                    
                    if x_col in row and y_col in row and z_col in row:
                        x_val = row[x_col]
                        y_val = row[y_col]
                        z_val = row[z_col]
                        
                        # Skip if any coordinate is NaN
                        if pd.isna(x_val) or pd.isna(y_val) or pd.isna(z_val):
                            joint_data[joint_id] = (0.0, 0.0, 0.0)
                            continue
                        
                        # Convert to meters if needed
                        if abs(z_val) > 10.0:
                            x_val, y_val, z_val = x_val/1000.0, y_val/1000.0, z_val/1000.0
                        
                        joint_data[joint_id] = (float(x_val), float(y_val), float(z_val))
                    else:
                        # Joint not found in CSV, use zeros
                        joint_data[joint_id] = (0.0, 0.0, 0.0)

                # ==================================================================== #
                # Object 3D coordinates (unchanged)
                # ==================================================================== #
                obj_id = "o_object"
                obj_xyz = (0.0, 0.0, 1.5)
                obj_u, obj_v = 0.0, 0.0

                for i in range(1, 4):
                    label_col = f'label{i}'
                    if label_col in row.index and pd.notna(row[label_col]):
                        raw_label = str(row[label_col]).strip().lower()
                        mapped_obj = OBJECT_MAP.get(raw_label, "o_object")
                        if mapped_obj != "o_object":
                            obj_id = mapped_obj
                            col_x1, col_y1 = f'x{i*2-1}', f'y{i*2-1}'
                            col_x2, col_y2 = f'x{i*2}', f'y{i*2}'
                            if col_x1 in row.index and col_y1 in row.index and col_x2 in row.index and col_y2 in row.index:
                                x1, y1 = row[col_x1], row[col_y1]
                                x2, y2 = row[col_x2], row[col_y2]
                                
                                if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
                                    x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
                                    
                                    if x1_f <= 1.0 and x2_f <= 1.0 and y1_f <= 1.0 and y2_f <= 1.0:
                                        obj_u = (x1_f + x2_f) / 2 * 1920
                                        obj_v = (y1_f + y2_f) / 2 * 1080
                                    else:
                                        if x2_f < 800:
                                            obj_u = (x1_f + x2_f) / 2 * (1920 / 640)
                                            obj_v = (y1_f + y2_f) / 2 * (1080 / 480)
                                        else:
                                            obj_u = (x1_f + x2_f) / 2
                                            obj_v = (y1_f + y2_f) / 2
                                    break

                depth_path = find_depth_path(root_dir, activity_name, rec_name, frame_num)
                if depth_path:
                    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                    if depth_img is not None:
                        obj_z = get_depth_z(depth_img, obj_u, obj_v)
                        if obj_z: 
                            obj_xyz = pixel_to_real_3d(obj_u, obj_v, obj_z)
                        else: 
                            obj_xyz = pixel_to_real_3d(obj_u, obj_v, hand_z)

                # ==================================================================== #
                # Build output row with ALL 20 JOINTS
                # ==================================================================== #
                out_row = {
                    "video_id": video_id, "human_id": human_id, "scene_id": scene_id,
                    "activity_id": activity_id, "activity_instance_id": activity_instance_id,
                    "action_instance_id": action_instance_id, "step_order": step_order,
                    "action_id": seg['action_id'], "object_id": obj_id, "frame": frame_num,
                    "timestamp": round(frame_num / 60.0, 4),
                    "hand_x": round(float(hand_x), 4), "hand_y": round(float(hand_y), 4), "hand_z": round(float(hand_z), 4),
                    "object_x": round(float(obj_xyz[0]), 4), "object_y": round(float(obj_xyz[1]), 4), "object_z": round(float(obj_xyz[2]), 4),
                }
                
                # Add all 20 joints as joint_0_x, joint_0_y, joint_0_z, joint_1_x, etc.
                for joint_id in INCLUDED_JOINTS:
                    x, y, z = joint_data[joint_id]
                    out_row[f'joint_{joint_id}_x'] = round(x, 4)
                    out_row[f'joint_{joint_id}_y'] = round(y, 4)
                    out_row[f'joint_{joint_id}_z'] = round(z, 4)
                
                out_rows.append(out_row)
                rows_extracted += 1
            except Exception as e:
                continue
            
    print(f"  [SUCCESS] Extracted {rows_extracted} rows for {video_id}")

def run_wut17_extraction(root_dir, output_csv="wut17_activities_1_to_7_20joints.csv"):
    print(f"Starting WUT-17 V22 Extraction (20-Joint Version)...")
    out_rows = []
    base_path = Path(root_dir)
    annot_dir = base_path / "Annotation"
    
    if not annot_dir.exists():
        print(f"ERROR: Annotation directory not found at {annot_dir}")
        return

    for activity in ACTIVITY_FOLDERS:
        if not (annot_dir / activity).exists():
            print(f"[WARNING] Activity folder not found: {activity}.")
            continue
            
        print(f"\nProcessing {activity}...")
        for person_num in range(1, 13):
            for cam_num in range(1, 3):
                files = find_files_for_recording(base_path, activity, person_num, cam_num)
                if not files:
                    print(f"  [SKIP] No matching files for P{person_num}C{cam_num} -> R{person_num}C{cam_num}")
                    continue
                for file_info in files:
                    process_recording(base_path, activity, file_info, out_rows)
            
    if out_rows:
        # Build fieldnames dynamically to include all 20 joints
        fieldnames = [
            "video_id", "human_id", "scene_id", "activity_id", "activity_instance_id",
            "action_instance_id", "step_order", "action_id", "object_id", "frame",
            "timestamp", "hand_x", "hand_y", "hand_z", "object_x", "object_y", "object_z"
        ]
        
        # Add joint columns
        for joint_id in INCLUDED_JOINTS:
            fieldnames.extend([f'joint_{joint_id}_x', f'joint_{joint_id}_y', f'joint_{joint_id}_z'])
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nSuccess! Extracted {len(out_rows)} frames -> {output_csv}")
    else:
        print("\nNo data extracted.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract WUT-17 Activities 1-7 with 20 Joints")
    parser.add_argument("--root", type=str, required=True, help="Path to WUT-17 root folder")
    parser.add_argument("--output", type=str, default="wut17_activities_1_to_7_20joints.csv", help="Output CSV filename")
    args = parser.parse_args()
    run_wut17_extraction(args.root, args.output)