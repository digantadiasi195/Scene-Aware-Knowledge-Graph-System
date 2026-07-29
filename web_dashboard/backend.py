# backend.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import shutil, os, cv2, json, traceback, uuid, sys
import numpy as np
import torch
from PIL import Image
from threading import Lock
import subprocess
from neo4j import GraphDatabase
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from ultralytics import YOLO  # ✅ For Object-Only View & Pose
import open_clip
import warnings
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
warnings.filterwarnings( "ignore", category=UserWarning, message=r".*_max_size.*")

from perception.depth_estimator import DepthEstimator
from perception.dynamic_mapper import DynamicMapper
from perception.scene_filter import filter_objects_by_scene
from reasoning.activity_inference import infer_macro_activity
from reasoning.llm_reasoner import LLMReasoner
from reasoning.relation_engine import RelationEngine
from reasoning.intent_engine import IntentEngine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

try:
    from graph.inference import predict_action
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

app = FastAPI(title="PhD Scene-Aware KG Dashboard - Real-Time")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR = "./sessions"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# ============================================================
# YOLO COCO CLASSES (For Object-Only Mapping)
# ============================================================
YOLO_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ============================================================
# STUFF CLASS FILTER
# ============================================================
STUFF_CLASSES = {
    'wall', 'floor', 'ceiling', 'sky', 'grass', 'dirt', 'road', 'pavement', 'sidewalk',
    'tree', 'plant', 'bush', 'leaves', 'mountain', 'hill', 'rock', 'mud', 'fog',
    'building', 'house', 'skyscraper', 'bridge', 'tent', 'fence', 'railing', 'stairs',
    'window', 'door', 'mirror-stuff', 'cardboard', 'textile-other', 'cloth', 'clothes',
    'blanket', 'towel', 'curtain', 'rug', 'carpet', 'mat', 'paper', 'plastic', 'metal',
    'wood', 'stone', 'cabinet', 'cupboard', 'shelf', 'counter', 'table', 'desk-stuff',
    'unknown', 'thing', 'stuff', 'marge wall'
}

def is_stuff_class(label):
    if not label: return True
    label_lower = label.lower().strip()
    return label_lower in STUFF_CLASSES

# ============================================================
# SESSION STATE
# ============================================================
class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.video_path = None
        self.video_url = None
        self.processed_frames = {}
        self.scene_occurrences = []
        self.scenes_seen = {}
        self.object_history = {}
        self.human_history = {}
        self.current_scene = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.lock = Lock()
        self._last_scene_id = None
        self._current_occurrence_id = 0
        
    def add_frame_analysis(self, frame_num: int, analysis: dict):
        with self.lock:
            self.processed_frames[frame_num] = analysis
            self.current_frame = frame_num
            
            scene_id = analysis.get('scene_id', 'SC1')
            scene_name = analysis.get('scene', 'Unknown')
            
            if scene_id != self._last_scene_id:
                self._current_occurrence_id += 1
                occurrence_id = f"{scene_id}_occ{self._current_occurrence_id}"
                self.scene_occurrences.append({
                    'occurrence_id': occurrence_id, 'scene_id': scene_id, 'scene_name': scene_name,
                    'frame_start': frame_num, 'frame_end': frame_num, 'objects': [], 'humans': []
                })
                self._last_scene_id = scene_id
            else:
                if self.scene_occurrences:
                    self.scene_occurrences[-1]['frame_end'] = frame_num
            
            for obj in analysis.get('objects', []):
                obj_id = obj.get('object_id', 'unknown')
                obj_label = obj.get('label', 'unknown')
                if obj_id not in self.object_history:
                    self.object_history[obj_id] = {'label': obj_label, 'occurrences': [], 'positions': []}
                
                current_occ = self.scene_occurrences[-1]
                occ_id = current_occ['occurrence_id']
                if occ_id not in self.object_history[obj_id]['occurrences']:
                    self.object_history[obj_id]['occurrences'].append(occ_id)
                self.object_history[obj_id]['positions'].append({'frame': frame_num, 'xyz': obj.get('xyz', [0,0,0]), 'bbox': obj.get('bbox', [0,0,0,0])})
                if obj_id not in [o['object_id'] for o in current_occ['objects']]:
                    current_occ['objects'].append(obj)
            
            for human in analysis.get('humans', []):
                human_id = human.get('id', 'human_1')
                if human_id not in self.human_history:
                    self.human_history[human_id] = {'occurrences': [], 'positions': []}
                current_occ = self.scene_occurrences[-1]
                occ_id = current_occ['occurrence_id']
                if occ_id not in self.human_history[human_id]['occurrences']:
                    self.human_history[human_id]['occurrences'].append(occ_id)
                self.human_history[human_id]['positions'].append({'frame': frame_num, 'xyz': human.get('xyz', [0,0,0])})
                if human_id not in [h['id'] for h in current_occ['humans']]:
                    current_occ['humans'].append(human)
    
    def get_hierarchical_graph(self):
        graph = {
            'root': {'id': 'root', 'label': 'Complete Context', 'type': 'root'},
            'scene_occurrences': [], 'temporal_connections': [], 'cross_occurrence_connections': []
        }
        for occ in self.scene_occurrences:
            occ_node = {
                'occurrence_id': occ['occurrence_id'], 'scene_id': occ['scene_id'],
                'label': f"{occ['scene_name']} ({occ['scene_id']})",
                'frame_range': [occ['frame_start'], occ['frame_end']], 'objects': [], 'humans': []
            }
            for obj in occ['objects']:
                obj_id = obj.get('object_id', 'unknown')
                occ_node['objects'].append({
                    'id': obj_id, 'label': obj.get('label', 'unknown'), 'xyz': obj.get('xyz', [0,0,0]),
                    'appears_in_occurrences': [occ['occurrence_id']]
                })
            for human_id, human_data in self.human_history.items():
                if occ['occurrence_id'] in human_data['occurrences']:
                    occ_node['humans'].append({'id': human_id, 'label': 'Person', 'appears_in_occurrences': [occ['occurrence_id']]})
            graph['scene_occurrences'].append(occ_node)
        return graph

sessions = {}
sessions_lock = Lock()

def get_session(session_id: str) -> SessionState:
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = SessionState(session_id)
        return sessions[session_id]

# ============================================================
# GLOBAL MODELS (HYBRID APPROACH)
# ============================================================
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Mask2Former: For FULL SCENE Panoptic Segmentation (Segmentation View)
print("🚀 Loading Mask2Former for Full Panoptic Segmentation...")
seg_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(device).eval()
id2label = seg_model.config.id2label

# 2. YOLOv8: For OBJECT-ONLY Segmentation & Pose (Object View & Logic)
print("🚀 Loading YOLOv8 for Object Detection & Pose...")
yolo_seg_model = YOLO('yolov8x-seg.pt')
pose_model = YOLO('yolov8x-pose.pt')

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
clip_model.to(device).eval()

print("🧠 Loading global models...")
depth_estimator = DepthEstimator(device=str(device))
dynamic_mapper = DynamicMapper(get_neo4j_driver(), NEO4J_DATABASE)
llm_reasoner = LLMReasoner(model_name="llama3.1:8b")
print("✅ All models loaded successfully!")

# ============================================================
# FRAME PROCESSING
# ============================================================
def process_single_frame(frame_rgb, frame_num, session: SessionState):
    frame_small = cv2.resize(frame_rgb, (640, 480))
    
    # 1. Scene Classification
    all_scene_names = dynamic_mapper.get_all_scenes_for_clip()
    scene_tokens = open_clip.tokenize(all_scene_names).to(device)
    with torch.no_grad():
        scene_text_feats = clip_model.encode_text(scene_tokens)
        scene_text_feats /= scene_text_feats.norm(dim=-1, keepdim=True)
        img_t = clip_preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
        img_feat = clip_model.encode_image(img_t)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        scene_idx = (img_feat @ scene_text_feats.T).softmax(dim=-1).argmax().item()
        scene_name = all_scene_names[scene_idx]
        scene_id = dynamic_mapper.map_scene(scene_name)
    
    # 2. FULL SCENE Panoptic Segmentation (For Segmentation View)
    with torch.no_grad():
        inputs = seg_processor(images=frame_small, return_tensors="pt").to(device)
        outputs = seg_model(**inputs)
        result = seg_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[frame_small.shape[:2]], label_ids_to_fuse=[]
        )[0]
        pan_seg_full = result["segmentation"].cpu().numpy()
        segments_full = result["segments_info"]
    
    # 3. OBJECT-ONLY Segmentation (For Object View & Logic)
    yolo_results = yolo_seg_model(frame_small, verbose=False)[0]
    detected_objects = []
    pan_seg_obj = np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.int32)
    segments_obj = []
    
    if yolo_results.masks is not None:
        masks = yolo_results.masks.data.cpu().numpy()
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        classes = yolo_results.boxes.cls.cpu().numpy()
        
        for i in range(len(masks)):
            binary_mask = (masks[i] > 0.5).astype(np.uint8)
            if binary_mask.sum() < 500:
                continue
            
            seg_id = i + 1
            pan_seg_obj[binary_mask > 0] = seg_id
            
            cls_idx = int(classes[i])
            label = YOLO_COCO_CLASSES[cls_idx] if cls_idx < len(YOLO_COCO_CLASSES) else "unknown"
            
            # Filter out stuff classes for object logic
            if not is_stuff_class(label):
                x1, y1, x2, y2 = boxes[i]
                obj_id = dynamic_mapper.map_object(label)
                detected_objects.append({
                    "object_id": obj_id, "label": label, "bbox": [int(x1), int(y1), int(x2), int(y2)], "xyz": [0, 0, 2]
                })
                segments_obj.append({
                    "id": seg_id, "label_id": cls_idx, "label": label, "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
    
    # 4. Pose estimation
    pose_results = pose_model(frame_small, verbose=False)[0]
    real_hand_xyz = (0.0, 0.0, 1.5)
    skeleton_joints = {"humans": []}
    
    if pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
        kpts = pose_results.keypoints.xy[0]
        confs = pose_results.keypoints.conf[0]
        if len(kpts) > 10 and confs[10] > 0.4:
            x2d, y2d = float(kpts[10][0]), float(kpts[10][1])
            real_hand_xyz = (float((x2d - 320) * 1.5 / 500), float((y2d - 240) * 1.5 / 500), 1.5)
            
        for person_idx in range(len(pose_results.keypoints.xy)):
            pkpts = pose_results.keypoints.xy[person_idx]
            pconfs = pose_results.keypoints.conf[person_idx]
            
            joints = []
            valid_count = 0
            for i in range(len(pkpts)):
                if pconfs[i] > 0.3:
                    x, y = float(pkpts[i][0]) / 640.0, float(pkpts[i][1]) / 480.0
                    joints.append({"x": x, "y": y, "conf": float(pconfs[i]), "id": i})
                    valid_count += 1
                else:
                    joints.append(None)
                    
            if valid_count < 3:
                continue
                
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
                (13, 15), (12, 14), (14, 16)
            ]
            valid_connections = [(s, e) for s, e in connections if joints[s] is not None and joints[e] is not None]
            
            skeleton_joints["humans"].append({
                "joints": joints, "connections": valid_connections, "person_id": person_idx
            })
    
    # 5. LLM Cognitive Reasoning
    predicted_goal = None
    if frame_num % 15 == 0:
        try:
            graph_relations = [{'subject': 'human', 'relation': 'near', 'object': obj['label']} for obj in detected_objects if obj['label'].lower() not in ['person', 'human']]
            predicted_goal = llm_reasoner.predict_goal(
                scene_name=scene_name, scene_id=scene_id, detected_objects=detected_objects,
                atomic_action="a1", graph_relations=graph_relations
            )
        except Exception as e:
            print(f"[WARN] LLM reasoning failed for frame {frame_num}: {e}")
            predicted_goal = None
    
    # 6. Build analysis
    analysis = {
        'frame_num': frame_num, 'scene': scene_name, 'scene_id': scene_id,
        'objects': detected_objects, 'humans': [{'id': 'human_1', 'xyz': real_hand_xyz}],
        'hand_xyz': list(real_hand_xyz), 'skeleton': skeleton_joints, 'predicted_goal': predicted_goal
    }
    
    # 7. Save FULL SCENE Panoptic Visualization (Segmentation View)
    seg_frame = create_panoptic_visualization(frame_small, pan_seg_full, segments_full, id2label)
    cv2.imwrite(os.path.join(FRAME_DIR, f"{session.session_id}_seg_{frame_num}.jpg"), seg_frame)
    analysis['segmentation_url'] = f"/api/segmentation/{session.session_id}/{frame_num}"

    # 8. Save OBJECT-ONLY Visualization (Object View)
    obj_view_frame = create_object_only_visualization(frame_small, pan_seg_obj, segments_obj)
    cv2.imwrite(os.path.join(FRAME_DIR, f"{session.session_id}_objview_{frame_num}.jpg"), obj_view_frame)
    analysis['object_view_url'] = f"/api/object-view/{session.session_id}/{frame_num}"
    
    session.add_frame_analysis(frame_num, analysis)
    return analysis

def create_panoptic_visualization(frame_rgb, pan_seg, segments, id2label):
    """Creates FULL SCENE segmentation (walls, floors, sky, objects)"""
    vis_frame = frame_rgb.copy()
    for seg in segments:
        mask = (pan_seg == seg["id"])
        if mask.sum() == 0: continue
            
        lbl_id = seg["label_id"]
        lbl = id2label.get(lbl_id, id2label.get(str(lbl_id), "unknown"))
        
        rng = np.random.RandomState(lbl_id)
        color = tuple(rng.randint(50, 255, size=3).tolist())
        
        overlay = vis_frame.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.85, vis_frame, 0.15, 0, vis_frame)
        
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (text_w, text_h), _ = cv2.getTextSize(lbl, font, font_scale, thickness)
            text_x, text_y = max(0, x1), max(text_h + 2, y1)
            cv2.rectangle(vis_frame, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(vis_frame, lbl, (text_x + 1, text_y - 1), font, font_scale, (255, 255, 255), thickness)
    return vis_frame

def create_object_only_visualization(frame_rgb, pan_seg, segments):
    """
    Creates OBJECT-ONLY segmentation:
    - Excludes ALL humans/persons
    - Excludes background/stuff classes (walls, floors, etc.)
    - Shows ONLY actual objects (bottles, chairs, tables, etc.) with colors
    - Background remains the natural original video frame
    """
    # Start with the ORIGINAL frame as the base
    vis_frame = frame_rgb.copy()
    
    # Labels to exclude (humans + stuff classes)
    EXCLUDE_LABELS = {'person', 'human', 'people'}
    
    for seg in segments:
        mask = (pan_seg == seg["id"])
        if mask.sum() == 0: 
            continue

        lbl = seg.get("label", "unknown").lower()
        
        # ✅ FILTER: Skip humans AND background/stuff classes
        if lbl in EXCLUDE_LABELS or is_stuff_class(lbl): 
            continue

        lbl_id = seg.get("label_id", 0)
        rng = np.random.RandomState(lbl_id)
        color = tuple(int(c) for c in rng.randint(50, 255, size=3))

        # Apply deep colored mask ONLY to object pixels
        overlay = vis_frame.copy()
        overlay[mask] = color
        # 85% deep color, 15% original texture
        cv2.addWeighted(overlay, 0.85, vis_frame, 0.15, 0, vis_frame)

        # Draw clean text label
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (text_w, text_h), _ = cv2.getTextSize(lbl, font, font_scale, thickness)
            text_x, text_y = max(0, x1), max(text_h + 2, y1)
            
            # Black background for text readability
            cv2.rectangle(vis_frame, (text_x, text_y - text_h - 2), 
                         (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
            # White text
            cv2.putText(vis_frame, lbl, (text_x + 1, text_y - 1), 
                       font, font_scale, (255, 255, 255), thickness)
            
    return vis_frame

# ============================================================
# ENDPOINTS
# ============================================================
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    session = get_session(session_id)
    original_path = os.path.join(UPLOAD_DIR, f"{session_id}_original_{file.filename}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"🔄 Converting video to browser-compatible H.264 MP4...")
    mp4_path = os.path.join(UPLOAD_DIR, f"{session_id}.mp4")
    try:
        subprocess.run(['ffmpeg', '-y', '-i', original_path, '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-crf', '23', '-movflags', '+faststart', mp4_path], check=True, capture_output=True)
        if os.path.exists(original_path): os.remove(original_path)
        video_path = mp4_path
        print("✅ Video conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        video_path = original_path
    
    cap = cv2.VideoCapture(video_path)
    session.video_path = video_path
    session.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    session.fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return {"session_id": session_id, "total_frames": session.total_frames, "fps": session.fps, "video_url": f"/api/video/{session_id}"}

@app.get("/api/video/{session_id}")
async def get_video(session_id: str):
    session = get_session(session_id)
    if not session.video_path or not os.path.exists(session.video_path):
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    return FileResponse(session.video_path, media_type="video/mp4")

@app.post("/api/process-frame/")
async def process_frame(data: dict):
    session = get_session(data.get('session_id'))
    frame_num = data.get('frame_num', 0)
    if not session.video_path or frame_num in session.processed_frames:
        return session.processed_frames.get(frame_num, {"error": "No video or frame not found"})
    
    cap = cv2.VideoCapture(session.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret: return JSONResponse(status_code=400, content={"error": "Frame not found"})
    
    return process_single_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_num, session)

@app.get("/api/segmentation/{session_id}/{frame_num}")
async def get_segmentation(session_id: str, frame_num: int):
    path = os.path.join(FRAME_DIR, f"{session_id}_seg_{frame_num}.jpg")
    return FileResponse(path, media_type="image/jpeg") if os.path.exists(path) else JSONResponse(status_code=404, content={"error": "Not found"})

@app.get("/api/object-view/{session_id}/{frame_num}")
async def get_object_view(session_id: str, frame_num: int):
    path = os.path.join(FRAME_DIR, f"{session_id}_objview_{frame_num}.jpg")
    return FileResponse(path, media_type="image/jpeg") if os.path.exists(path) else JSONResponse(status_code=404, content={"error": "Not found"})

@app.get("/api/hierarchical-graph/{session_id}")
async def get_hierarchical_graph(session_id: str):
    return get_session(session_id).get_hierarchical_graph()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)