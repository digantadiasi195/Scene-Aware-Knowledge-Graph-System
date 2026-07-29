# main_pipeline.py
import cv2
import torch
import time
from tqdm import tqdm
from neo4j import GraphDatabase

# Local Imports
from perception.detector import ObjectDetector
from perception.pose_estimator import PoseEstimator
from perception.scene_classifier import SceneClassifier
from perception.action_recognizer import ActionRecognizer
from reasoning.gnn_reasoner import GATReasoner

# V2 Graph Imports
import sys
sys.path.insert(0, 'graph')
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from cv_integration import ingest_pose_observation, ingest_action_instance
from stats_learning import learn_all
from inference import predict_action
from geometry import geometry_from_coords

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize CV Models
    print("Loading CV Models...")
    detector = ObjectDetector()
    pose_estimator = PoseEstimator()
    scene_cls = SceneClassifier(device)
    action_recog = ActionRecognizer(device)
    
    # 2. Initialize GNN
    print("Loading GNN Reasoner...")
    gnn = GATReasoner().to(device)
    gnn.eval()

    # 3. Connect to Neo4j
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 4. Video Setup
    VIDEO_SOURCE = "0004-M.avi" # <--- CHANGE THIS
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # State tracking
    current_action = None
    start_frame = 0
    inst_id = 0
    prev_obj_id = None
    prev_obj_xyz = None

    print("Starting Real-World Perception Loop...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        frame_small = cv2.resize(frame, (640, 480))
        depth_frame = None # Set to your ZED/Azure Kinect depth array if available
        
        # --- STEP 1: Perception ---
        scene_id = scene_cls.predict(frame_small)
        objects = detector.detect(frame_small, depth_frame)
        hand_xyz = pose_estimator.get_hand_xyz(frame_small, depth_frame)
        cv_action_id = action_recog.predict(frame_small)
        
        if not hand_xyz or not objects: 
            continue

        # --- STEP 2: Graph Ingestion (Observation Layer) ---
        ingest_pose_observation(
            driver, video_id="V_LIVE_01", frame=frame_idx, timestamp=frame_idx/25.0,
            human_id="H1", scene_id=scene_id,
            hand_xyz=hand_xyz, nearby_objects=objects
        )

        # --- STEP 3: Graph Ingestion (Action Instance Layer) ---
        if cv_action_id != current_action:
            # Close the PREVIOUS action instance if one exists
            if current_action is not None and prev_obj_id is not None:
                ingest_action_instance(
                    driver, instance_id=f"INST_{inst_id}", video_id="V_LIVE_01",
                    human_id="H1", scene_id=scene_id, action_id=current_action, 
                    object_id=prev_obj_id, start_frame=start_frame, end_frame=frame_idx - 1,
                    hand_xyz=hand_xyz, object_xyz=prev_obj_xyz
                )
                inst_id += 1
                
            # Start the NEW action instance
            current_action = cv_action_id
            prev_obj_id = objects[0]["object_id"]
            prev_obj_xyz = objects[0]["xyz"]
            start_frame = frame_idx

        # --- STEP 4: V2 Probabilistic Inference ---
        obs_for_graph = []
        for obj in objects:
            geom = geometry_from_coords(hand_xyz, obj["xyz"])
            obs_for_graph.append({
                "object_id": obj["object_id"], 
                "d_Ho": geom["d_Ho"], 
                "theta": geom["theta"], 
                "e": geom["e"]
            })
            
        graph_preds = predict_action(driver, scene_id, obs_for_graph, human_id="H1", prev_action_id=current_action)
        top_graph_pred = graph_preds[0] if graph_preds else None

        # --- STEP 5: GNN Deep Reasoning ---
        # To maintain real-time FPS, we pass the geometry we just calculated 
        # and use neutral priors (0.5) for the GNN. (In production, you would 
        # cache the Neo4j priors in RAM when the scene changes).
        gnn_objects = []
        for obj in objects:
            geom = geometry_from_coords(hand_xyz, obj["xyz"])
            gnn_objects.append({
                "xyz": obj["xyz"],
                "scene_obj_p": 0.5,      # Neutral prior for real-time speed
                "mu_dHo": geom["d_Ho"],  # Use current distance as proxy
                "sigma2_dHo": 0.1
            })
            
        gnn_action_id, gnn_conf = gnn.predict_from_context(scene_id, gnn_objects, hand_xyz, device)

        # --- VISUALIZATION ---
        display_frame = frame_small.copy()
        cv2.putText(display_frame, f"Scene: {scene_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if top_graph_pred:
            # FIX 1: Use 'combined_log_score' instead of the old 'combined_score'
            log_score = top_graph_pred['combined_log_score']
            cv2.putText(display_frame, f"Graph: {top_graph_pred['action_name']} (logP:{log_score:.2f})", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        cv2.putText(display_frame, f"GNN Deep: {gnn_action_id} ({gnn_conf:.3f})", 
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("PhD Perception Pipeline", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # =========================================================================
    # FIX 2: Close the FINAL action instance of the video
    # =========================================================================
    if current_action is not None and prev_obj_id is not None:
        ingest_action_instance(
            driver, instance_id=f"INST_{inst_id}", video_id="V_LIVE_01",
            human_id="H1", scene_id=scene_id, action_id=current_action, 
            object_id=prev_obj_id, start_frame=start_frame, end_frame=total_frames - 1,
            hand_xyz=hand_xyz, object_xyz=prev_obj_xyz
        )
        inst_id += 1
        print(f"Saved final action instance: INST_{inst_id-1}")

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL STEP: Update Learned Statistics ---
    print("Video finished. Running V2 Statistical Learning over new instances...")
    learn_all(driver)
    print("System Ready. Database updated with live observations.")
    driver.close()

if __name__ == "__main__":
    main()