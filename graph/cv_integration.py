# cv_integration.py
"""
Integration interface for your future computer-vision pipeline.

The intended live architecture:

    YOLO / Mask2Former  --object detections-->  ingest_object_detection()
    Pose model          --hand/skeleton-->       ingest_pose_observation()
    Action recognizer    --action label-->        ingest_action_instance()

Each function below writes directly into the same graph schema that
etl_pipeline.py populates offline from a CSV, so the ontology + learned
statistics (stats_learning.py) and the inference layer (inference.py)
work identically whether the data arrived via batch ETL or a live stream.

None of these functions compute statistics themselves -- they only record
evidence (Observation / ActionInstance / ActivityInstance nodes). Call
stats_learning.learn_all(driver) periodically (e.g. nightly, or after every
N new instances) to refresh the learned AFFORDS / HAS_OBJECT / HAS_ACTIVITY
/ NEAR / TRANSITIONS_TO statistics from the newly accumulated evidence.
"""

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from geometry import geometry_from_coords


def ensure_human(tx, human_id):
    tx.run("MERGE (h:Human {human_id: $id})", id=human_id)


def ingest_pose_observation(driver, video_id, frame, timestamp, human_id, scene_id,
                             hand_xyz, nearby_objects):
    """
    Called once per processed frame by the pose-estimation stage.

    nearby_objects: list of {"object_id": ..., "xyz": (x,y,z)} for every
    object the detector found near the hand in this frame. Writes one
    Observation node with a NEAR_OBJECT edge (distance) to each.
    """
    observation_id = f"{video_id}_f{frame}_{human_id}"
    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(ensure_human, human_id)
        session.execute_write(
            lambda tx: tx.run(
                """
                MERGE (ob:Observation {observation_id: $obs_id})
                SET ob.video_id = $video_id, ob.frame = $frame, ob.timestamp = $timestamp,
                    ob.hand_x = $hx, ob.hand_y = $hy, ob.hand_z = $hz
                WITH ob
                MATCH (h:Human {human_id: $human_id})
                MERGE (ob)-[:OF_HUMAN]->(h)
                WITH ob
                MATCH (s:Scene {scene_id: $scene_id})
                MERGE (ob)-[:IN_SCENE]->(s)
                """,
                obs_id=observation_id, video_id=video_id, frame=frame, timestamp=timestamp,
                hx=hand_xyz[0], hy=hand_xyz[1], hz=hand_xyz[2],
                human_id=human_id, scene_id=scene_id,
            )
        )
        for obj in nearby_objects:
            geom = geometry_from_coords(hand_xyz, obj["xyz"])
            session.execute_write(
                lambda tx, obj=obj, geom=geom: tx.run(
                    """
                    MATCH (ob:Observation {observation_id: $obs_id})
                    MATCH (o:Object {object_id: $object_id})
                    MERGE (ob)-[r:NEAR_OBJECT]->(o)
                    SET r.distance = $d_ho
                    """,
                    obs_id=observation_id, object_id=obj["object_id"], d_ho=geom["d_Ho"],
                )
            )
    return observation_id


def ingest_action_instance(driver, instance_id, video_id, human_id, scene_id,
                            action_id, object_id, start_frame, end_frame,
                            hand_xyz, object_xyz,
                            activity_instance_id=None, activity_id=None, step_order=None,
                            prev_action_instance_id=None):
    """
    Called once per detected action segment by the action-recognition
    stage (e.g. "person H1 performed 'reaching' on 'bottle', frames
    120-145"). Computes geometry from the given coordinates and writes an
    ActionInstance node linked into the graph exactly like the batch ETL
    does, plus (optionally) attaches it to an ActivityInstance and chains
    it after a previous ActionInstance for temporal-context learning.
    """
    geom = geometry_from_coords(hand_xyz, object_xyz)

    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(ensure_human, human_id)
        session.execute_write(
            lambda tx: tx.run(
                """
                MERGE (ai:ActionInstance {instance_id: $iid})
                SET ai.video_id = $video_id, ai.start_frame = $start_frame,
                    ai.end_frame = $end_frame, ai.d_Ho = $d_ho, ai.theta = $theta, ai.e = $e
                WITH ai
                MATCH (a:ActionType {action_id: $action_id})
                MERGE (ai)-[:INSTANCE_OF]->(a)
                WITH ai
                MATCH (o:Object {object_id: $object_id})
                MERGE (ai)-[:ON_OBJECT]->(o)
                WITH ai
                MATCH (h:Human {human_id: $human_id})
                MERGE (ai)-[:PERFORMED_BY]->(h)
                WITH ai
                MATCH (s:Scene {scene_id: $scene_id})
                MERGE (ai)-[:OCCURRED_IN]->(s)
                """,
                iid=instance_id, video_id=video_id, start_frame=start_frame, end_frame=end_frame,
                d_ho=geom["d_Ho"], theta=geom["theta"], e=geom["e"],
                action_id=action_id, object_id=object_id, human_id=human_id, scene_id=scene_id,
            )
        )

        if activity_instance_id and activity_id:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (aci:ActivityInstance {instance_id: $aciid})
                    ON CREATE SET aci.video_id = $video_id
                    WITH aci
                    MATCH (ac:Activity {activity_id: $activity_id})
                    MERGE (aci)-[:INSTANCE_OF]->(ac)
                    WITH aci
                    MATCH (h:Human {human_id: $human_id})
                    MERGE (aci)-[:PERFORMED_BY]->(h)
                    WITH aci
                    MATCH (s:Scene {scene_id: $scene_id})
                    MERGE (aci)-[:OCCURRED_IN]->(s)
                    WITH aci
                    MATCH (ai:ActionInstance {instance_id: $iid})
                    MERGE (aci)-[:HAS_STEP {step_order: $step_order}]->(ai)
                    """,
                    aciid=activity_instance_id, video_id=video_id, activity_id=activity_id,
                    human_id=human_id, scene_id=scene_id, iid=instance_id,
                    step_order=step_order if step_order is not None else 0,
                )
            )

        if prev_action_instance_id:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (a1:ActionInstance {instance_id: $prev})
                    MATCH (a2:ActionInstance {instance_id: $curr})
                    MERGE (a1)-[:NEXT]->(a2)
                    """,
                    prev=prev_action_instance_id, curr=instance_id,
                )
            )

    return instance_id


if __name__ == "__main__":
    # Minimal smoke test of the live-ingestion API shape (not run against
    # a real Aura instance here -- see README for how to point this at yours).
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        obs_id = ingest_pose_observation(
            driver, video_id="LIVE001", frame=1, timestamp=0.033,
            human_id="H_live", scene_id="SC2",
            hand_xyz=(80, 10, 60),
            nearby_objects=[{"object_id": "o_bottle", "xyz": (60, 5, 65)}],
        )
        print("Wrote Observation:", obs_id)

        ingest_action_instance(
            driver, instance_id="LIVE_AI_0001", video_id="LIVE001",
            human_id="H_live", scene_id="SC2",
            action_id="a2", object_id="o_bottle",
            start_frame=0, end_frame=11,
            hand_xyz=(80, 10, 60), object_xyz=(60, 5, 65),
            activity_instance_id="LIVE_ACI_0001", activity_id="AC1", step_order=0,
        )
        print("Wrote live ActionInstance.")
    finally:
        driver.close()