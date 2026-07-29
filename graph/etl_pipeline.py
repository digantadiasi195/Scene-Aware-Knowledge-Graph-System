# elt_pipeline.py
"""
ETL pipeline:
  Stage A -- load the ontology layer (knowledge_seed.py: Scene,
             SceneCondition, Object, ActionType, Activity, nominal
             State-graphs). Pure domain knowledge, no stats.
  Stage B -- load the instance/observation layer from a features dataframe
             produced by feature_engineering.py: Human, Observation,
             ActionInstance, ActivityInstance nodes + their relationships,
             including the NEXT chain (real observed temporal order) that
             stats_learning.py later mines for transition probabilities.

Run:
    python etl_pipeline.py                      # uses raw_observations.csv
    python etl_pipeline.py my_real_data.csv      # your real extracted data
    python etl_pipeline.py --wipe                # wipe DB first
"""

import sys
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from schema import apply_schema, wipe_database
from knowledge_seed import SCENES, SCENE_CONDITIONS, OBJECTS, ACTIONS, ACTIVITIES
from feature_engineering import run_feature_engineering


# --------------------------------------------------------------------- #
# Stage A: ontology layer
# --------------------------------------------------------------------- #

def load_scenes(tx):
    tx.run(
        "UNWIND $rows AS row MERGE (s:Scene {scene_id: row.scene_id}) SET s.name = row.name",
        rows=SCENES,
    )


def load_scene_conditions(tx):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (c:SceneCondition {condition_id: row.condition_id})
        SET c.lighting = row.lighting, c.hazard_level = row.hazard_level,
            c.workspace_type = row.workspace_type
        WITH c, row
        MATCH (s:Scene {scene_id: row.scene_id})
        MERGE (s)-[:HAS_CONDITION]->(c)
        """,
        rows=SCENE_CONDITIONS,
    )


def load_objects(tx):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (o:Object {object_id: row.object_id})
        SET o.name = row.name, o.category = row.category
        """,
        rows=OBJECTS,
    )


def load_actions(tx):
    tx.run(
        "UNWIND $rows AS row MERGE (a:ActionType {action_id: row.action_id}) SET a.name = row.name",
        rows=ACTIONS,
    )


def load_activities_and_states(tx):
    """Nominal Activity nodes + Fig.9-style State graph + Activity-STEP index."""
    for act in ACTIVITIES:
        tx.run(
            "MERGE (ac:Activity {activity_id: $id}) SET ac.name = $name",
            id=act["activity_id"], name=act["name"],
        )
        for scene_id in act["possible_scenes"]:
            tx.run(
                """
                MATCH (s:Scene {scene_id: $scene_id})
                MATCH (ac:Activity {activity_id: $activity_id})
                MERGE (s)-[:HAS_ACTIVITY]->(ac)
                """,
                scene_id=scene_id, activity_id=act["activity_id"],
            )

        for seq_idx, sequence in enumerate(act["sequences"]):
            prev_uid = f"{act['activity_id']}_seq{seq_idx}_s0"
            tx.run(
                """
                MERGE (st:State {uid: $uid}) SET st.activity_id = $aid, st.name = 'start'
                WITH st MATCH (ac:Activity {activity_id: $aid}) MERGE (ac)-[:STARTS_WITH]->(st)
                """,
                uid=prev_uid, aid=act["activity_id"],
            )
            for step_order, (action_id, object_id) in enumerate(sequence):
                next_uid = f"{act['activity_id']}_seq{seq_idx}_s{step_order + 1}"
                is_last = step_order == len(sequence) - 1
                tx.run(
                    """
                    MERGE (st2:State {uid: $next_uid})
                    SET st2.activity_id = $aid, st2.name = $name
                    WITH st2
                    MATCH (st1:State {uid: $prev_uid})
                    MERGE (st1)-[r:TRANSITION {step_order: $step_order}]->(st2)
                    SET r.action_id = $action_id, r.object_id = $object_id
                    """,
                    next_uid=next_uid, aid=act["activity_id"],
                    name=("end" if is_last else f"mid{step_order + 1}"),
                    prev_uid=prev_uid, step_order=step_order,
                    action_id=action_id, object_id=object_id,
                )
                if is_last:
                    tx.run(
                        """
                        MATCH (ac:Activity {activity_id: $aid})
                        MATCH (st:State {uid: $uid})
                        MERGE (ac)-[:ENDS_WITH]->(st)
                        """,
                        aid=act["activity_id"], uid=next_uid,
                    )
                tx.run(
                    """
                    MATCH (ac:Activity {activity_id: $aid})
                    MATCH (a:ActionType {action_id: $action_id})
                    MERGE (ac)-[r:STEP {step_order: $step_order, seq_idx: $seq_idx}]->(a)
                    SET r.object_id = $object_id
                    """,
                    aid=act["activity_id"], action_id=action_id,
                    step_order=step_order, seq_idx=seq_idx, object_id=object_id,
                )
                prev_uid = next_uid


def load_ontology(driver):
    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(load_scenes)
        session.execute_write(load_scene_conditions)
        session.execute_write(load_objects)
        session.execute_write(load_actions)
        session.execute_write(load_activities_and_states)


# --------------------------------------------------------------------- #
# Stage B: instance / observation layer
# --------------------------------------------------------------------- #

def load_humans(tx, human_ids):
    tx.run(
        "UNWIND $ids AS hid MERGE (h:Human {human_id: hid})",
        ids=list(human_ids),
    )


def load_observations(tx, obs_rows):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (ob:Observation {observation_id: row.observation_id})
        SET ob.video_id = row.video_id, ob.frame = row.frame,
            ob.timestamp = row.timestamp,
            ob.hand_x = row.hand_x, ob.hand_y = row.hand_y, ob.hand_z = row.hand_z
        WITH ob, row
        MATCH (h:Human {human_id: row.human_id})
        MERGE (ob)-[:OF_HUMAN]->(h)
        WITH ob, row
        MATCH (s:Scene {scene_id: row.scene_id})
        MERGE (ob)-[:IN_SCENE]->(s)
        WITH ob, row
        MATCH (o:Object {object_id: row.object_id})
        MERGE (ob)-[r:NEAR_OBJECT]->(o)
        SET r.distance = row.d_Ho
        """,
        rows=obs_rows,
    )


def load_action_instances(tx, action_instance_rows):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (ai:ActionInstance {instance_id: row.instance_id})
        SET ai.video_id = row.video_id, ai.start_frame = row.start_frame,
            ai.end_frame = row.end_frame, ai.d_Ho = row.d_Ho,
            ai.theta = row.theta, ai.e = row.e, ai.n_frames = row.n_frames
        WITH ai, row
        MATCH (a:ActionType {action_id: row.action_id})
        MERGE (ai)-[:INSTANCE_OF]->(a)
        WITH ai, row
        MATCH (o:Object {object_id: row.object_id})
        MERGE (ai)-[:ON_OBJECT]->(o)
        WITH ai, row
        MATCH (h:Human {human_id: row.human_id})
        MERGE (ai)-[:PERFORMED_BY]->(h)
        WITH ai, row
        MATCH (s:Scene {scene_id: row.scene_id})
        MERGE (ai)-[:OCCURRED_IN]->(s)
        """,
        rows=action_instance_rows,
    )


def load_activity_instances(tx, activity_instance_rows, action_instance_rows):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (aci:ActivityInstance {instance_id: row.instance_id})
        SET aci.video_id = row.video_id, aci.start_frame = row.start_frame,
            aci.end_frame = row.end_frame, aci.n_steps = row.n_steps
        WITH aci, row
        MATCH (ac:Activity {activity_id: row.activity_id})
        MERGE (aci)-[:INSTANCE_OF]->(ac)
        WITH aci, row
        MATCH (h:Human {human_id: row.human_id})
        MERGE (aci)-[:PERFORMED_BY]->(h)
        WITH aci, row
        MATCH (s:Scene {scene_id: row.scene_id})
        MERGE (aci)-[:OCCURRED_IN]->(s)
        """,
        rows=activity_instance_rows,
    )

    step_rows = action_instance_rows[["activity_instance_id", "instance_id", "step_order"]] \
        .rename(columns={"activity_instance_id": "activity_uid", "instance_id": "action_uid"})
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (aci:ActivityInstance {instance_id: row.activity_uid})
        MATCH (ai:ActionInstance {instance_id: row.action_uid})
        MERGE (aci)-[:HAS_STEP {step_order: row.step_order}]->(ai)
        """,
        rows=step_rows.to_dict("records"),
    )


def load_next_chains(tx, action_instance_rows):
    """
    Builds the real observed temporal chain of ActionInstances within each
    ActivityInstance, ordered by step_order. stats_learning.py mines this
    to compute empirical TRANSITIONS_TO probabilities between ActionTypes.
    """
    df = action_instance_rows.sort_values(["activity_instance_id", "step_order"])
    pairs = []
    for _, group in df.groupby("activity_instance_id"):
        ids = group["instance_id"].tolist()
        for a, b in zip(ids, ids[1:]):
            pairs.append({"from_id": a, "to_id": b})
    if pairs:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a1:ActionInstance {instance_id: row.from_id})
            MATCH (a2:ActionInstance {instance_id: row.to_id})
            MERGE (a1)-[:NEXT]->(a2)
            """,
            rows=pairs,
        )


def load_instances(driver, features):
    action_df = features["action_instances"]
    activity_df = features["activity_instances"]
    obs_df = features["observations"]

    with driver.session(database=NEO4J_DATABASE) as session:
        print("  loading Human nodes...")
        session.execute_write(load_humans, action_df["human_id"].unique())
        print("  loading Observation nodes...")
        session.execute_write(load_observations, obs_df.to_dict("records"))
        print("  loading ActionInstance nodes...")
        session.execute_write(load_action_instances, action_df.to_dict("records"))
        print("  loading ActivityInstance nodes + HAS_STEP...")
        session.execute_write(load_activity_instances, activity_df.to_dict("records"), action_df)
        print("  loading NEXT chains (observed temporal order)...")
        session.execute_write(load_next_chains, action_df)


def run_etl(csv_path, driver, wipe=False):
    if wipe:
        print("Wiping database...")
        wipe_database(driver)

    print("Applying schema...")
    apply_schema(driver)

    print("Loading ontology layer (Scene/Object/ActionType/Activity)...")
    load_ontology(driver)

    print(f"Running feature engineering on {csv_path}...")
    features = run_feature_engineering(csv_path)

    print("Loading instance/observation layer...")
    load_instances(driver, features)

    print("ETL complete.")
    return features


if __name__ == "__main__":
    wipe_flag = "--wipe" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_path = positional[0] if positional else "raw_observations.csv"

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        run_etl(csv_path, driver, wipe=wipe_flag)
    finally:
        driver.close()