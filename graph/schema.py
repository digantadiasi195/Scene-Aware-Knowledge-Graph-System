# schema.py
"""
Schema for the Scene-Aware Human-Centric Probabilistic Knowledge Graph
(v2). This replaces the flat v1 design with two clearly separated layers:

ONTOLOGY LAYER  (general, hand-authored knowledge -- "what is possible")
    Scene, SceneCondition, Object, ActionType, Activity, State
    -- describes the world: which scenes/objects/actions/activities exist
       and which nominal action-sequences build each activity (Fig. 9 of
       the paper). Contains no statistics.

INSTANCE / OBSERVATION LAYER  (specific, data-driven -- "what was seen")
    Human, Observation, ActionInstance, ActivityInstance
    -- describes concrete video evidence: a specific person, at a specific
       frame, performing a specific action on a specific object in a
       specific scene. This is what a computer-vision pipeline
       (detector + pose model + action recognizer) writes into the graph.

LEARNED / STATISTICAL LAYER  (derived, computed by the ETL / stats_learning
    aggregation from the instance layer -- never hand-written)
    Scene-[:HAS_OBJECT {p}]->Object
    Scene-[:HAS_ACTIVITY {p}]->Activity
    Object-[:NEAR {scene_id, p}]->Object
    Object-[:AFFORDS {scene_id, mu_*, sigma2_*}]->ActionType
    ActionType-[:TRANSITIONS_TO {scene_id, p}]->ActionType

Node labels
-----------
Scene            : physical environment (office, kitchen, ...)
SceneCondition   : lighting / hazard / workspace-type metadata for a scene
Object           : a manipulable object type
ActionType       : general action category ("reaching", "pouring", ...)
Activity         : a full activity type ("drinking_water", ...)
State            : nominal state node used by the Activity's action graph

Human            : one recorded/observed person
Observation      : one frame-level (or short-window) sensor reading
ActionInstance   : one concrete observed occurrence of an ActionType
ActivityInstance : one concrete observed occurrence of an Activity
"""

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

CONSTRAINTS = [
    # Ontology layer
    "CREATE CONSTRAINT scene_id      IF NOT EXISTS FOR (s:Scene)            REQUIRE s.scene_id IS UNIQUE",
    "CREATE CONSTRAINT cond_id       IF NOT EXISTS FOR (c:SceneCondition)   REQUIRE c.condition_id IS UNIQUE",
    "CREATE CONSTRAINT object_id     IF NOT EXISTS FOR (o:Object)           REQUIRE o.object_id IS UNIQUE",
    "CREATE CONSTRAINT action_id     IF NOT EXISTS FOR (a:ActionType)       REQUIRE a.action_id IS UNIQUE",
    "CREATE CONSTRAINT activity_id   IF NOT EXISTS FOR (ac:Activity)        REQUIRE ac.activity_id IS UNIQUE",
    "CREATE CONSTRAINT state_uid     IF NOT EXISTS FOR (st:State)           REQUIRE st.uid IS UNIQUE",
    # Instance / observation layer
    "CREATE CONSTRAINT human_id      IF NOT EXISTS FOR (h:Human)            REQUIRE h.human_id IS UNIQUE",
    "CREATE CONSTRAINT obs_id        IF NOT EXISTS FOR (ob:Observation)     REQUIRE ob.observation_id IS UNIQUE",
    "CREATE CONSTRAINT action_inst_id IF NOT EXISTS FOR (ai:ActionInstance) REQUIRE ai.instance_id IS UNIQUE",
    "CREATE CONSTRAINT activity_inst_id IF NOT EXISTS FOR (aci:ActivityInstance) REQUIRE aci.instance_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX object_name_idx    IF NOT EXISTS FOR (o:Object)          ON (o.name)",
    "CREATE INDEX action_name_idx    IF NOT EXISTS FOR (a:ActionType)      ON (a.name)",
    "CREATE INDEX activity_name_idx  IF NOT EXISTS FOR (ac:Activity)       ON (ac.name)",
    "CREATE INDEX action_inst_video  IF NOT EXISTS FOR (ai:ActionInstance) ON (ai.video_id)",
    "CREATE INDEX action_inst_frame  IF NOT EXISTS FOR (ai:ActionInstance) ON (ai.start_frame)",
    "CREATE INDEX obs_video_frame    IF NOT EXISTS FOR (ob:Observation)    ON (ob.video_id, ob.frame)",
]


def apply_schema(driver):
    with driver.session(database=NEO4J_DATABASE) as session:
        for stmt in CONSTRAINTS + INDEXES:
            session.run(stmt)


def wipe_database(driver):
    """Danger: deletes every node/relationship. Useful while iterating."""
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    apply_schema(driver)
    driver.close()
    print("Schema (constraints + indexes) applied.")