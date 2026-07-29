# stats_learning.py
"""
Learns every statistical/probabilistic relationship in the graph FROM the
instance/observation layer that etl_pipeline.py wrote in. Nothing here is
hand-typed -- this is the automated, data-driven replacement for v1's
placeholder seed_data.py values.
# In stats_learning.py, update the comment for learn_affords:
"""
"""Learns Object -[:AFFORDS]-> ActionType geometry stats.
NOTE: This computes the mean and variance of ACTION INSTANCE MEANS 
(not raw frames). This acts as a hierarchical Bayesian approximation, 
preventing single long actions from dominating the variance calculation."""
"""

Learns:
  Object -[:AFFORDS {scene_id, mu_dHo, sigma2_dHo, mu_theta, sigma2_theta,
                      mu_e, sigma2_e, n_samples}]-> ActionType
      via Neo4j's own avg()/stDev() aggregation over ActionInstance
      geometry -- computed inside the database itself.

  Scene -[:HAS_OBJECT {p, n_samples}]-> Object
      P(object seen | scene), from ActionInstance/Observation counts.

  Scene -[:HAS_ACTIVITY {p, n_samples}]-> Activity
      P(activity | scene), from ActivityInstance counts.

  Object -[:NEAR {scene_id, p, n_samples}]-> Object
      spatial co-occurrence prior: how often two objects are both near
      the hand within the same scene (from Observation-[:NEAR_OBJECT]).

  ActionType -[:TRANSITIONS_TO {scene_id, p, n_samples}]-> ActionType
      empirical P(next action | current action, scene), mined from the
      observed ActionInstance-[:NEXT]->ActionInstance chains.

Run:
    python stats_learning.py
"""

from collections import defaultdict
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

MIN_SAMPLES_AFFORDS = 3
MIN_SAMPLES_TRANSITION = 1


# --------------------------------------------------------------------- #
# 1. Object -[:AFFORDS]-> ActionType   (geometry mu/sigma, computed in-DB)
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# 1. Object -[:AFFORDS]-> ActionType   (geometry mu/sigma, computed in-DB)
# --------------------------------------------------------------------- #

def learn_affords(tx):

    # Scene-specific AFFORDS statistics
    tx.run(
        """
        MATCH (ai:ActionInstance)-[:ON_OBJECT]->(o:Object)
        MATCH (ai)-[:INSTANCE_OF]->(a:ActionType)
        MATCH (ai)-[:OCCURRED_IN]->(s:Scene)

        WITH o, a, s,
             avg(ai.d_Ho) AS mu_dHo,
             stDev(ai.d_Ho) AS sd_dHo,
             avg(ai.theta) AS mu_theta,
             stDev(ai.theta) AS sd_theta,
             avg(ai.e) AS mu_e,
             stDev(ai.e) AS sd_e,
             count(ai) AS n

        WHERE n >= $min_n

        MERGE (o)-[r:AFFORDS {scene_id: s.scene_id}]->(a)

        SET r.mu_dHo = mu_dHo,
            r.sigma2_dHo = coalesce(sd_dHo * sd_dHo, 0.0),
            r.mu_theta = mu_theta,
            r.sigma2_theta = coalesce(sd_theta * sd_theta, 0.0),
            r.mu_e = mu_e,
            r.sigma2_e = coalesce(sd_e * sd_e, 0.0),
            r.n_samples = n
        """,
        min_n=MIN_SAMPLES_AFFORDS,
    )


    # Global fallback AFFORDS statistics
    # NOTE: Neo4j does not allow null relationship properties.
    # We use scene_id="GLOBAL" instead.

    tx.run(
        """
        MATCH (ai:ActionInstance)-[:ON_OBJECT]->(o:Object)
        MATCH (ai)-[:INSTANCE_OF]->(a:ActionType)

        WITH o, a,
             avg(ai.d_Ho) AS mu_dHo,
             stDev(ai.d_Ho) AS sd_dHo,
             avg(ai.theta) AS mu_theta,
             stDev(ai.theta) AS sd_theta,
             avg(ai.e) AS mu_e,
             stDev(ai.e) AS sd_e,
             count(ai) AS n

        WHERE n >= $min_n

        MERGE (o)-[r:AFFORDS {scene_id: "GLOBAL"}]->(a)

        SET r.mu_dHo = mu_dHo,
            r.sigma2_dHo = coalesce(sd_dHo * sd_dHo, 0.0),
            r.mu_theta = mu_theta,
            r.sigma2_theta = coalesce(sd_theta * sd_theta, 0.0),
            r.mu_e = mu_e,
            r.sigma2_e = coalesce(sd_e * sd_e, 0.0),
            r.n_samples = n
        """,
        min_n=MIN_SAMPLES_AFFORDS,
    )


# --------------------------------------------------------------------- #
# 2. Scene -[:HAS_OBJECT]-> Object      P(object | scene)
# --------------------------------------------------------------------- #

def learn_scene_object_prior(tx):
    totals = {r["scene_id"]: r["total"] for r in tx.run(
        """
        MATCH (s:Scene)<-[:OCCURRED_IN]-(ai:ActionInstance)
        RETURN s.scene_id AS scene_id, count(ai) AS total
        """
    )}
    counts = tx.run(
        """
        MATCH (s:Scene)<-[:OCCURRED_IN]-(ai:ActionInstance)-[:ON_OBJECT]->(o:Object)
        RETURN s.scene_id AS scene_id, o.object_id AS object_id, count(ai) AS cnt
        """
    )
    rows = []
    for r in counts:
        total = totals.get(r["scene_id"], 0)
        if total == 0:
            continue
        rows.append({
            "scene_id": r["scene_id"], "object_id": r["object_id"],
            "p": r["cnt"] / total, "n_samples": r["cnt"],
        })
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Scene {scene_id: row.scene_id})
        MATCH (o:Object {object_id: row.object_id})
        MERGE (s)-[r:HAS_OBJECT]->(o)
        SET r.p = row.p, r.n_samples = row.n_samples
        """,
        rows=rows,
    )


# --------------------------------------------------------------------- #
# 3. Scene -[:HAS_ACTIVITY]-> Activity   P(activity | scene)
# --------------------------------------------------------------------- #

def learn_scene_activity_prior(tx):
    totals = {r["scene_id"]: r["total"] for r in tx.run(
        """
        MATCH (s:Scene)<-[:OCCURRED_IN]-(aci:ActivityInstance)
        RETURN s.scene_id AS scene_id, count(aci) AS total
        """
    )}
    counts = tx.run(
        """
        MATCH (s:Scene)<-[:OCCURRED_IN]-(aci:ActivityInstance)-[:INSTANCE_OF]->(ac:Activity)
        RETURN s.scene_id AS scene_id, ac.activity_id AS activity_id, count(aci) AS cnt
        """
    )
    rows = []
    for r in counts:
        total = totals.get(r["scene_id"], 0)
        if total == 0:
            continue
        rows.append({
            "scene_id": r["scene_id"], "activity_id": r["activity_id"],
            "p": r["cnt"] / total, "n_samples": r["cnt"],
        })
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Scene {scene_id: row.scene_id})
        MATCH (ac:Activity {activity_id: row.activity_id})
        MERGE (s)-[r:HAS_ACTIVITY]->(ac)
        SET r.p = row.p, r.n_samples = row.n_samples
        """,
        rows=rows,
    )


# --------------------------------------------------------------------- #
# 4. Object -[:NEAR]-> Object   spatial co-occurrence prior per scene
# --------------------------------------------------------------------- #

def learn_spatial_near(tx):
    scene_obs_totals = {r["scene_id"]: r["total"] for r in tx.run(
        """
        MATCH (s:Scene)<-[:IN_SCENE]-(ob:Observation)
        RETURN s.scene_id AS scene_id, count(DISTINCT ob) AS total
        """
    )}
    pair_counts = tx.run(
        """
        MATCH (s:Scene)<-[:IN_SCENE]-(ob:Observation)-[:NEAR_OBJECT]->(o1:Object)
        MATCH (ob)-[:NEAR_OBJECT]->(o2:Object)
        WHERE o1.object_id < o2.object_id
        RETURN s.scene_id AS scene_id, o1.object_id AS o1, o2.object_id AS o2,
               count(DISTINCT ob) AS cnt
        """
    )
    rows = []
    for r in pair_counts:
        total = scene_obs_totals.get(r["scene_id"], 0)
        if total == 0:
            continue
        rows.append({
            "scene_id": r["scene_id"], "o1": r["o1"], "o2": r["o2"],
            "p": r["cnt"] / total, "n_samples": r["cnt"],
        })
    if rows:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:Object {object_id: row.o1})
            MATCH (b:Object {object_id: row.o2})
            MERGE (a)-[r:NEAR {scene_id: row.scene_id}]->(b)
            SET r.p = row.p, r.n_samples = row.n_samples
            """,
            rows=rows,
        )


# --------------------------------------------------------------------- #
# 5. ActionType -[:TRANSITIONS_TO]-> ActionType   temporal context
# --------------------------------------------------------------------- #

def learn_transitions(tx):
    totals = defaultdict(int)
    for r in tx.run(
        """
        MATCH (ai1:ActionInstance)-[:NEXT]->(:ActionInstance)
        MATCH (ai1)-[:INSTANCE_OF]->(a1:ActionType)
        MATCH (ai1)-[:OCCURRED_IN]->(s:Scene)
        RETURN a1.action_id AS action_id, s.scene_id AS scene_id, count(*) AS cnt
        """
    ):
        totals[(r["action_id"], r["scene_id"])] += r["cnt"]

    pair_counts = tx.run(
        """
        MATCH (ai1:ActionInstance)-[:NEXT]->(ai2:ActionInstance)
        MATCH (ai1)-[:INSTANCE_OF]->(a1:ActionType)
        MATCH (ai2)-[:INSTANCE_OF]->(a2:ActionType)
        MATCH (ai1)-[:OCCURRED_IN]->(s:Scene)
        RETURN a1.action_id AS from_id, a2.action_id AS to_id,
               s.scene_id AS scene_id, count(*) AS cnt
        """
    )
    rows = []
    for r in pair_counts:
        total = totals.get((r["from_id"], r["scene_id"]), 0)
        if total < MIN_SAMPLES_TRANSITION:
            continue
        rows.append({
            "from_id": r["from_id"], "to_id": r["to_id"], "scene_id": r["scene_id"],
            "p": r["cnt"] / total, "n_samples": r["cnt"],
        })
    if rows:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a1:ActionType {action_id: row.from_id})
            MATCH (a2:ActionType {action_id: row.to_id})
            MERGE (a1)-[r:TRANSITIONS_TO {scene_id: row.scene_id}]->(a2)
            SET r.p = row.p, r.n_samples = row.n_samples
            """,
            rows=rows,
        )


# --------------------------------------------------------------------- #

def learn_all(driver):
    with driver.session(database=NEO4J_DATABASE) as session:
        print("Learning Object-AFFORDS->ActionType geometry stats...")
        session.execute_write(learn_affords)
        print("Learning Scene-HAS_OBJECT->Object priors...")
        session.execute_write(learn_scene_object_prior)
        print("Learning Scene-HAS_ACTIVITY->Activity priors...")
        session.execute_write(learn_scene_activity_prior)
        print("Learning Object-NEAR->Object spatial priors...")
        session.execute_write(learn_spatial_near)
        print("Learning ActionType-TRANSITIONS_TO->ActionType temporal priors...")
        session.execute_write(learn_transitions)
    print("Stats learning complete.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        learn_all(driver)
    finally:
        driver.close()