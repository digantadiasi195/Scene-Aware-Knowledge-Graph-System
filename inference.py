# inference.py
"""
Inference / reasoning layer.

Implements the extended formulation your supervisor asked for:

    Original paper:   P(Action | Object, Geometry)
    Extended (this):  P(Action | Scene, Object, Human, Geometry, Temporal Context)

We factor it (naive-Bayes-style conditional independence, same spirit as
the paper's own factorization of Eq. 10 into distance/angular/edge terms)
as:

    P(a | sc, o, h, g, t)
        ~ P(a | o, sc, g)          geometry_score(...)            [paper's Eq.10, scene-scoped stats]
        x P(o | sc)                scene_object_prior              [NEW: Scene-HAS_OBJECT]
        x P(a | a_prev, sc)        temporal_prior                  [NEW: ActionType-TRANSITIONS_TO]
        x P(a | h)                 human_prior                     [NEW: Human action history]

Each factor defaults to a neutral weight (1.0) when data is missing, so
the ranking degrades gracefully to the original paper's geometry-only
behaviour rather than zeroing out candidates for which we simply have no
extra context yet.
"""

import math
from neo4j import GraphDatabase
from graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from graph.geometry import geometry_score


# --------------------------------------------------------------------- #
# Read helpers
# --------------------------------------------------------------------- #

def get_scene_candidate_objects(tx, scene_id, min_p=0.0):
    result = tx.run(
        """
        MATCH (s:Scene {scene_id: $scene_id})-[r:HAS_OBJECT]->(o:Object)
        WHERE r.p >= $min_p
        RETURN o.object_id AS object_id, o.name AS name, r.p AS p
        ORDER BY r.p DESC
        """,
        scene_id=scene_id, min_p=min_p,
    )
    return [dict(rec) for rec in result]


def get_scene_candidate_activities(tx, scene_id, min_p=0.0):
    result = tx.run(
        """
        MATCH (s:Scene {scene_id: $scene_id})-[r:HAS_ACTIVITY]->(ac:Activity)
        WHERE r.p >= $min_p
        RETURN ac.activity_id AS activity_id, ac.name AS name, r.p AS p
        ORDER BY r.p DESC
        """,
        scene_id=scene_id, min_p=min_p,
    )
    return [dict(rec) for rec in result]


def get_affords(tx, object_id, scene_id):
    """Scene-scoped AFFORDS row, falling back to the pooled GLOBAL row."""
    result = tx.run(
        """
        MATCH (o:Object {object_id: $object_id})-[r:AFFORDS]->(a:ActionType)
        WHERE r.scene_id = $scene_id
        RETURN a.action_id AS action_id, a.name AS action_name, r AS r
        """,
        object_id=object_id, scene_id=scene_id,
    )
    rows = [dict(action_id=rec["action_id"], action_name=rec["action_name"], **dict(rec["r"]))
            for rec in result]
    if rows:
        return rows
    
    # Fallback 1: Look for "GLOBAL" pooled statistics
    result = tx.run(
        """
        MATCH (o:Object {object_id: $object_id})-[r:AFFORDS]->(a:ActionType)
        WHERE r.scene_id = "GLOBAL"
        RETURN a.action_id AS action_id, a.name AS action_name, r AS r
        """,
        object_id=object_id,
    )
    rows = [dict(action_id=rec["action_id"], action_name=rec["action_name"], **dict(rec["r"]))
            for rec in result]
    if rows:
        return rows

    # Fallback 2: Ultimate fallback for completely unseen/novel objects
    # Returns neutral parameters so the system doesn't crash on novel objects.
    return [{
        "action_id": "unknown",
        "action_name": "unknown",
        "mu_dHo": 0.5, "sigma2_dHo": 0.1,
        "mu_theta": 0.0, "sigma2_theta": 0.1,
        "mu_e": 1.0, "sigma2_e": 0.1,
        "n_samples": 0
    }]


def get_transition_prior(tx, prev_action_id, action_id, scene_id):
    if prev_action_id is None:
        return 1.0
    result = tx.run(
        """
        MATCH (a1:ActionType {action_id: $prev})-[r:TRANSITIONS_TO {scene_id: $scene_id}]->
              (a2:ActionType {action_id: $curr})
        RETURN r.p AS p
        """,
        prev=prev_action_id, curr=action_id, scene_id=scene_id,
    ).single()
    return result["p"] if result else 1.0  # neutral if unseen


def get_human_action_prior(tx, human_id, action_id):
    """
    Empirical P(action | human), from that human's own ActionInstance
    history -- lets the graph learn per-person habits/patterns.
    """
    if human_id is None:
        return 1.0
    result = tx.run(
        """
        MATCH (h:Human {human_id: $human_id})<-[:PERFORMED_BY]-(ai:ActionInstance)
        WITH count(ai) AS total
        MATCH (h2:Human {human_id: $human_id})<-[:PERFORMED_BY]-
              (ai2:ActionInstance)-[:INSTANCE_OF]->(a:ActionType {action_id: $action_id})
        RETURN toFloat(count(ai2)) / total AS p, total AS total
        """,
        human_id=human_id, action_id=action_id,
    ).single()
    if not result or result["total"] < 5:
        return 1.0  # not enough personal history yet -- stay neutral
    return result["p"] if result["p"] else 1e-3


def get_activities_for_object_action_pair(tx, object_id, action_id):
    result = tx.run(
        """
        MATCH (ac:Activity)-[r:STEP {object_id: $object_id}]->(a:ActionType {action_id: $action_id})
        RETURN DISTINCT ac.activity_id AS activity_id, ac.name AS name
        """,
        object_id=object_id, action_id=action_id,
    )
    return [dict(rec) for rec in result]


# --------------------------------------------------------------------- #
# Extended scoring: P(Action | Scene, Object, Human, Geometry, Temporal)
# --------------------------------------------------------------------- #

def score_candidate(session, scene_id, object_id, d_ho, theta, e, affords_row,
                    scene_object_p, human_id=None, prev_action_id=None):
    """
    Calculates the mathematically stable log-probability score for a candidate action.
    """
    # 1. Geometry Score (Paper's Eq. 10)
    g_score = geometry_score(d_ho, theta, e, affords_row)
    
    # 2. Fetch priors (Ensure no zero probabilities to prevent log(0) errors)
    temporal_p = max(session.execute_read(
        get_transition_prior, prev_action_id, affords_row["action_id"], scene_id
    ), 1e-6)
    
    human_p = max(session.execute_read(
        get_human_action_prior, human_id, affords_row["action_id"]
    ), 1e-6)

    # 3. MATH FIX: Use log-probabilities for stable, mathematically sound ranking
    # log(A * B * C * D) = log(A) + log(B) + log(C) + log(D)
    log_combined = (
        math.log(max(g_score, 1e-9)) + 
        math.log(max(scene_object_p, 1e-6)) + 
        math.log(temporal_p) + 
        math.log(human_p)
    )

    return {
        "action_id": affords_row["action_id"],
        "action_name": affords_row["action_name"],
        "object_id": object_id,
        "geometry_score": g_score,
        "scene_object_prior": scene_object_p,
        "temporal_prior": temporal_p,
        "human_prior": human_p,
        "combined_log_score": log_combined,
    }


def predict_action(driver, scene_id, observed_objects, human_id=None, prev_action_id=None):
    """
    observed_objects: list of dicts, one per object currently detected near
    the hand: {"object_id": ..., "d_Ho": ..., "theta": ..., "e": ...}
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        # OPTIMIZATION: Fetch scene objects and their priors ONCE per frame
        scene_objs = session.execute_read(get_scene_candidate_objects, scene_id)
        scene_objects = {o["object_id"] for o in scene_objs}
        scene_obj_p_map = {o["object_id"]: o["p"] for o in scene_objs}

        # Gate objects: only process objects that belong to this scene (fallback to all if none match)
        gated = [o for o in observed_objects if o["object_id"] in scene_objects] or observed_objects

        scored = []
        for obs in gated:
            affords_rows = session.execute_read(get_affords, obs["object_id"], scene_id)
            
            # Get scene object prior once per object (fallback to 1e-6 if unseen in scene)
            s_obj_p = scene_obj_p_map.get(obs["object_id"], 1e-6)
            
            for row in affords_rows:
                scored.append(score_candidate(
                    session, scene_id, obs["object_id"],
                    obs["d_Ho"], obs["theta"], obs["e"], row,
                    scene_object_p=s_obj_p,
                    human_id=human_id, 
                    prev_action_id=prev_action_id,
                ))
        
        # Sort by the mathematically stable log-probability score
        scored.sort(key=lambda x: x["combined_log_score"], reverse=True)
        
        return scored


def predict_activity(driver, scene_id, action_id, object_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        scene_activity_ids = {a["activity_id"] for a in
                               session.execute_read(get_scene_candidate_activities, scene_id)}
        candidates = session.execute_read(get_activities_for_object_action_pair, object_id, action_id)
        filtered = [a for a in candidates if a["activity_id"] in scene_activity_ids]
        return filtered or candidates


# --------------------------------------------------------------------- #
# CLI Testing / Smoke Test
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # Note: d_Ho is in METERS (e.g., 1.2m), not centimeters.
        observed = [
            {"object_id": "o_table", "d_Ho": 1.2, "theta": 0.15, "e": 0.35},
        ]
        
        ranked = predict_action(driver, scene_id="SC1", observed_objects=observed,
                                human_id="H1", prev_action_id="a1")
        
        print("Ranked candidate (action, object) pairs:")
        for r in ranked[:5]:
            # FIX: Use 'combined_log_score' instead of the old 'combined_score'
            print(f"  {r['action_name']:>10} on {r['object_id']:<10}  "
                  f"log_score={r['combined_log_score']:.4f}  "
                  f"(geom={r['geometry_score']:.4f}, scene_obj={r['scene_object_prior']:.3f},  "
                  f"temporal={r['temporal_prior']:.3f}, human={r['human_prior']:.3f})")
        
        if ranked:
            top = ranked[0]
            acts = predict_activity(driver, "SC1", top["action_id"], top["object_id"])
            print("\nConsistent ongoing activities:")
            for a in acts:
                print(" ", a["name"])
    finally:
        driver.close()