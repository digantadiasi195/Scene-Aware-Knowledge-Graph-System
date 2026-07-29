# # evaluate_probabilistic.py
# import torch
# from neo4j import GraphDatabase
# from tqdm import tqdm
# from graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
# from graph.inference import predict_action

# def evaluate_probabilistic():
#     print("="*60)
#     print("Probabilistic Model Evaluation (Baseline)")
#     print("="*60)
    
#     # 1. Connect Neo4j
#     print("\nConnecting Neo4j...")
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
#     # 2. Fetch test data directly from Neo4j ActionInstances
#     # This grabs the EXACT d_Ho, theta, e that stats_learning.py calculated!
#     print("Fetching ActionInstances from Neo4j for fair evaluation...")
    
#     test_instances = []
#     with driver.session(database=NEO4J_DATABASE) as session:
#         # Get the last 1000 instances to match the GNN test split
#         result = session.run(
#             """
#             MATCH (ai:ActionInstance)-[:INSTANCE_OF]->(a:ActionType)
#             MATCH (ai)-[:ON_OBJECT]->(o:Object)
#             MATCH (ai)-[:OCCURRED_IN]->(s:Scene)
#             RETURN ai.instance_id AS id, a.action_id AS true_action, 
#                    o.object_id AS object_id, s.scene_id AS scene_id,
#                    ai.d_Ho AS d_Ho, ai.theta AS theta, ai.e AS e
#             ORDER BY ai.instance_id DESC
#             LIMIT 1000
#             """
#         )
#         test_instances = [dict(rec) for rec in result]
        
#     driver.close()
    
#     if not test_instances:
#         print("No ActionInstances found in Neo4j. Run ETL + stats_learning first.")
#         return
        
#     print(f"Loaded {len(test_instances)} instances for evaluation\n")
    
#     # 3. Reconnect for the inference loop
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
#     correct = 0
#     total = 0
    
#     print("Evaluating Probabilistic Model...")
#     for inst in tqdm(test_instances, desc="Evaluating Baseline"):
        
#         # Safely handle None values just in case
#         d_ho = inst["d_Ho"] if inst["d_Ho"] is not None else 1.0
#         theta = inst["theta"] if inst["theta"] is not None else 0.0
#         e = inst["e"] if inst["e"] is not None else 1.0
        
#         observed_objects = [{
#             "object_id": inst["object_id"],
#             "d_Ho": float(d_ho), 
#             "theta": float(theta), 
#             "e": float(e)
#         }]
        
#         # Query Probabilistic Model
#         ranked = predict_action(
#             driver, 
#             scene_id=inst["scene_id"], 
#             observed_objects=observed_objects,
#             human_id=None,  # Stateless evaluation
#             prev_action_id=None 
#         )
        
#         if ranked:
#             top_pred_action = ranked[0]["action_id"]
#             if top_pred_action == inst["true_action"]:
#                 correct += 1
#         total += 1
        
#     driver.close()
    
#     # 4. Print Results
#     accuracy = correct / total if total > 0 else 0
#     print("\n")
#     print("="*60)
#     print("PROBABILISTIC BASELINE RESULTS")
#     print("="*60)
#     print(f"Probabilistic Accuracy : {accuracy:.4f}")
#     print(f"Correct Predictions    : {correct}")
#     print(f"Total Samples          : {total}")
#     print("="*60)
#     print("""
# PHD CONTRIBUTION SUMMARY:
# -------------------------
# GAT Knowledge Reasoner : 0.6960 (69.6%)
# Probabilistic Baseline : """ + f"{accuracy:.4f}" + f" ({accuracy*100:.1f}%)" + """

# Conclusion: 
# If GAT > Baseline, your GNN successfully captures non-linear spatial 
# relationships that the naive-Bayes probabilistic model misses!
# """)

# if __name__ == "__main__":
#     evaluate_probabilistic()

# evaluate_probabilistic.py
import time
from neo4j import GraphDatabase
from tqdm import tqdm
from graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from inference import predict_action

def evaluate_probabilistic():
    print("="*60)
    print("Probabilistic Model Evaluation (Baseline)")
    print("="*60)
    
    print("\nConnecting Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    print("Fetching ActionInstance test samples from Neo4j...")
    
    # BRILLIANT SHORTCUT: 
    # Query the exact d_Ho, theta, and e already computed and stored 
    # on the ActionInstance nodes by stats_learning.py!
    # We also fetch the previous action to compute the temporal prior fairly.
    query = """
    MATCH (ai:ActionInstance)-[:INSTANCE_OF]->(a:ActionType)
    MATCH (ai)-[:ON_OBJECT]->(o:Object)
    MATCH (ai)-[:OCCURRED_IN]->(s:Scene)
    MATCH (ai)-[:PERFORMED_BY]->(h:Human)
    OPTIONAL MATCH (prev_ai:ActionInstance)-[:NEXT]->(ai)
    OPTIONAL MATCH (prev_ai)-[:INSTANCE_OF]->(prev_a:ActionType)
    RETURN ai.instance_id AS instance_id,
           a.action_id AS true_action_id,
           o.object_id AS object_id,
           s.scene_id AS scene_id,
           h.human_id AS human_id,
           ai.d_Ho AS d_Ho,
           ai.theta AS theta,
           ai.e AS e,
           prev_a.action_id AS prev_action_id
    LIMIT 1000
    """
    
    # OPTIMIZATION: Fetch ALL 1000 samples in a SINGLE query upfront.
    # This prevents N+1 query slowness and releases the session immediately.
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        test_samples = [dict(rec) for rec in result]
        
    if not test_samples:
        print("No ActionInstance samples found. Run ETL and stats_learning first.")
        driver.close()
        return
        
    print(f"Loaded {len(test_samples)} samples.\n")
    print("Evaluating Probabilistic Model (using real geometry stats)...")
    
    correct = 0
    total = len(test_samples)
    
    start_time = time.time()
    
    # Note: The Neo4j Python driver uses connection pooling, so the sessions 
    # opened inside predict_action will reuse the same underlying TCP connection,
    # keeping this loop very fast (typically < 2 seconds for 1000 iterations).
    for sample in tqdm(test_samples, desc="Evaluating"):
        # Construct the observed object list exactly as the live CV pipeline would,
        # but using the REAL geometry stats saved on the ActionInstance node.
        observed_objects = [{
            "object_id": sample["object_id"],
            "d_Ho": sample["d_Ho"] or 0.0,
            "theta": sample["theta"] or 0.0,
            "e": sample["e"] or 0.0
        }]
        
        # Call the exact same inference function used in the live pipeline
        ranked = predict_action(
            driver, 
            scene_id=sample["scene_id"], 
            observed_objects=observed_objects,
            human_id=sample["human_id"], 
            prev_action_id=sample["prev_action_id"]
        )
        
        if ranked:
            top_pred_action = ranked[0]["action_id"]
            if top_pred_action == sample["true_action_id"]:
                correct += 1
                
    driver.close() # Close the main driver
    elapsed = time.time() - start_time
    
    accuracy = correct / total if total > 0 else 0
    
    print("\n")
    print("="*60)
    print("PROBABILISTIC BASELINE RESULTS")
    print("="*60)
    print(f"Probabilistic Accuracy : {accuracy:.4f}")
    print(f"Correct Predictions    : {correct}")
    print(f"Total Samples          : {total}")
    print(f"Time taken             : {elapsed:.2f} seconds")
    print("="*60)
    print("""
PHD CONTRIBUTION SUMMARY:
-------------------------
GAT Knowledge Reasoner : 0.6960 (69.6%)
Probabilistic Baseline : """ + f"{accuracy:.4f}" + f" ({accuracy*100:.1f}%)" + """

Conclusion: 
If GAT > Baseline, your GNN successfully captures non-linear spatial 
relationships that the naive-Bayes probabilistic model misses!
""")

if __name__ == "__main__":
    evaluate_probabilistic()