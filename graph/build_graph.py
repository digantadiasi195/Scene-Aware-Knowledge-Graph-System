# build_graph.py
"""
Top-level orchestrator. Runs the full pipeline end-to-end:

    1. schema.apply_schema        -- constraints/indexes
    2. etl_pipeline.run_etl       -- ontology layer + instance/observation layer
    3. stats_learning.learn_all   -- learns every probability/statistic
                                      from the instance layer (no hardcoded numbers)

Run:
    python build_graph.py                        # uses raw_observations.csv
    python build_graph.py my_real_data.csv        # your real extracted features
    python build_graph.py --wipe                 # wipe DB first
    python build_graph.py --generate              # (re)generate the synthetic demo CSV first
"""

import sys
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from etl_pipeline import run_etl
from stats_learning import learn_all


def main():
    wipe_flag = "--wipe" in sys.argv
    generate_flag = "--generate" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_path = positional[0] if positional else "raw_observations.csv"

    if generate_flag:
        from sample_dataset_generator import generate
        csv_path = generate(csv_path)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        run_etl(csv_path, driver, wipe=wipe_flag)
        learn_all(driver)
        print("\nDatabase build complete: ontology + instances + learned statistics.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()