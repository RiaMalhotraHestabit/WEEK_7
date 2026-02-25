import os
import json
from retriever.hybrid_retriever import query_hybrid_text
from pipelines.image_ingest import query_image
from pipelines.sql_pipeline import query_sql

LOG_FILE = "EVAL-LOGS.json"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f, indent=2)

test_cases = [
    {
        "mode": "Text",
        "query": "Explain executive compensation structure in the documents."
    },
    {
        "mode": "Image",
        "query": "src/data/raw/images/bar_chart_1.png"  # Replace with a real image path
    },
    {
        "mode": "SQL",
        "query": "How many customers have a balance over 1000?"
    }
]

def run_tests():
    results = []

    for case in test_cases:
        mode = case["mode"]
        query = case["query"]
        print(f"\n=== Mode: {mode} ===")
        print(f"Query: {query}")

        if mode == "Text":
            res = query_hybrid_text(query)
        elif mode == "Image":
            if not os.path.exists(query):
                print(f"Image not found: {query}")
                continue
            res = query_image(query)
        elif mode == "SQL":
            res = query_sql(query)
        else:
            continue

        print(f"Answer: {res['answer'][:500]}...")
        print(f"Confidence: {res['confidence']:.2f}")

        results.append({
            "mode": mode,
            "query": query,
            "answer": res["answer"],
            "confidence": res["confidence"]
        })

    # Save evaluation logs
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n Evaluation complete. Results saved to EVAL-LOGS.json")

if __name__ == "__main__":
    run_tests()