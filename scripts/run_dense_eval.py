"""
Run Dense Semantic Retrieval Evaluation.

Encodes candidate documents and queries, computes cosine similarities,
evaluates dense retrieval on dev/test splits, and outputs comparison reports.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dense_evaluator import DenseEvaluator


def main():
    print("=" * 60)
    print("PHASE 4: Dense Semantic Retrieval Evaluation (InLegalBERT)")
    print("=" * 60)

    # Initialize evaluator
    evaluator = DenseEvaluator(
        model_name="inlegalbert",
        batch_size=32,
    )

    # Run evaluation on dev split (Full-Doc baseline)
    dev_results = evaluator.evaluate_split(split="dev", evaluate_chunks=False)

    # Save initial dev metrics
    out_metrics_file = PROJECT_ROOT / "results" / "evaluation" / "dense_metrics.json"
    import json
    results_dict = {"dev": dev_results}
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved dev metrics to {out_metrics_file}")


if __name__ == "__main__":
    main()
