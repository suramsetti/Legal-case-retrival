"""
Evaluation package for Legal Prior Case Retrieval.
"""

from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    evaluate_query_ranking,
    aggregate_retrieval_metrics,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_query_ranking",
    "aggregate_retrieval_metrics",
]

