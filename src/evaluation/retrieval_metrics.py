"""
Retrieval Metrics for Legal Prior Case Retrieval Evaluation.

Implements standard Information Retrieval evaluation metrics:
- Precision@K (P@1, P@5, P@10)
- Recall@K (R@5, R@10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@5, nDCG@10)
"""

from typing import Any, Dict, List, Sequence, Set, Union
import numpy as np


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Computes Precision@K: (number of relevant documents in top K) / K.
    
    Args:
        retrieved_ids: Ordered sequence of retrieved candidate IDs.
        relevant_ids: Set or sequence of ground-truth relevant document IDs.
        k: Cut-off rank.
        
    Returns:
        float: Precision at rank k.
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return float(hits / k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Computes Recall@K: (number of relevant documents in top K) / (total relevant documents).
    
    Args:
        retrieved_ids: Ordered sequence of retrieved candidate IDs.
        relevant_ids: Set or sequence of ground-truth relevant document IDs.
        k: Cut-off rank.
        
    Returns:
        float: Recall at rank k. Returns 0.0 if there are no relevant documents.
    """
    rel_set = set(relevant_ids)
    if len(rel_set) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return float(hits / len(rel_set))


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of first relevant document retrieved.
    
    Args:
        retrieved_ids: Ordered sequence of retrieved candidate IDs.
        relevant_ids: Set or sequence of ground-truth relevant document IDs.
        
    Returns:
        float: 1/rank if a relevant document is found, else 0.0.
    """
    rel_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel_set:
            return float(1.0 / rank)
    return 0.0


def dcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Computes Discounted Cumulative Gain at rank K with binary relevance.
    Formula: sum_{i=1}^K rel_i / log2(i + 1)
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in rel_set:
            dcg += 1.0 / np.log2(i + 1.0)
    return float(dcg)


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K.
    Formula: DCG@K / IDCG@K
    
    Args:
        retrieved_ids: Ordered sequence of retrieved candidate IDs.
        relevant_ids: Set or sequence of ground-truth relevant document IDs.
        k: Cut-off rank.
        
    Returns:
        float: nDCG at rank k. Returns 0.0 if total relevant documents is 0.
    """
    rel_set = set(relevant_ids)
    n_rel = len(rel_set)
    if n_rel == 0 or k <= 0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, rel_set, k)

    # Ideal DCG: top min(k, n_rel) positions are all relevant
    ideal_k = min(k, n_rel)
    idcg = sum(1.0 / np.log2(i + 1.0) for i in range(1, ideal_k + 1))

    if idcg <= 0.0:
        return 0.0
    return float(actual_dcg / idcg)


def evaluate_query_ranking(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
) -> Dict[str, float]:
    """
    Evaluates a single query's ranked retrieval results across standard metrics.
    
    Returns:
        Dict containing P@1, P@5, P@10, R@5, R@10, MRR, nDCG@5, nDCG@10.
    """
    rel_set = set(relevant_ids)
    return {
        "P@1": precision_at_k(retrieved_ids, rel_set, 1),
        "P@5": precision_at_k(retrieved_ids, rel_set, 5),
        "P@10": precision_at_k(retrieved_ids, rel_set, 10),
        "R@5": recall_at_k(retrieved_ids, rel_set, 5),
        "R@10": recall_at_k(retrieved_ids, rel_set, 10),
        "MRR": reciprocal_rank(retrieved_ids, rel_set),
        "nDCG@5": ndcg_at_k(retrieved_ids, rel_set, 5),
        "nDCG@10": ndcg_at_k(retrieved_ids, rel_set, 10),
    }


def aggregate_retrieval_metrics(
    query_evaluations: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregates per-query evaluation dictionaries into mean benchmark scores.
    """
    if not query_evaluations:
        return {
            "P@1": 0.0, "P@5": 0.0, "P@10": 0.0,
            "R@5": 0.0, "R@10": 0.0,
            "MRR": 0.0,
            "nDCG@5": 0.0, "nDCG@10": 0.0,
        }

    keys = ["P@1", "P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@5", "nDCG@10"]
    return {
        key: round(float(np.mean([q[key] for q in query_evaluations])), 4)
        for key in keys
    }

