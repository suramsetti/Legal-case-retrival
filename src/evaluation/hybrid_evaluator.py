"""
Hybrid Retrieval Evaluator for Prior Case Retrieval (PCR).

Calculates standard Information Retrieval evaluation metrics across query rankings:
- Precision@1, Precision@5, Precision@10
- Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@5, nDCG@10)
"""

from typing import Dict, List, Set, Sequence, Tuple, Any, Optional
import numpy as np
import pandas as pd


class HybridEvaluator:
    """Evaluates ranked candidate predictions against ground-truth precedent labels."""

    @staticmethod
    def dcg_at_k(r: Sequence[int], k: int) -> float:
        """Computes Discounted Cumulative Gain at rank k."""
        r = np.asarray(r, dtype=float)[:k]
        if r.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, r.size + 2))
        return float(np.sum(r / discounts))

    @staticmethod
    def ndcg_at_k(r: Sequence[int], k: int, num_positives: int) -> float:
        """Computes Normalized Discounted Cumulative Gain at rank k."""
        if num_positives == 0:
            return 0.0
        dcg = HybridEvaluator.dcg_at_k(r, k)
        ideal_r = [1] * min(num_positives, k) + [0] * max(0, k - num_positives)
        idcg = HybridEvaluator.dcg_at_k(ideal_r, k)
        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)

    @classmethod
    def evaluate_predictions(
        cls,
        df_scores: pd.DataFrame,
        score_col: str,
        query_col: str = "query_id",
        cand_col: str = "candidate_id",
        label_col: str = "label",
    ) -> Dict[str, float]:
        """
        Evaluates query-candidate ranking predictions across all queries.
        
        Args:
            df_scores: DataFrame containing query_id, candidate_id, score, and label.
            score_col: Column name containing ranking score (higher is more relevant).
            query_col: Column name for query ID.
            cand_col: Column name for candidate ID.
            label_col: Column name for binary label (1=relevant, 0=non-relevant).
            
        Returns:
            Dictionary of averaged IR metrics.
        """
        p1_list, p5_list, p10_list = [], [], []
        r5_list, r10_list = [], []
        mrr_list = []
        ndcg5_list, ndcg10_list = [], []

        # Group by query
        grouped = df_scores.groupby(query_col, sort=False)

        for qid, group in grouped:
            # Sort candidates by score descending
            sorted_group = group.sort_values(by=score_col, ascending=False)
            labels = sorted_group[label_col].values.astype(int)
            total_positives = int(np.sum(labels))

            if total_positives == 0:
                # Queries with 0 resolvable positives in the candidate pool
                continue

            # P@k
            p1 = float(labels[0]) if len(labels) >= 1 else 0.0
            p5 = float(np.sum(labels[:5])) / min(5, len(labels))
            p10 = float(np.sum(labels[:10])) / min(10, len(labels))

            # R@k
            r5 = float(np.sum(labels[:5])) / total_positives
            r10 = float(np.sum(labels[:10])) / total_positives

            # MRR (first relevant candidate rank)
            rel_indices = np.where(labels == 1)[0]
            if len(rel_indices) > 0:
                first_rank = rel_indices[0] + 1
                mrr = 1.0 / first_rank
            else:
                mrr = 0.0

            # nDCG@k
            ndcg5 = cls.ndcg_at_k(labels, k=5, num_positives=total_positives)
            ndcg10 = cls.ndcg_at_k(labels, k=10, num_positives=total_positives)

            p1_list.append(p1)
            p5_list.append(p5)
            p10_list.append(p10)
            r5_list.append(r5)
            r10_list.append(r10)
            mrr_list.append(mrr)
            ndcg5_list.append(ndcg5)
            ndcg10_list.append(ndcg10)

        n_eval = max(len(p1_list), 1)
        return {
            "P@1": round(float(np.mean(p1_list)), 4),
            "P@5": round(float(np.mean(p5_list)), 4),
            "P@10": round(float(np.mean(p10_list)), 4),
            "R@5": round(float(np.mean(r5_list)), 4),
            "R@10": round(float(np.mean(r10_list)), 4),
            "MRR": round(float(np.mean(mrr_list)), 4),
            "nDCG@5": round(float(np.mean(ndcg5_list)), 4),
            "nDCG@10": round(float(np.mean(ndcg10_list)), 4),
            "evaluated_queries": len(p1_list),
        }
