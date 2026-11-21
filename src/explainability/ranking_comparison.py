"""
Explainable Ranking Comparison and Movement Analysis for Legal Prior Case Retrieval.

Compares candidate rankings across 4 retrieval regimes:
1. TF-IDF-only ranking
2. InLegalBERT-only ranking
3. Legal Knowledge-only ranking
4. Full Hybrid ranking

Analyzes rank movements:
- Promoted True Positives: Precedents moving upward into top ranks due to semantic/legal synergy.
- Demoted False Positives: Non-relevant candidates demoted due to lack of dense/legal backing.
- Top-1 Precision Stability: Ground-truth precedents maintaining rank 1 across regimes.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import pandas as pd


class RankingComparisonAnalyzer:
    """Analyzes and explains candidate rank transitions across retrieval paradigms."""

    def __init__(
        self,
        query_col: str = "query_id",
        cand_col: str = "candidate_id",
        label_col: str = "label",
    ):
        self.query_col = query_col
        self.cand_col = cand_col
        self.label_col = label_col

    def compare_query_rankings(
        self,
        df: pd.DataFrame,
        query_id: str,
    ) -> Dict[str, Any]:
        """
        Computes multi-model rankings for a given query and identifies key rank transitions.

        Returns:
            Dict containing:
            - query_id
            - num_candidates
            - num_positives
            - promoted_positives: list of true precedents that improved rank in hybrid vs TF-IDF
            - demoted_negatives: list of non-relevant candidates that dropped in hybrid vs TF-IDF
            - top_candidates_table: dataframe of top-10 hybrid candidates with rank comparison
        """
        sub = df[df[self.query_col] == query_id].copy()
        if len(sub) == 0:
            raise ValueError(f"Query {query_id} not found in DataFrame.")

        # Compute ranks (1-indexed, ascending: rank 1 is best)
        sub["rank_tfidf"] = sub["tfidf_score"].rank(ascending=False, method="min").astype(int)
        sub["rank_dense"] = sub["inlegalbert_sim"].rank(ascending=False, method="min").astype(int)
        sub["rank_legal"] = sub["composite_legal_score"].rank(ascending=False, method="min").astype(int)
        sub["rank_hybrid"] = sub["hybrid_score"].rank(ascending=False, method="min").astype(int)

        # Rank delta: positive means rank number decreased (i.e. IMPROVED in position, e.g. rank 10 -> rank 3 = delta +7)
        sub["rank_delta_vs_tfidf"] = sub["rank_tfidf"] - sub["rank_hybrid"]
        sub["rank_delta_vs_dense"] = sub["rank_dense"] - sub["rank_hybrid"]

        positives = sub[sub[self.label_col] == 1]
        negatives = sub[sub[self.label_col] == 0]

        # 1. Promoted Positives (True precedents that moved UP in hybrid vs TF-IDF)
        promoted_positives = []
        for _, row in positives.iterrows():
            if row["rank_delta_vs_tfidf"] > 0:
                promoted_positives.append({
                    "candidate_id": str(row[self.cand_col]),
                    "tfidf_rank": int(row["rank_tfidf"]),
                    "dense_rank": int(row["rank_dense"]),
                    "legal_rank": int(row["rank_legal"]),
                    "hybrid_rank": int(row["rank_hybrid"]),
                    "rank_improvement": int(row["rank_delta_vs_tfidf"]),
                    "reason": self._explain_promotion(row),
                })

        # 2. Demoted False Positives (Non-relevant candidates in top 10 TF-IDF that dropped)
        demoted_negatives = []
        high_tfidf_negs = negatives[negatives["rank_tfidf"] <= 10]
        for _, row in high_tfidf_negs.iterrows():
            if row["rank_delta_vs_tfidf"] < 0:
                demoted_negatives.append({
                    "candidate_id": str(row[self.cand_col]),
                    "tfidf_rank": int(row["rank_tfidf"]),
                    "dense_rank": int(row["rank_dense"]),
                    "legal_rank": int(row["rank_legal"]),
                    "hybrid_rank": int(row["rank_hybrid"]),
                    "rank_drop": int(-row["rank_delta_vs_tfidf"]),
                    "reason": self._explain_demotion(row),
                })

        # 3. Top-10 Hybrid Candidates Table
        top_hybrid = sub.sort_values(by="hybrid_score", ascending=False).head(10)
        table_rows = []
        for _, row in top_hybrid.iterrows():
            table_rows.append({
                "candidate_id": str(row[self.cand_col]),
                "is_precedent": bool(row[self.label_col] == 1),
                "hybrid_rank": int(row["rank_hybrid"]),
                "hybrid_score": round(float(row["hybrid_score"]), 4),
                "tfidf_rank": int(row["rank_tfidf"]),
                "dense_rank": int(row["rank_dense"]),
                "legal_rank": int(row["rank_legal"]),
                "delta_vs_tfidf": int(row["rank_delta_vs_tfidf"]),
            })

        return {
            "query_id": query_id,
            "total_candidates": len(sub),
            "total_positives": len(positives),
            "promoted_positives": promoted_positives,
            "demoted_false_positives": demoted_negatives,
            "top_10_hybrid_candidates": table_rows,
        }

    @staticmethod
    def _explain_promotion(row: pd.Series) -> str:
        dense_rank = row["rank_dense"]
        legal_rank = row["rank_legal"]
        tfidf_rank = row["rank_tfidf"]
        reasons = []
        if dense_rank < tfidf_rank:
            reasons.append(f"strong dense semantic alignment (Dense Rank #{int(dense_rank)})")
        if legal_rank < tfidf_rank:
            reasons.append(f"high legal feature overlap (Legal Rank #{int(legal_rank)})")
        if not reasons:
            reasons.append("balanced synergistic contribution across modalities")
        return "Promoted due to " + " and ".join(reasons) + "."

    @staticmethod
    def _explain_demotion(row: pd.Series) -> str:
        dense_rank = row["rank_dense"]
        legal_rank = row["rank_legal"]
        tfidf_rank = row["rank_tfidf"]
        reasons = []
        if dense_rank > tfidf_rank:
            reasons.append(f"poor semantic similarity (Dense Rank #{int(dense_rank)})")
        if legal_rank > tfidf_rank:
            reasons.append(f"lack of statutory/citation overlap (Legal Rank #{int(legal_rank)})")
        if not reasons:
            reasons.append("lower comparative score against higher-scoring hybrid candidates")
        return "Correctly demoted due to " + " and ".join(reasons) + "."

