"""
Query-Adaptive Margin-Gated Legal Case Re-Ranker.

Implements confidence-aware two-stage retrieval:
1. Calculates Stage-1 score margin between Rank-1 and Rank-2 candidates.
2. High-margin queries (margin >= tau) retain their verified Stage-1 ranking.
3. Low-margin queries (margin < tau) are selectively re-ranked using a combination
   of pairwise neural re-ranking and rhetorical discourse alignment.
4. Preserves full candidate pools, relevance labels, and deterministic tie-breaking.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator


class QueryAdaptiveReRanker:
    """
    Query-Adaptive Margin-Gated Re-Ranker for Legal Prior Case Retrieval.
    """

    def __init__(
        self,
        tau: float = 0.07,
        alpha: float = 0.12,
        gamma: float = 0.04,
        top_k: int = 30,
        pairwise_reranker: Optional[PairwiseNeuralReRanker] = None,
    ):
        """
        Args:
            tau: Margin threshold separating high-confidence from low-confidence queries.
            alpha: Weight assigned to the pairwise neural re-ranker score in low-margin regime.
            gamma: Weight assigned to rhetorical discourse similarity in low-margin regime.
            top_k: Candidate pool depth re-ranked for low-margin queries.
            pairwise_reranker: Optional pre-loaded PairwiseNeuralReRanker instance.
        """
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.top_k = int(top_k)
        self.pairwise_reranker = pairwise_reranker

    @classmethod
    def compute_stage1_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures query-normalized features, Stage-1 hybrid score, and Stage-1 ranks exist.
        Stage 1 Score = 0.65 * qnorm_tfidf + 0.25 * qnorm_dense + 0.10 * qnorm_legal.
        """
        df = df.copy()
        if "stage1_score" not in df.columns:
            for col in ["tfidf_score", "inlegalbert_sim", "composite_legal_score", "bm25_score"]:
                s_min = df.groupby("query_id")[col].transform("min")
                s_max = df.groupby("query_id")[col].transform("max")
                diff = s_max - s_min
                df["qnorm_" + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

            df["stage1_score"] = (
                0.65 * df["qnorm_tfidf_score"]
                + 0.25 * df["qnorm_inlegalbert_sim"]
                + 0.10 * df["qnorm_composite_legal_score"]
            )

        # Stage-1 rank: deterministic descending by stage1_score, tie-breaking by candidate_id
        if "s1_rank" not in df.columns:
            df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

        return df

    @classmethod
    def compute_query_margins(cls, df: pd.DataFrame, score_col: str = "stage1_score") -> pd.Series:
        """
        Calculates the margin between Rank-1 and Rank-2 candidates for each query.
        margin(q) = score(q, rank1) - score(q, rank2).
        For queries with only 1 candidate, margin is set to 1.0 (unambiguous).
        """
        margin_dict: Dict[str, float] = {}
        grouped = df.groupby("query_id")
        for qid, grp in grouped:
            top_vals = grp[score_col].nlargest(2).values
            if len(top_vals) >= 2:
                margin_dict[qid] = float(top_vals[0] - top_vals[1])
            elif len(top_vals) == 1:
                margin_dict[qid] = 1.0
            else:
                margin_dict[qid] = 0.0

        return df["query_id"].map(margin_dict)

    def classify_queries(self, df: pd.DataFrame, score_col: str = "stage1_score") -> pd.DataFrame:
        """
        Partitions queries into 'high_margin' and 'low_margin' regimes.
        Adds columns: 'stage1_margin', 'is_low_margin', 'routing_regime'.
        """
        df = self.compute_stage1_features(df)
        if "stage1_margin" not in df.columns:
            df["stage1_margin"] = self.compute_query_margins(df, score_col=score_col)

        df["is_low_margin"] = df["stage1_margin"] < self.tau
        df["routing_regime"] = np.where(df["is_low_margin"], "low_margin_reranked", "high_margin_preserved")
        return df

    def rerank(
        self,
        df: pd.DataFrame,
        tau: Optional[float] = None,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Executes query-adaptive margin-gated re-ranking.

        High-margin queries (margin >= tau):
            Preserve exact Stage-1 score and ordering.

        Low-margin queries (margin < tau):
            For top_k candidates:
                s_final = (1 - alpha - gamma) * stage1_score + alpha * s2_norm + gamma * qn_rhet
            For candidates outside top_k:
                s_final = stage1_score - 100.0 (strictly preserved below top_k)

        Returns:
            pd.DataFrame with 'final_adaptive_score', 'adaptive_rank', 'stage1_margin', 'routing_regime'.
        """
        tau_val = float(tau if tau is not None else self.tau)
        alpha_val = float(alpha if alpha is not None else self.alpha)
        gamma_val = float(gamma if gamma is not None else self.gamma)
        k_val = int(top_k if top_k is not None else self.top_k)

        df = self.compute_stage1_features(df)
        df = self.classify_queries(df, score_col="stage1_score")

        # Initialize final adaptive score to stage1_score
        df["final_adaptive_score"] = df["stage1_score"].values.copy()

        # Identify low-margin queries
        low_margin_qids = df[df["stage1_margin"] < tau_val]["query_id"].unique()
        if len(low_margin_qids) == 0:
            df["adaptive_rank"] = df.groupby("query_id")["final_adaptive_score"].rank(ascending=False, method="first")
            return df

        # Isolate low-margin top_k slice
        is_low_margin = df["query_id"].isin(low_margin_qids)
        is_top_k = df["s1_rank"] <= k_val
        mask_rerank = is_low_margin & is_top_k
        mask_low_other = is_low_margin & (~is_top_k)

        # 1. Depress low-margin candidates outside top_k so they never jump into top_k
        df.loc[mask_low_other, "final_adaptive_score"] = df.loc[mask_low_other, "stage1_score"] - 100.0

        # 2. Compute Stage-2 pairwise score on low-margin top_k slice
        top_slice = df[mask_rerank].copy()
        if len(top_slice) > 0 and self.pairwise_reranker is not None:
            # Predict raw neural scores
            s2_raw = self.pairwise_reranker.predict_scores(top_slice)
            top_slice["s2_raw"] = s2_raw

            # Query-level min-max normalize Stage-2 score within the top slice
            s2_min = top_slice.groupby("query_id")["s2_raw"].transform("min")
            s2_max = top_slice.groupby("query_id")["s2_raw"].transform("max")
            s2_diff = s2_max - s2_min
            s2_norm = (top_slice["s2_raw"] - s2_min) / np.where(s2_diff > 1e-9, s2_diff, 1.0)
            top_slice["s2_norm"] = s2_norm
        else:
            top_slice["s2_norm"] = top_slice["stage1_score"]

        # 3. Rhetorical alignment feature adjustment
        if "rhetorical_cosine" in top_slice.columns and gamma_val > 0.0:
            r_min = top_slice.groupby("query_id")["rhetorical_cosine"].transform("min")
            r_max = top_slice.groupby("query_id")["rhetorical_cosine"].transform("max")
            r_diff = r_max - r_min
            qn_rhet = (top_slice["rhetorical_cosine"] - r_min) / np.where(r_diff > 1e-9, r_diff, 1.0)
            top_slice["qn_rhet"] = qn_rhet
        else:
            top_slice["qn_rhet"] = 0.0

        # 4. Convex blend for low-margin ambiguous candidates
        s1_w = max(0.0, 1.0 - alpha_val - gamma_val)
        final_scores = (
            s1_w * top_slice["stage1_score"]
            + alpha_val * top_slice["s2_norm"]
            + gamma_val * top_slice["qn_rhet"]
        )

        df.loc[top_slice.index, "final_adaptive_score"] = final_scores.values

        # Compute adaptive rank
        df["adaptive_rank"] = df.groupby("query_id")["final_adaptive_score"].rank(ascending=False, method="first")
        return df

    def evaluate_split(
        self,
        df: pd.DataFrame,
        tau: Optional[float] = None,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs adaptive re-ranking and computes overall, high-margin, and low-margin retrieval metrics.
        """
        t0 = time.time()
        reranked_df = self.rerank(df, tau=tau, alpha=alpha, gamma=gamma, top_k=top_k)
        elapsed_sec = time.time() - t0

        overall_metrics = HybridEvaluator.evaluate_predictions(reranked_df, score_col="final_adaptive_score")
        s1_metrics = HybridEvaluator.evaluate_predictions(reranked_df, score_col="stage1_score")

        # Routing statistics
        q_regimes = reranked_df.groupby("query_id")["routing_regime"].first()
        total_queries = len(q_regimes)
        low_count = int((q_regimes == "low_margin_reranked").sum())
        high_count = total_queries - low_count
        pct_reranked = float(low_count / max(total_queries, 1) * 100.0)

        # Promotion / demotion counts
        reranked_df["s1_r"] = reranked_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")
        reranked_df["s2_r"] = reranked_df["adaptive_rank"]

        promoted = 0
        demoted = 0
        for qid, grp in reranked_df.groupby("query_id"):
            pos = grp[grp["label"] == 1]
            if pos.empty:
                continue
            r_s1 = pos["s1_r"].min()
            r_s2 = pos["s2_r"].min()
            if r_s1 > 1 and r_s2 == 1:
                promoted += 1
            if r_s1 == 1 and r_s2 > 1:
                demoted += 1

        # Regime-specific evaluations
        high_df = reranked_df[reranked_df["routing_regime"] == "high_margin_preserved"]
        low_df = reranked_df[reranked_df["routing_regime"] == "low_margin_reranked"]

        high_metrics = HybridEvaluator.evaluate_predictions(high_df, score_col="final_adaptive_score") if len(high_df) > 0 else {}
        low_metrics = HybridEvaluator.evaluate_predictions(low_df, score_col="final_adaptive_score") if len(low_df) > 0 else {}

        ms_per_query = (elapsed_sec / max(total_queries, 1)) * 1000.0

        return {
            "tau": float(tau if tau is not None else self.tau),
            "alpha": float(alpha if alpha is not None else self.alpha),
            "gamma": float(gamma if gamma is not None else self.gamma),
            "top_k": int(top_k if top_k is not None else self.top_k),
            "total_queries": total_queries,
            "queries_reranked": low_count,
            "queries_preserved": high_count,
            "pct_reranked": pct_reranked,
            "promoted_to_rank1": promoted,
            "demoted_from_rank1": demoted,
            "net_rank1_change": promoted - demoted,
            "inference_time_sec": round(elapsed_sec, 4),
            "latency_ms_per_query": round(ms_per_query, 2),
            "stage1_baseline": s1_metrics,
            "overall_adaptive": overall_metrics,
            "high_margin_regime": high_metrics,
            "low_margin_regime": low_metrics,
        }

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Saves adaptive reranker hyperparameter configuration."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        config = {
            "tau": self.tau,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "top_k": self.top_k,
            "model_type": "QueryAdaptiveReRanker",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filepath: Union[str, Path], pairwise_reranker: Optional[PairwiseNeuralReRanker] = None) -> "QueryAdaptiveReRanker":
        """Loads configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            tau=config.get("tau", 0.07),
            alpha=config.get("alpha", 0.12),
            gamma=config.get("gamma", 0.04),
            top_k=config.get("top_k", 30),
            pairwise_reranker=pairwise_reranker,
        )

