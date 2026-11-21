"""
Adaptive Cross-Encoder Re-Ranker (Phase 12).

Integrates InLegalBERTCrossEncoder into the Phase 11 margin-gated retrieval architecture:
1. High-margin queries (margin >= tau) strictly preserve their Stage-1 order.
2. Low-margin ambiguous queries (margin < tau) invoke the Cross-Encoder on Top-N candidates (N in {3, 5, 10}).
3. Employs section-aware chunking and rhetorical-weighted score aggregation.
4. Preserves full candidate pools, relevance labels, and deterministic tie-breaking.
"""

from typing import Dict, List, Optional, Tuple, Any, Sequence, Union
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from src.ranking.legal_cross_encoder import InLegalBERTCrossEncoder, SectionAwareChunker
from src.evaluation.hybrid_evaluator import HybridEvaluator


class AdaptiveCrossReranker:
    """
    Combines Phase 11 confidence margin gating with Phase 12 section-aware Cross-Encoder re-ranking.
    """

    def __init__(
        self,
        tau: float = 0.07,
        beta: float = 0.25,
        gamma: float = 0.04,
        top_n: int = 5,
        cross_encoder: Optional[InLegalBERTCrossEncoder] = None,
        chunk_aggregation: str = "weighted",
    ):
        """
        Args:
            tau: Margin threshold separating high-confidence from low-confidence queries.
            beta: Weight assigned to the Cross-Encoder score in low-margin regime.
            gamma: Weight assigned to rhetorical discourse similarity in low-margin regime.
            top_n: Number of top Stage-1 candidates re-ranked by the Cross-Encoder (e.g. 3, 5, 10).
            cross_encoder: Pre-trained InLegalBERTCrossEncoder instance.
            chunk_aggregation: 'weighted', 'max', or 'mean'.
        """
        self.tau = float(tau)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.top_n = int(top_n)
        self.cross_encoder = cross_encoder
        self.chunk_aggregation = chunk_aggregation
        self.chunker = SectionAwareChunker()

    @classmethod
    def compute_stage1_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures query-normalized features, Stage-1 hybrid score, and Stage-1 ranks exist."""
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

        if "s1_rank" not in df.columns:
            df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

        return df

    @classmethod
    def compute_query_margins(cls, df: pd.DataFrame, score_col: str = "stage1_score") -> pd.Series:
        """Calculates margin between Rank-1 and Rank-2 candidates for each query."""
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
        """Partitions queries into 'high_margin' and 'low_margin' regimes."""
        df = self.compute_stage1_features(df)
        if "stage1_margin" not in df.columns:
            df["stage1_margin"] = self.compute_query_margins(df, score_col=score_col)

        df["is_low_margin"] = df["stage1_margin"] < self.tau
        df["routing_regime"] = np.where(df["is_low_margin"], "low_margin_reranked", "high_margin_preserved")
        return df

    def rerank_with_precomputed_ce_scores(
        self,
        df: pd.DataFrame,
        ce_scores: Dict[Tuple[str, str], float],
        tau: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fast re-ranking using precomputed Cross-Encoder (query_id, candidate_id) scores.
        """
        tau_val = float(tau if tau is not None else self.tau)
        beta_val = float(beta if beta is not None else self.beta)
        gamma_val = float(gamma if gamma is not None else self.gamma)
        n_val = int(top_n if top_n is not None else self.top_n)

        df = self.classify_queries(df, score_col="stage1_score")
        df["final_cross_score"] = df["stage1_score"].values.copy()

        low_margin_qids = df[df["stage1_margin"] < tau_val]["query_id"].unique()
        if len(low_margin_qids) == 0:
            df["cross_rank"] = df.groupby("query_id")["final_cross_score"].rank(ascending=False, method="first")
            return df

        is_low_margin = df["query_id"].isin(low_margin_qids)
        is_top_n = df["s1_rank"] <= n_val
        mask_rerank = is_low_margin & is_top_n
        mask_low_other = is_low_margin & (~is_top_n)

        # Depress candidates outside top_n
        df.loc[mask_low_other, "final_cross_score"] = df.loc[mask_low_other, "stage1_score"] - 100.0

        top_slice = df[mask_rerank].copy()
        if len(top_slice) > 0:
            # Map precomputed cross-encoder scores
            raw_ce = [ce_scores.get((row["query_id"], row["candidate_id"]), 0.0) for _, row in top_slice.iterrows()]
            top_slice["ce_raw"] = raw_ce

            # Query-level min-max normalize CE scores
            ce_min = top_slice.groupby("query_id")["ce_raw"].transform("min")
            ce_max = top_slice.groupby("query_id")["ce_raw"].transform("max")
            ce_diff = ce_max - ce_min
            top_slice["ce_norm"] = (top_slice["ce_raw"] - ce_min) / np.where(ce_diff > 1e-9, ce_diff, 1.0)

            # Rhetorical alignment
            if "rhetorical_cosine" in top_slice.columns and gamma_val > 0.0:
                r_min = top_slice.groupby("query_id")["rhetorical_cosine"].transform("min")
                r_max = top_slice.groupby("query_id")["rhetorical_cosine"].transform("max")
                r_diff = r_max - r_min
                top_slice["qn_rhet"] = (top_slice["rhetorical_cosine"] - r_min) / np.where(r_diff > 1e-9, r_diff, 1.0)
            else:
                top_slice["qn_rhet"] = 0.0

            s1_w = max(0.0, 1.0 - beta_val - gamma_val)
            final_scores = (
                s1_w * top_slice["stage1_score"]
                + beta_val * top_slice["ce_norm"]
                + gamma_val * top_slice["qn_rhet"]
            )
            df.loc[top_slice.index, "final_cross_score"] = final_scores.values

        df["cross_rank"] = df.groupby("query_id")["final_cross_score"].rank(ascending=False, method="first")
        return df

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Saves adaptive cross-encoder configuration."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        config = {
            "tau": self.tau,
            "beta": self.beta,
            "gamma": self.gamma,
            "top_n": self.top_n,
            "chunk_aggregation": self.chunk_aggregation,
            "model_type": "AdaptiveCrossReranker",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filepath: Union[str, Path], cross_encoder: Optional[InLegalBERTCrossEncoder] = None) -> "AdaptiveCrossReranker":
        """Loads configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            tau=config.get("tau", 0.07),
            beta=config.get("beta", 0.25),
            gamma=config.get("gamma", 0.04),
            top_n=config.get("top_n", 5),
            cross_encoder=cross_encoder,
            chunk_aggregation=config.get("chunk_aggregation", "weighted"),
        )

