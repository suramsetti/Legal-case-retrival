"""
Phase 16: Advanced Multi-Stage Legal Retrieval Engine.

Implements a 4-Stage Domain-Specific Retrieval and Ranking Pipeline:
1. Stage 1: Calibrated Multi-Modal Candidate Retrieval
   Combines TF-IDF (0.68), BM25 (0.06), InLegalBERT Dense (0.14), and Composite Legal Knowledge (0.12).
2. Stage 2: Deep Legal Feature Analysis
   Quantifies statutory containment, citation networks, entity overlap, and rhetorical discourse alignment.
3. Stage 3: Neural & Cross-Encoder Reranking
   Applies margin-gated neural refinement (tau = 0.065) exclusively to ambiguous, low-margin query pools.
4. Stage 4: Confidence-Gated Adaptive Fusion with Dual Guardrails
   Evaluates borderline candidates (score diff < 0.008) with statutory superiority (> 0.35) and lexical guardrails (>= -0.20).
   Guarantees 100% deterministic fallback to Phase 14 / Phase 15 baselines.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.ranking.adaptive_reranker import QueryAdaptiveReRanker
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.ranking.phase14_reranker import Phase14SelectiveReRanker


class Phase16MultiStageReranker(Phase14SelectiveReRanker):
    """
    Phase 16 Multi-Stage Retrieval Engine.
    Integrates Domain-Specific Transformer Tuning, Multi-Modal Stage-1 Candidate Retrieval,
    Margin-Gated Neural Interaction, and Confidence-Calibrated Adaptive Overrides.
    """

    def __init__(
        self,
        tau: float = 0.065,
        alpha: float = 0.12,
        gamma: float = 0.04,
        top_k: int = 30,
        override_delta: float = 0.008,
        statute_threshold: float = 0.35,
        lexical_guardrail: float = -0.20,
        w_tfidf: float = 0.68,
        w_bm25: float = 0.06,
        w_dense: float = 0.14,
        w_legal: float = 0.12,
        pairwise_reranker: Optional[PairwiseNeuralReRanker] = None,
        gemini_service: Optional[Any] = None,
    ):
        super().__init__(
            tau=tau,
            alpha=alpha,
            gamma=gamma,
            top_k=top_k,
            override_delta=override_delta,
            statute_threshold=statute_threshold,
            lexical_guardrail=lexical_guardrail,
            pairwise_reranker=pairwise_reranker,
            gemini_service=gemini_service,
        )
        self.w_tfidf = w_tfidf
        self.w_bm25 = w_bm25
        self.w_dense = w_dense
        self.w_legal = w_legal

    @classmethod
    def compute_stage1_features(
        cls,
        df: pd.DataFrame,
        w_tfidf: float = 0.68,
        w_bm25: float = 0.06,
        w_dense: float = 0.14,
        w_legal: float = 0.12,
    ) -> pd.DataFrame:
        """
        Stage 1: Multi-modal score calculation combining TF-IDF, BM25, InLegalBERT, and Legal Knowledge.
        """
        df = df.copy()
        for col in ["tfidf_score", "inlegalbert_sim", "composite_legal_score", "bm25_score"]:
            if col in df.columns and "qnorm_" + col not in df.columns:
                s_min = df.groupby("query_id")[col].transform("min")
                s_max = df.groupby("query_id")[col].transform("max")
                diff = s_max - s_min
                df["qnorm_" + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

        # Multi-modal fusion score
        tfidf_val = df.get("qnorm_tfidf_score", df.get("norm_tfidf_score", 0.0))
        bm25_val = df.get("qnorm_bm25_score", df.get("norm_bm25_score", 0.0))
        dense_val = df.get("qnorm_inlegalbert_sim", df.get("norm_inlegalbert_sim", 0.0))
        legal_val = df.get("qnorm_composite_legal_score", df.get("norm_composite_legal_score", 0.0))

        df["stage1_score"] = (
            w_tfidf * tfidf_val
            + w_bm25 * bm25_val
            + w_dense * dense_val
            + w_legal * legal_val
        )

        df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")
        return df

    def rerank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes full Phase 16 4-Stage Retrieval:
        1. Multi-modal Stage-1 Candidate scoring.
        2. Pairwise feature interaction engineering.
        3. Adaptive margin gating (high-margin preservation vs low-margin neural refinement).
        4. Confidence-gated selective overrides with dual guardrails.
        5. Recalculation of final ranks and metrics.
        """
        # Ensure Stage-1 features are computed with Phase 16 weights
        df_stage1 = self.compute_stage1_features(
            df,
            w_tfidf=self.w_tfidf,
            w_bm25=self.w_bm25,
            w_dense=self.w_dense,
            w_legal=self.w_legal,
        )

        # Run base adaptive engine
        df_out = self.base_engine.rerank(df_stage1).copy()
        df_out["p16_score"] = df_out["final_adaptive_score"].values.copy()
        df_out["override_applied"] = False
        df_out["override_reason"] = "none"

        for qid, grp in df_out.groupby("query_id"):
            sub = grp.sort_values("final_adaptive_score", ascending=False)
            if len(sub) < 2:
                continue

            # Only evaluate ambiguous queries in the low-margin regime
            if sub.iloc[0]["stage1_margin"] >= self.tau:
                continue

            r1 = sub.iloc[0]
            r2 = sub.iloc[1]

            score_diff = float(r1["final_adaptive_score"] - r2["final_adaptive_score"])
            stat_diff = float(r2.get("norm_statute_jaccard", 0.0) - r1.get("norm_statute_jaccard", 0.0))
            tfidf_diff = float(r2.get("norm_tfidf_score", 0.0) - r1.get("norm_tfidf_score", 0.0))

            # Confidence-gated override guardrails:
            # 1. Borderline score margin between Rank-1 and Rank-2 (< override_delta)
            # 2. Rank-2 has verified statutory superiority (> statute_threshold)
            # 3. Rank-1 does not possess overwhelming lexical advantage (>= lexical_guardrail)
            if (
                score_diff < self.override_delta
                and stat_diff > self.statute_threshold
                and tfidf_diff >= self.lexical_guardrail
            ):
                r2_idx = r2.name
                boost = score_diff + 0.002
                df_out.loc[r2_idx, "p16_score"] += boost
                df_out.loc[r2_idx, "override_applied"] = True
                df_out.loc[r2_idx, "override_reason"] = f"statutory_superiority_boost_{boost:.4f}"

        # Re-rank based on p16_score
        df_out["p16_rank"] = df_out.groupby("query_id")["p16_score"].rank(ascending=False, method="first")
        return df_out

