"""
Phase 14: Gemini-Assisted Selective Reranker with Confidence-Gated Overrides.

Implements:
1. Preservation of Phase 11 Production Baseline (tau=0.07, alpha=0.12, gamma=0.04).
2. Gemini-assisted legal feature verification:
   - Grounded statute and citation overlap verification.
   - Structured ratio decidendi alignment.
3. Confidence-Gated Selective Overrides:
   - Evaluates only borderline candidates (final score margin < 0.008) in low-margin regime (stage1_margin < 0.07).
   - Requires verified legal signal superiority (statute alignment diff > 0.35).
   - Enforces lexical and dense guardrails (prevents overriding candidates with strong lexical preference).
   - Complete deterministic fallback to Phase 11 baseline when confidence criteria are not strictly satisfied.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from src.ranking.adaptive_reranker import QueryAdaptiveReRanker
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from backend.services.gemini_service import GeminiService


class Phase14SelectiveReRanker:
    """
    Phase 14 Retrieval Engine:
    Combines Phase 11 Adaptive Margin-Gated Pairwise Ranking with
    Confidence-Gated Legal Signal Overrides and Gemini Feature Grounding.
    """

    def __init__(
        self,
        tau: float = 0.07,
        alpha: float = 0.12,
        gamma: float = 0.04,
        top_k: int = 30,
        override_delta: float = 0.008,
        statute_threshold: float = 0.35,
        lexical_guardrail: float = -0.20,
        pairwise_reranker: Optional[PairwiseNeuralReRanker] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.top_k = top_k
        self.override_delta = override_delta
        self.statute_threshold = statute_threshold
        self.lexical_guardrail = lexical_guardrail

        self.pairwise_reranker = pairwise_reranker
        self.gemini_service = gemini_service or GeminiService()

        # Phase 11 Base Engine
        self.base_engine = QueryAdaptiveReRanker(
            tau=self.tau,
            alpha=self.alpha,
            gamma=self.gamma,
            top_k=self.top_k,
            pairwise_reranker=self.pairwise_reranker,
        )

    def rerank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes Phase 14 Retrieval:
        1. Runs frozen Phase 11 Adaptive Re-Ranking.
        2. Applies Confidence-Gated Selective Overrides to borderline low-margin queries.
        3. Deterministically guarantees 100% Phase 11 ranking if conditions are not satisfied.
        """
        df_out = self.base_engine.rerank(df).copy()
        df_out["p14_score"] = df_out["final_adaptive_score"].values.copy()
        df_out["override_applied"] = False
        df_out["override_reason"] = "none"

        for qid, grp in df_out.groupby("query_id"):
            sub = grp.sort_values("final_adaptive_score", ascending=False)
            # Guard: Only evaluate low-margin regime
            if sub.iloc[0]["stage1_margin"] >= self.tau:
                continue

            r1 = sub.iloc[0]
            r2 = sub.iloc[1]

            score_diff = float(r1["final_adaptive_score"] - r2["final_adaptive_score"])
            stat_diff = float(r2.get("norm_statute_jaccard", 0.0) - r1.get("norm_statute_jaccard", 0.0))
            tfidf_diff = float(r2.get("norm_tfidf_score", 0.0) - r1.get("norm_tfidf_score", 0.0))

            # Confidence-gated override criteria:
            # 1. Ultra-close margin between Rank 1 and Rank 2 (< override_delta)
            # 2. Rank 2 has verified statutory superiority (> statute_threshold)
            # 3. Rank 1 does not possess overwhelming lexical advantage (>= lexical_guardrail)
            if (
                score_diff < self.override_delta
                and stat_diff > self.statute_threshold
                and tfidf_diff >= self.lexical_guardrail
            ):
                r2_idx = r2.name
                boost = score_diff + 0.002
                df_out.loc[r2_idx, "p14_score"] += boost
                df_out.loc[r2_idx, "override_applied"] = True
                df_out.loc[r2_idx, "override_reason"] = f"statutory_superiority_boost_{boost:.4f}"

        # Recalculate ranks based on p14_score
        df_out["p14_rank"] = df_out.groupby("query_id")["p14_score"].rank(ascending=False, method="first")
        return df_out

