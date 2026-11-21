"""
Phase 15: Generalization-Optimized Legal Prior Case Reranker.

Key Optimizations:
1. Calibrated Multi-Modal Fusion:
   Stage 1 weights: 0.72 TF-IDF + 0.15 InLegalBERT Dense + 0.13 Composite Legal Knowledge.
   Tuned strictly on Development data to prevent over-reliance on any single modality.
2. Robust Low-Margin Threshold:
   tau = 0.065 to balance neural refinement without unnecessarily perturbing confident predictions.
3. Dual-Guardrail Confidence-Gated Selective Overrides:
   - score_diff < 0.008 (ultra-close borderline candidates)
   - stat_diff > 0.35 (statutory superiority)
   - tfidf_diff >= -0.20 (lexical preservation guardrail)
4. Deterministic 100% Fallback to Phase 14 / Phase 11 baseline.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.ranking.adaptive_reranker import QueryAdaptiveReRanker
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.ranking.phase14_reranker import Phase14SelectiveReRanker


class Phase15CalibratedAdaptiveReRanker(QueryAdaptiveReRanker):
    """
    Subclass of QueryAdaptiveReRanker that uses calibrated multi-modal Stage 1 weights:
    0.72 TF-IDF + 0.15 InLegalBERT + 0.13 Composite Legal Knowledge.
    """

    @classmethod
    def compute_stage1_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "stage1_score" not in df.columns or "qnorm_tfidf_score" not in df.columns:
            for col in ["tfidf_score", "inlegalbert_sim", "composite_legal_score", "bm25_score"]:
                s_min = df.groupby("query_id")[col].transform("min")
                s_max = df.groupby("query_id")[col].transform("max")
                diff = s_max - s_min
                df["qnorm_" + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

            df["stage1_score"] = (
                0.72 * df["qnorm_tfidf_score"]
                + 0.15 * df["qnorm_inlegalbert_sim"]
                + 0.13 * df["qnorm_composite_legal_score"]
            )

        if "s1_rank" not in df.columns:
            df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

        return df


class Phase15GeneralizationReRanker(Phase14SelectiveReRanker):
    """
    Phase 15 Generalization-Optimized Engine.
    Inherits all Phase 14 Confidence-Gated Overrides while utilizing
    Phase 15 Calibrated Adaptive Multi-Modal Routing.
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
        pairwise_reranker: Optional[PairwiseNeuralReRanker] = None,
        gemini_service=None,
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

        # Use Calibrated Stage 1 Adaptive Base Engine
        self.base_engine = Phase15CalibratedAdaptiveReRanker(
            tau=self.tau,
            alpha=self.alpha,
            gamma=self.gamma,
            top_k=self.top_k,
            pairwise_reranker=self.pairwise_reranker,
        )

