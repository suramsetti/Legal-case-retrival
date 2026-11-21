"""
Explanation Service for Legal Prior Case Retrieval Web Application.

Generates case-level structured explanations, mathematical score breakdowns,
factual entity overlaps, and interpretability summaries using the Phase 7 modules.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.explainability.case_explanation import CaseExplanationGenerator
from src.explainability.feature_attribution import FeatureAttributor

logger = logging.getLogger("ExplanationService")


class ExplanationService:
    """Provides auditable legal case retrieval explanations for web queries."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        test_dir = self.project_root / "data/processed/IL-TUR-PCR"

        q_legal_path = test_dir / "test_queries" / "legal_features.json"
        c_legal_path = test_dir / "test_candidates" / "legal_features.json"

        self.case_generator = CaseExplanationGenerator(str(q_legal_path), str(c_legal_path))
        self.attributor = FeatureAttributor(
            weights=[0.00, 0.65, 0.25, 0.10],
            feature_cols=["bm25_score", "tfidf_score", "inlegalbert_sim", "composite_legal_score"],
        )

    def explain(
        self,
        query_id: str,
        case_id: str,
        pair_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates detailed factual explanation for why candidate `case_id` was retrieved for `query_id`.
        """
        explanation = self.case_generator.generate_explanation(
            query_id=query_id,
            candidate_id=case_id,
            pair_row=pair_features or {},
        )

        # Structure explanation for UI
        return {
            "query_id": query_id,
            "case_id": case_id,
            "is_ground_truth_precedent": explanation.get("is_ground_truth_precedent", False),
            "hybrid_score": explanation.get("hybrid_score", 0.0),
            "score_breakdown": explanation.get("score_breakdown", {}),
            "legal_knowledge_factors": {
                "shared_statutory_provisions": explanation["legal_features"].get("shared_statutory_provisions", []),
                "shared_precedent_citations": explanation["legal_features"].get("shared_precedent_citations", []),
                "shared_judges": explanation["legal_features"].get("shared_judges", []),
                "shared_courts": explanation["legal_features"].get("shared_courts", []),
                "rhetorical_similarity": explanation["legal_features"].get("rhetorical_similarity", 0.0),
            },
            "interpretation": explanation.get("interpretation", {
                "matching_factors": [],
                "non_matching_factors": [],
            }),
            "model_architecture": "Weighted Linear Score Fusion (0.65 TF-IDF + 0.25 InLegalBERT + 0.10 Legal Knowledge)",
            "legal_disclaimer": "Retrieval score represents textual and statutory relevance to the query description, not a guarantee of legal precedent bindingness.",
        }
