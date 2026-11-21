"""Explainability and Interpretability package for Hybrid Legal Retrieval."""

from src.explainability.feature_attribution import FeatureAttributor
from src.explainability.shap_explainer import LinearFusionSHAPExplainer, LinearSHAPExplainer
from src.explainability.case_explanation import CaseExplanationGenerator
from src.explainability.ranking_comparison import RankingComparisonAnalyzer

__all__ = [
    "FeatureAttributor",
    "LinearFusionSHAPExplainer",
    "LinearSHAPExplainer",
    "CaseExplanationGenerator",
    "RankingComparisonAnalyzer",
]

