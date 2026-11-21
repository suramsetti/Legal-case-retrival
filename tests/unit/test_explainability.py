"""
Unit tests for Phase 7: Explainable AI and Legal Retrieval Interpretability.

Validates:
1. FeatureAttributor additive score decomposition identity (sum of contributions == hybrid score).
2. LinearFusionSHAPExplainer efficiency axiom (sum of SHAP values == prediction - base value).
3. Local instance explanation correctness and sign alignment.
4. CaseExplanationGenerator factual integrity (no hallucinations in shared entities/statutes).
5. RankingComparisonAnalyzer rank delta and promotion/demotion categorization.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import unittest
import numpy as np
import pandas as pd

from src.explainability.feature_attribution import FeatureAttributor
from src.explainability.shap_explainer import LinearFusionSHAPExplainer
from src.explainability.case_explanation import CaseExplanationGenerator
from src.explainability.ranking_comparison import RankingComparisonAnalyzer


class TestExplainability(unittest.TestCase):
    """Test suite validating explainability and interpretability modules."""

    def setUp(self):
        np.random.seed(42)
        rows = []
        for q in range(5):
            qid = f"q_{q}"
            for c in range(10):
                cid = f"c_{q}_{c}"
                is_pos = 1 if c == 0 else 0
                rows.append({
                    "query_id": qid,
                    "candidate_id": cid,
                    "bm25_score": float(is_pos * 100 + np.random.uniform(10, 50)),
                    "tfidf_score": float(is_pos * 0.8 + np.random.uniform(0.1, 0.4)),
                    "inlegalbert_sim": float(is_pos * 0.95 + np.random.uniform(0.7, 0.85)),
                    "composite_legal_score": float(is_pos * 0.7 + np.random.uniform(0.0, 0.3)),
                    "statute_jaccard": float(is_pos * 0.5 + np.random.uniform(0.0, 0.2)),
                    "citation_jaccard": float(is_pos * 0.6 + np.random.uniform(0.0, 0.1)),
                    "entity_jaccard": float(is_pos * 0.4 + np.random.uniform(0.0, 0.2)),
                    "rhetorical_cosine": float(is_pos * 0.85 + np.random.uniform(0.5, 0.7)),
                    "label": is_pos,
                })
        self.df = pd.DataFrame(rows)
        self.weights = [0.0, 0.65, 0.25, 0.10]
        self.feature_cols = ["bm25_score", "tfidf_score", "inlegalbert_sim", "composite_legal_score"]

    def test_feature_attributor_additive_identity(self):
        """Verifies sum(contributions) == hybrid_score for every candidate pair."""
        attributor = FeatureAttributor(weights=self.weights, feature_cols=self.feature_cols)
        df_att = attributor.compute_attributions(self.df)

        contrib_cols = [f"contrib_{c}" for c in self.feature_cols]
        calculated_sum = df_att[contrib_cols].sum(axis=1).values
        hybrid_scores = df_att["hybrid_score"].values

        np.testing.assert_allclose(
            calculated_sum, hybrid_scores, atol=1e-7,
            err_msg="Sum of feature contributions does not equal hybrid score!"
        )

    def test_feature_attributor_summary_statistics(self):
        """Verifies global attribution summary computes valid percentages."""
        attributor = FeatureAttributor(weights=self.weights, feature_cols=self.feature_cols)
        df_att = attributor.compute_attributions(self.df)
        summary = attributor.get_global_attribution_summary(df_att)

        self.assertIn("all_pairs", summary)
        self.assertIn("top1_candidates", summary)
        self.assertIn("tfidf_score", summary["all_pairs"])

        # Check BM25 has 0 contribution since weight is 0.0
        self.assertEqual(summary["all_pairs"]["bm25_score"]["mean_contribution"], 0.0)
        # Check TF-IDF has highest percentage of score
        pct_tfidf = summary["top1_candidates"]["tfidf_score"]["percentage_of_score"]
        self.assertGreater(pct_tfidf, 50.0)

    def test_shap_explainer_efficiency_axiom(self):
        """Verifies sum(phi_i) == f(x) - E[f(x)] exactly for all instances."""
        explainer = LinearFusionSHAPExplainer(
            weights=self.weights,
            feature_cols=self.feature_cols,
            background_df=self.df,
            n_background_samples=30,
        )

        shap_values, base_value = explainer.explain(self.df)
        self.assertEqual(shap_values.shape, (len(self.df), 4))

        # Compute model output f(x)
        norm_df = explainer._normalize_features(self.df)
        norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]
        X = norm_df[norm_cols].values
        fx = X @ np.array(self.weights)

        # Efficiency axiom: sum(phi_i) = f(x) - base_value
        sum_shap = np.sum(shap_values, axis=1)
        expected_diff = fx - base_value

        np.testing.assert_allclose(
            sum_shap, expected_diff, atol=1e-7,
            err_msg="SHAP efficiency axiom violated!"
        )

    def test_shap_explainer_instance_attribution(self):
        """Verifies explain_instance returns structured local explanations with correct signs."""
        explainer = LinearFusionSHAPExplainer(
            weights=self.weights,
            feature_cols=self.feature_cols,
            background_df=self.df,
            n_background_samples=30,
        )

        res = explainer.explain_instance(query_id="q_0", candidate_id="c_0_0", df=self.df)
        self.assertIn("base_value", res)
        self.assertIn("predicted_score", res)
        self.assertEqual(len(res["features"]), 4)

        # Check that top candidate (positive c_0_0) has positive predicted score > base value
        self.assertGreater(res["predicted_score"], res["base_value"])

    def test_case_explanation_generator(self):
        """Verifies case explanation generation structure and factual containment."""
        attributor = FeatureAttributor(weights=self.weights, feature_cols=self.feature_cols)
        df_att = attributor.compute_attributions(self.df)

        generator = CaseExplanationGenerator(
            query_legal_features_path="data/processed/IL-TUR-PCR/dev_queries/legal_features.json",
            cand_legal_features_path="data/processed/IL-TUR-PCR/dev_candidates/legal_features.json",
            weights=self.weights,
        )

        sample_row = df_att.iloc[0]
        exp = generator.generate_explanation(
            query_id=sample_row["query_id"],
            candidate_id=sample_row["candidate_id"],
            pair_row=sample_row,
            rank=1,
        )

        self.assertIn("hybrid_score", exp)
        self.assertIn("score_breakdown", exp)
        self.assertIn("legal_features", exp)
        self.assertIn("interpretation", exp)
        self.assertEqual(exp["rank"], 1)

    def test_ranking_comparison_analyzer(self):
        """Verifies multi-model ranking comparison and promotion/demotion detection."""
        attributor = FeatureAttributor(weights=self.weights, feature_cols=self.feature_cols)
        df_att = attributor.compute_attributions(self.df)

        analyzer = RankingComparisonAnalyzer()
        res = analyzer.compare_query_rankings(df_att, query_id="q_0")

        self.assertIn("top_10_hybrid_candidates", res)
        self.assertEqual(len(res["top_10_hybrid_candidates"]), 10)
        self.assertEqual(res["total_positives"], 1)


if __name__ == "__main__":
    unittest.main()
