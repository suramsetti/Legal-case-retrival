"""
Unit Tests for Phase 6A: Hybrid Feature Alignment and Dataset Construction.

Validates:
- NegativeSampler: Determinism, self-exclusion, positive retention, hard negative sampling.
- FeatureNormalizer: Skew handling, reciprocal rank, zero split leakage (train-only fit).
- FeatureDatasetBuilder: Alignment across lexical, dense, and legal modalities.
- Label correctness and duplicate pair checks.
"""

import unittest
import numpy as np
import pandas as pd

from src.ranking.negative_sampler import NegativeSampler
from src.ranking.feature_normalizer import FeatureNormalizer, ALL_FEATURE_COLUMNS
from src.ranking.feature_dataset_builder import FeatureDatasetBuilder
from src.legal_features.legal_feature_encoder import DocumentLegalFeatures


class TestNegativeSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = NegativeSampler(
            n_hard_negatives=3,
            n_random_negatives=2,
            random_seed=42,
        )

    def test_positive_retention_and_exclusion(self):
        query_id = "Q1"
        positives = {"C1", "C2", "Q1"}  # Q1 is self-match
        cand_pool = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "Q1"]
        scores = np.array([10.0, 8.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 15.0])

        res = self.sampler.sample_candidates_for_query(
            query_id=query_id,
            positive_candidate_ids=positives,
            candidate_pool_ids=cand_pool,
            candidate_scores=scores,
        )

        selected = res["selected_candidate_ids"]
        labels = res["labels"]

        # 1. Self-exclusion: query_id must NEVER be in selected candidates
        self.assertNotIn("Q1", selected)

        # 2. Positive retention: C1 and C2 must both be retained
        self.assertIn("C1", selected)
        self.assertIn("C2", selected)
        self.assertEqual(res["positive_count"], 2)

        # 3. Label matching
        for cid, lbl in zip(selected, labels):
            if cid in {"C1", "C2"}:
                self.assertEqual(lbl, 1)
            else:
                self.assertEqual(lbl, 0)

        # 4. Hard negatives: top non-relevant by score are C3 (9.0), C4 (7.0), C5 (6.0)
        self.assertIn("C3", selected)
        self.assertIn("C4", selected)
        self.assertIn("C5", selected)
        self.assertEqual(res["hard_negative_count"], 3)
        self.assertEqual(res["random_negative_count"], 2)

    def test_deterministic_sampling(self):
        positives = {"C1"}
        cand_pool = [f"C{i}" for i in range(1, 20)]
        scores = np.random.RandomState(123).rand(len(cand_pool))

        res1 = self.sampler.sample_candidates_for_query("Q1", positives, cand_pool, scores)
        res2 = self.sampler.sample_candidates_for_query("Q1", positives, cand_pool, scores)

        self.assertEqual(res1["selected_candidate_ids"], res2["selected_candidate_ids"])
        self.assertEqual(res1["labels"], res2["labels"])


class TestFeatureNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = FeatureNormalizer()

    def test_fit_and_zero_leakage(self):
        # Create synthetic train and dev datasets
        np.random.seed(42)
        train_data = {col: np.random.exponential(scale=5.0, size=100) for col in ALL_FEATURE_COLUMNS}
        train_data["statute_shared_count"] = np.random.poisson(lam=3.0, size=100)
        train_data["bm25_rank"] = np.random.randint(1, 50, size=100)
        train_data["tfidf_rank"] = np.random.randint(1, 50, size=100)
        train_data["inlegalbert_rank"] = np.random.randint(1, 50, size=100)
        train_df = pd.DataFrame(train_data)

        # Dev dataset with different distribution
        dev_data = {col: np.random.exponential(scale=15.0, size=50) for col in ALL_FEATURE_COLUMNS}
        dev_data["statute_shared_count"] = np.random.poisson(lam=8.0, size=50)
        dev_data["bm25_rank"] = np.random.randint(1, 100, size=50)
        dev_data["tfidf_rank"] = np.random.randint(1, 100, size=50)
        dev_data["inlegalbert_rank"] = np.random.randint(1, 100, size=50)
        dev_df = pd.DataFrame(dev_data)

        # Fit strictly on train
        self.normalizer.fit(train_df)
        self.assertTrue(self.normalizer.fitted)

        # Transform both
        train_trans = self.normalizer.transform(train_df)
        dev_trans = self.normalizer.transform(dev_df)

        # Check raw preserved
        self.assertIn("raw_bm25_score", train_trans.columns)
        self.assertIn("norm_bm25_score", train_trans.columns)
        self.assertIn("raw_statute_shared_count", dev_trans.columns)
        self.assertIn("norm_statute_shared_count", dev_trans.columns)

        # Verify train normalized mean is approximately 0 and std is 1
        for col in ["bm25_score", "statute_shared_count", "bm25_rank"]:
            norm_col = f"norm_{col}"
            self.assertAlmostEqual(float(train_trans[norm_col].mean()), 0.0, places=2)
            self.assertAlmostEqual(float(train_trans[norm_col].std(ddof=1)), 1.0, places=2)

        # Verify dev mean is NOT 0 (since it was transformed using train stats without leakage)
        self.assertNotEqual(round(float(dev_trans["norm_bm25_score"].mean()), 2), 0.0)


class TestFeatureDatasetBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = FeatureDatasetBuilder(random_seed=42)

    def test_complete_pair_building(self):
        # Create minimal synthetic queries and candidates
        queries_df = pd.DataFrame([
            {"id": "Q1", "full_text": "murder conviction under section 302 ipc", "relevant_candidates": ["C1"]},
            {"id": "Q2", "full_text": "quashing criminal proceedings under section 482 crpc", "relevant_candidates": ["C2"]},
        ])

        candidates_df = pd.DataFrame([
            {"id": "C1", "full_text": "appeal against conviction under section 302 of the indian penal code"},
            {"id": "C2", "full_text": "application under section 482 of the code of criminal procedure"},
            {"id": "C3", "full_text": "arbitration dispute under section 11 of the arbitration act"},
        ])

        # Random normalized embeddings
        rng = np.random.RandomState(42)
        q_embs = rng.randn(2, 768).astype(np.float32)
        q_embs /= np.linalg.norm(q_embs, axis=1, keepdims=True)
        c_embs = rng.randn(3, 768).astype(np.float32)
        c_embs /= np.linalg.norm(c_embs, axis=1, keepdims=True)

        # Minimal legal features
        def dummy_feat(doc_id, stats):
            return DocumentLegalFeatures(
                doc_id=doc_id,
                statutes=[],
                citations=[],
                entities=[],
                rhetorical_distribution=[0.1] * 11,
                num_sentences=5,
                num_statutes=len(stats),
                num_citations=0,
                num_entities=1,
                statute_keys=stats,
                citation_keys=[],
                entity_keys=["COURT:SUPREME_COURT_OF_INDIA"],
            )

        q_legal = {"Q1": dummy_feat("Q1", ["IPC:302"]), "Q2": dummy_feat("Q2", ["CRPC:482"])}
        c_legal = {
            "C1": dummy_feat("C1", ["IPC:302"]),
            "C2": dummy_feat("C2", ["CRPC:482"]),
            "C3": dummy_feat("C3", ["ARBITRATION:11"]),
        }

        # Build full evaluation matrix (no negative sampling)
        df_pairs = self.builder.build_dataset_for_split(
            split_name="dev",
            queries_df=queries_df,
            candidates_df=candidates_df,
            query_embeddings=q_embs,
            candidate_embeddings=c_embs,
            query_legal_features=q_legal,
            cand_legal_features=c_legal,
            negative_sampler=None,
        )

        # Expected 2 queries x 3 candidates = 6 pairs
        self.assertEqual(len(df_pairs), 6)

        # Verify columns exist
        expected_cols = [
            "query_id", "candidate_id", "split", "label",
            "bm25_score", "bm25_rank", "tfidf_score", "tfidf_rank",
            "inlegalbert_sim", "inlegalbert_rank",
            "statute_jaccard", "citation_jaccard", "entity_jaccard",
            "composite_legal_score"
        ]
        for col in expected_cols:
            self.assertIn(col, df_pairs.columns)

        # Verify positive labels: (Q1, C1) -> 1, (Q2, C2) -> 1, others -> 0
        q1_c1 = df_pairs[(df_pairs["query_id"] == "Q1") & (df_pairs["candidate_id"] == "C1")]
        self.assertEqual(int(q1_c1["label"].iloc[0]), 1)
        self.assertGreater(float(q1_c1["statute_jaccard"].iloc[0]), 0.0)

        q1_c3 = df_pairs[(df_pairs["query_id"] == "Q1") & (df_pairs["candidate_id"] == "C3")]
        self.assertEqual(int(q1_c3["label"].iloc[0]), 0)

        # Verify no duplicate pairs
        pair_counts = df_pairs.groupby(["query_id", "candidate_id"]).size()
        self.assertTrue((pair_counts == 1).all())


if __name__ == "__main__":
    unittest.main()
