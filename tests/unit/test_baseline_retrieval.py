"""
Unit tests for Lexical Baseline Retrieval (BM25, TF-IDF, Aggregation, Metrics).
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.document_retriever import DocumentRetriever
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    evaluate_query_ranking,
    aggregate_retrieval_metrics,
)


class TestBaselineRetrieval(unittest.TestCase):
    """Test suite verifying retrieval baselines, aggregation, ranking, and evaluation metrics."""

    def setUp(self):
        self.corpus = [
            "the supreme court held that section 302 of the indian penal code applies to murder",
            "the high court dismissed the appeal regarding criminal negligence under section 304a",
            "bail application allowed under section 437 code of criminal procedure",
            "arbitration and conciliation act dispute regarding contractual commercial terms",
        ]
        self.doc_ids = ["doc_001", "doc_002", "doc_003", "doc_004"]

    def test_bm25_retrieval_scoring(self):
        """Tests that BM25 retrieves relevant candidate for an exact statutory query."""
        retriever = BM25Retriever(k1=1.5, b=0.75, min_df=1)
        retriever.fit(self.corpus)

        query = ["supreme court murder under section 302"]
        scores = retriever.score_queries(query).toarray()[0]

        # doc_001 has "supreme court", "section 302", "murder"
        self.assertEqual(len(scores), len(self.corpus))
        self.assertEqual(int(np.argmax(scores)), 0)
        self.assertGreater(scores[0], scores[1])

        # Test index stats
        stats = retriever.get_index_stats()
        self.assertEqual(stats["n_docs"], 4)
        self.assertTrue(stats["is_fitted"])

    def test_tfidf_retrieval_scoring(self):
        """Tests that TF-IDF cosine similarity scores the most relevant document highest."""
        retriever = TFIDFRetriever(ngram_range=(1, 1), min_df=1)
        retriever.fit(self.corpus)

        query = ["arbitration contractual commercial dispute"]
        scores = retriever.score_queries(query).toarray()[0]

        # doc_004 is about arbitration and commercial terms
        self.assertEqual(int(np.argmax(scores)), 3)
        self.assertGreater(scores[3], 0.0)

        # Cosine similarity with L2 norm should be <= 1.0 + epsilon
        self.assertLessEqual(scores[3], 1.0001)

    def test_chunk_aggregation_strategies(self):
        """Tests Max, Top-K Mean, and Top-K Sum aggregation across multiple chunks per document."""
        cand_docs = ["doc_A", "doc_B"]
        # doc_A has 3 chunks, doc_B has 2 chunks
        chunk_docs = ["doc_A", "doc_A", "doc_A", "doc_B", "doc_B"]
        chunk_scores = np.array([0.9, 0.4, 0.2, 0.7, 0.8], dtype=np.float32)

        doc_retriever = DocumentRetriever(
            candidate_doc_ids=cand_docs,
            chunk_doc_ids=chunk_docs,
        )

        # Max aggregation
        max_scores = doc_retriever.aggregate_chunk_scores(chunk_scores, strategy="max")
        # doc_A max is 0.9, doc_B max is 0.8
        self.assertAlmostEqual(max_scores[0], 0.9, places=4)
        self.assertAlmostEqual(max_scores[1], 0.8, places=4)

        # Top-2 Mean aggregation
        top2_mean = doc_retriever.aggregate_chunk_scores(chunk_scores, strategy="top_k_mean", top_k=2)
        # doc_A top-2: (0.9 + 0.4)/2 = 0.65; doc_B top-2: (0.8 + 0.7)/2 = 0.75
        self.assertAlmostEqual(top2_mean[0], 0.65, places=4)
        self.assertAlmostEqual(top2_mean[1], 0.75, places=4)

        # Top-2 Sum aggregation
        top2_sum = doc_retriever.aggregate_chunk_scores(chunk_scores, strategy="top_k_sum", top_k=2)
        self.assertAlmostEqual(top2_sum[0], 1.3, places=4)
        self.assertAlmostEqual(top2_sum[1], 1.5, places=4)

    def test_query_self_exclusion(self):
        """Tests that a query case ID is strictly masked out from its candidate rankings."""
        cand_docs = ["case_101", "case_102", "case_103", "case_104"]
        # Dummy scores where case_101 has the highest score
        scores = np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32)

        doc_retriever = DocumentRetriever(candidate_doc_ids=cand_docs)

        # Normal ranking without self-exclusion
        ranked_without = doc_retriever.rank_candidates(scores, query_id="case_999", top_n=3)
        self.assertEqual(ranked_without[0][0], "case_101")

        # Ranking WITH query self-exclusion for query_id="case_101"
        ranked_with_exclusion = doc_retriever.rank_candidates(scores, query_id="case_101", top_n=3)
        retrieved_ids = [doc_id for doc_id, _ in ranked_with_exclusion]

        self.assertNotIn("case_101", retrieved_ids)
        self.assertEqual(retrieved_ids[0], "case_102")
        self.assertEqual(len(retrieved_ids), 3)

    def test_retrieval_metrics(self):
        """Validates Precision, Recall, MRR, and nDCG calculations against known ground truth."""
        retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
        relevant = {"doc_B", "doc_D"}  # Ranks 2 and 4 (1-indexed)

        # Precision@K
        self.assertEqual(precision_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(precision_at_k(retrieved, relevant, 2), 0.5)  # 1 hit in 2
        self.assertEqual(precision_at_k(retrieved, relevant, 5), 0.4)  # 2 hits in 5

        # Recall@K
        self.assertEqual(recall_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.5)  # 1 of 2 relevant
        self.assertEqual(recall_at_k(retrieved, relevant, 5), 1.0)  # 2 of 2 relevant

        # MRR (first relevant is doc_B at rank 2 -> 1/2 = 0.5)
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.5)

        # nDCG
        # DCG@2 = 0/log2(2) + 1/log2(3) = 1/1.58496 = 0.6309
        # IDCG@2 = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # nDCG@2 = 0.6309 / 1.6309 = 0.3868
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 2), 0.3868, places=3)

        # Full query evaluation
        eval_dict = evaluate_query_ranking(retrieved, relevant)
        self.assertEqual(eval_dict["P@1"], 0.0)
        self.assertEqual(eval_dict["MRR"], 0.5)
        self.assertEqual(eval_dict["R@5"], 1.0)

    def test_split_isolation(self):
        """Verifies query partitions across train, dev, and test remain strictly disjoint."""
        processed_dir = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR"
        if (processed_dir / "dev_queries" / "documents.parquet").exists():
            import pandas as pd
            dev_queries = set(pd.read_parquet(processed_dir / "dev_queries" / "documents.parquet")["id"])
            test_queries = set(pd.read_parquet(processed_dir / "test_queries" / "documents.parquet")["id"])
            train_queries = set(pd.read_parquet(processed_dir / "train_queries" / "documents.parquet")["id"])

            self.assertEqual(len(dev_queries), 118)
            self.assertEqual(len(test_queries), 237)
            self.assertEqual(len(train_queries), 827)
            self.assertEqual(len(dev_queries.intersection(train_queries)), 0)
            self.assertEqual(len(test_queries.intersection(train_queries)), 0)
            self.assertEqual(len(dev_queries.intersection(test_queries)), 0)


if __name__ == "__main__":
    unittest.main()
