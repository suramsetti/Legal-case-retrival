"""
Unit tests for Dense Semantic Retrieval (Pooling, ModelLoader, DenseRetriever).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.pooling import masked_mean_pooling, l2_normalize
from src.embeddings.model_loader import ModelLoader, SUPPORTED_MODELS
from src.retrieval.dense_retriever import DenseRetriever


class TestDenseRetrieval(unittest.TestCase):
    """Test suite validating masked mean pooling, normalization, and dense retrieval."""

    def test_masked_mean_pooling(self):
        """Verifies masked mean pooling ignores padded tokens."""
        # Batch of 1 sequence, length 3, hidden_dim 2
        # Tokens: [1.0, 2.0], [3.0, 4.0], [100.0, 200.0] (where token 3 is padding)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]], dtype=torch.float32)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

        pooled = masked_mean_pooling(hidden, mask)
        # Expected: mean of [1, 2] and [3, 4] -> [2.0, 3.0]
        self.assertEqual(pooled.shape, (1, 2))
        self.assertAlmostEqual(pooled[0, 0].item(), 2.0, places=4)
        self.assertAlmostEqual(pooled[0, 1].item(), 3.0, places=4)

    def test_l2_normalize(self):
        """Verifies L2 normalization for PyTorch and NumPy."""
        # PyTorch
        t_vec = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        t_normed = l2_normalize(t_vec)
        self.assertAlmostEqual(torch.norm(t_normed, p=2, dim=-1).item(), 1.0, places=5)
        self.assertAlmostEqual(t_normed[0, 0].item(), 0.6, places=4)
        self.assertAlmostEqual(t_normed[0, 1].item(), 0.8, places=4)

        # NumPy
        np_vec = np.array([[3.0, 4.0]], dtype=np.float32)
        np_normed = l2_normalize(np_vec)
        self.assertAlmostEqual(float(np.linalg.norm(np_normed, axis=-1)[0]), 1.0, places=5)
        self.assertAlmostEqual(float(np_normed[0, 0]), 0.6, places=4)
        self.assertAlmostEqual(float(np_normed[0, 1]), 0.8, places=4)

    def test_model_loader_metadata(self):
        """Verifies model metadata resolution for supported legal encoders."""
        self.assertIn("inlegalbert", SUPPORTED_MODELS)
        self.assertIn("bert-base-uncased", SUPPORTED_MODELS)

        inlegal_meta = ModelLoader.get_model_metadata("inlegalbert")
        self.assertEqual(inlegal_meta["hf_id"], "law-ai/InLegalBERT")
        self.assertEqual(inlegal_meta["hidden_size"], 768)
        self.assertEqual(inlegal_meta["max_length"], 512)

    def test_dense_retriever_ranking_and_self_exclusion(self):
        """Verifies dense cosine similarity scoring, aggregation, and query self-exclusion."""
        # 3 candidate documents: doc_1 has 2 chunks, doc_2 has 1 chunk, doc_3 has 1 chunk
        # Dimensions: 4
        # doc_1 chunks: [1, 0, 0, 0], [0.8, 0.6, 0, 0]
        # doc_2 chunk: [0, 1, 0, 0]
        # doc_3 chunk: [0, 0, 1, 0]
        cand_embs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        chunk_doc_ids = ["doc_1", "doc_1", "doc_2", "doc_3"]
        cand_doc_ids = ["doc_1", "doc_2", "doc_3"]

        retriever = DenseRetriever(
            candidate_embeddings=cand_embs,
            candidate_chunk_doc_ids=chunk_doc_ids,
            candidate_doc_ids=cand_doc_ids,
        )

        # Query aligned with [1, 0, 0, 0] (most similar to doc_1)
        q_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Rank without self-exclusion
        rank_no_ex = retriever.retrieve_for_query(q_emb, query_id=None, aggregation="max")
        self.assertEqual(rank_no_ex[0][0], "doc_1")
        self.assertAlmostEqual(rank_no_ex[0][1], 1.0, places=4)

        # Rank with self-exclusion for query_id="doc_1"
        rank_with_ex = retriever.retrieve_for_query(q_emb, query_id="doc_1", aggregation="max")
        retrieved_ids = [d for d, _ in rank_with_ex]
        self.assertNotIn("doc_1", retrieved_ids)
        self.assertEqual(len(retrieved_ids), 2)  # Only doc_2 and doc_3 remaining


if __name__ == "__main__":
    unittest.main()

