"""
Integration tests for Flask Web API endpoints.
Tests search, case inspection, explainability, metrics, and health endpoints.
"""

import json
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app


class TestWebAPI(unittest.TestCase):
    """Integration test suite for the Legal Retrieval REST API."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_health_endpoint(self):
        """Tests /api/health endpoint returns healthy status and metadata."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models", data)
        self.assertGreater(data["candidate_pool_size"], 0)

    def test_sample_queries_endpoint(self):
        """Tests /api/sample-queries returns benchmark test sample cases."""
        resp = self.client.get("/api/sample-queries")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("sample_queries", data)
        self.assertGreaterEqual(len(data["sample_queries"]), 1)

    def test_metrics_endpoint(self):
        """Tests /api/metrics serves official evaluation metrics."""
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("model_comparison", data)
        self.assertIn("ablation_study", data)
        self.assertIn("bootstrap_significance", data)

    def test_benchmark_search_and_self_exclusion(self):
        """Tests /api/search with benchmark query ID and verifies self-exclusion."""
        sample_qid = "0000021652"
        payload = {"query": sample_qid, "top_k": 5}
        resp = self.client.post("/api/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["query_type"], "benchmark_query")
        self.assertEqual(data["query_id"], sample_qid)
        self.assertEqual(len(data["results"]), 5)

        # Strict self-exclusion check: query ID must not appear in retrieved candidates
        retrieved_ids = [r["case_id"] for r in data["results"]]
        self.assertNotIn(sample_qid, retrieved_ids)

        # Check result structure
        top_cand = data["results"][0]
        self.assertIn("rank", top_cand)
        self.assertIn("case_id", top_cand)
        self.assertIn("hybrid_score", top_cand)
        self.assertIn("score_breakdown", top_cand)
        self.assertIn("tfidf_contrib", top_cand["score_breakdown"])
        self.assertIn("dense_contrib", top_cand["score_breakdown"])
        self.assertIn("legal_contrib", top_cand["score_breakdown"])

    def test_free_text_search(self):
        """Tests /api/search with user free-text legal description."""
        query_text = "Section 306 IPC abetment of suicide and dying declaration"
        payload = {"query": query_text, "top_k": 5}
        resp = self.client.post("/api/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["query_type"], "free_text")
        self.assertGreaterEqual(len(data["results"]), 1)
        self.assertIn("IPC:306", data["query_statutes"])

    def test_empty_search_error_handling(self):
        """Tests /api/search returns 400 Bad Request when query is empty."""
        resp = self.client.post("/api/search", json={"query": "   "})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_case_details_endpoint(self):
        """Tests /api/case/<case_id> returns case metadata and text."""
        # Query for sample case
        sample_cid = "0000590455"
        resp = self.client.get(f"/api/case/{sample_cid}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["case_id"], sample_cid)
        self.assertIn("full_text", data)
        self.assertIn("court", data)
        self.assertIn("statutory_provisions", data)

    def test_nonexistent_case_returns_404(self):
        """Tests /api/case/<case_id> returns 404 for invalid case ID."""
        resp = self.client.get("/api/case/non_existent_99999999")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_explanation_endpoint(self):
        """Tests /api/explanation/<qid>/<cid> returns structured explanation."""
        qid = "0000021652"
        cid = "0000590455"
        resp = self.client.get(f"/api/explanation/{qid}/{cid}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["query_id"], qid)
        self.assertEqual(data["case_id"], cid)
        self.assertIn("hybrid_score", data)
        self.assertIn("score_breakdown", data)
        self.assertIn("legal_knowledge_factors", data)
        self.assertIn("interpretation", data)


if __name__ == "__main__":
    unittest.main()

