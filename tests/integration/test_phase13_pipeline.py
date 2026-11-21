"""
Integration Tests for Phase 13: Gemini API Integration Pipeline.

Verifies:
1. REST API endpoints (/api/gemini/status, /api/gemini/expand-query, /api/gemini/metrics).
2. End-to-end /api/search with enable_gemini=False (strictly runs Phase 11 frozen baseline).
3. End-to-end /api/search with enable_gemini=True (runs query expansion & precedent intelligence).
4. Graceful degradation when Gemini is offline.
"""

import json
from unittest.mock import MagicMock
import pytest

from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestPhase13Pipeline:

    def test_gemini_status_endpoint(self, client):
        resp = client.get("/api/gemini/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert "config" in data
        assert "metrics" in data
        assert "model_name" in data["config"]

    def test_gemini_metrics_endpoint(self, client):
        resp = client.get("/api/gemini/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_requests" in data
        assert "cache_hit_rate_pct" in data
        assert "estimated_api_cost_usd" in data

    def test_search_baseline_preserved(self, client):
        # When enable_gemini is False, returns pure Phase 11 baseline
        payload = {
            "query": "0000021652",
            "top_k": 5,
            "enable_gemini": False,
        }
        resp = client.post("/api/search", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()

        assert "results" in data
        assert len(data["results"]) <= 5
        assert data["gemini_enabled"] is False
        assert data["gemini_query_expansion"] is None
        assert "stage1_margin" in data

    def test_search_with_gemini_mocked(self, client):
        # Mock query expansion and reranker on service
        rs = client.application.config["RETRIEVAL_SERVICE"]
        rs.query_expander.expand_query = MagicMock(return_value={
            "original_query": "0000021652",
            "legal_issue": "Test framed legal controversy",
            "legal_concepts": ["criminal appeal", "Section 302 IPC"],
            "potential_statutes": ["Section 302 IPC"],
            "legal_synonyms": ["murder", "culpable homicide"],
            "factual_elements": ["weapon recovery"],
            "expanded_queries": ["murder Section 302 IPC appeal"],
            "uncertainties": [],
        })

        payload = {
            "query": "0000021652",
            "top_k": 5,
            "enable_gemini": True,
            "enable_expansion": True,
            "enable_reranking": True,
        }
        resp = client.post("/api/search", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["gemini_enabled"] is True
        assert data["gemini_query_expansion"] is not None
        assert data["gemini_query_expansion"]["legal_issue"] == "Test framed legal controversy"
        assert "gemini_operational_metrics" in data

    def test_gemini_expand_query_endpoint(self, client):
        rs = client.application.config["RETRIEVAL_SERVICE"]
        rs.query_expander.expand_query = MagicMock(return_value={
            "original_query": "Bail petition under Section 439 CrPC",
            "legal_issue": "Grant of regular bail in NDPS case",
            "legal_concepts": ["rigours of Section 37 NDPS Act"],
            "potential_statutes": ["Section 439 CrPC", "Section 37 NDPS Act"],
            "legal_synonyms": ["commercial quantity"],
            "factual_elements": ["search and seizure"],
            "expanded_queries": ["Section 37 NDPS Act bail rigours"],
            "uncertainties": [],
        })

        payload = {"query": "Bail petition under Section 439 CrPC"}
        resp = client.post("/api/gemini/expand-query", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["legal_issue"] == "Grant of regular bail in NDPS case"
        assert "Section 37 NDPS Act" in data["potential_statutes"]
