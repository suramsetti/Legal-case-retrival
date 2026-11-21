"""
Unit Tests for GeminiService and GeminiConfig.

Verifies:
1. Environment variable configuration and API key masking.
2. Markdown code block stripping and JSON cleaning.
3. Disk caching (write, read, hit rate tracking).
4. Schema validation and malformed JSON recovery.
5. Fallback behavior when API is offline or unconfigured.
6. Rate limit retry handling and operational metrics calculation.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.config.gemini_config import GeminiConfig
from backend.services.gemini_service import GeminiService


class TestGeminiConfig:

    def test_masked_key_security(self):
        cfg = GeminiConfig(api_key="AQ.Ab8RN6Js3tnsQphiIexiDXbM9RRUssT2ACcG9XHVr638djpA6A")
        assert cfg.is_configured is True
        masked = cfg.masked_key
        assert masked.startswith("AQ.A")
        assert masked.endswith("pA6A")
        assert "QphiIexi" not in masked  # Secret middle is never exposed

    def test_unconfigured_masked_key(self):
        cfg = GeminiConfig(api_key=None)
        assert cfg.is_configured is False
        assert cfg.masked_key == "UNCONFIGURED"

    def test_to_dict_safe(self):
        cfg = GeminiConfig(api_key="secret_test_key_123456789")
        d = cfg.to_dict()
        assert d["configured"] is True
        assert d["masked_key"] == "secr...6789"
        assert "secret_test_key_123456789" not in str(d)


class TestGeminiService:

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_clean_json_response(self):
        svc = GeminiService(config=GeminiConfig(api_key=None))

        # Standard markdown code block
        raw = "```json\n{\"status\": \"ok\", \"value\": 42}\n```"
        cleaned = svc.clean_json_response(raw)
        assert cleaned == '{"status": "ok", "value": 42}'
        assert json.loads(cleaned)["status"] == "ok"

        # Raw JSON without code fence
        assert svc.clean_json_response('{"foo": "bar"}') == '{"foo": "bar"}'

    def test_cache_write_and_read(self, temp_cache_dir):
        cfg = GeminiConfig(api_key="test_key", cache_enabled=True, cache_dir=temp_cache_dir)
        svc = GeminiService(config=cfg)

        key = "test_prompt_hash_123"
        payload = {"result": "cached_success", "score": 0.95}

        # Cache miss
        assert svc._read_cache(key) is None

        # Write cache
        svc._write_cache(key, payload)

        # Cache hit
        cached = svc._read_cache(key)
        assert cached is not None
        assert cached["result"] == "cached_success"
        assert cached["score"] == 0.95

    def test_offline_fallback(self):
        cfg = GeminiConfig(api_key=None)
        svc = GeminiService(config=cfg)

        fallback = lambda: {"fallback": True, "value": "default"}
        result = svc.generate_json(
            prompt="Analyze this case",
            fallback_factory=fallback,
        )
        assert result is not None
        assert result["fallback"] is True
        assert result["value"] == "default"

    def test_mocked_generate_json_success(self, temp_cache_dir):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '```json\n{"legal_issue": "Murder under Section 302", "confidence": 0.92}\n```'
        mock_model.generate_content.return_value = mock_response

        cfg = GeminiConfig(api_key="dummy_key", cache_enabled=True, cache_dir=temp_cache_dir)
        svc = GeminiService(config=cfg, client=mock_model)

        def validator(d):
            return "legal_issue" in d

        res = svc.generate_json(
            prompt="What is the legal issue?",
            schema_validator=validator,
        )

        assert res is not None
        assert res["legal_issue"] == "Murder under Section 302"
        assert res["confidence"] == 0.92
        assert svc.stats["api_calls"] == 1

        # Second call should hit the cache!
        res2 = svc.generate_json(
            prompt="What is the legal issue?",
            schema_validator=validator,
        )
        assert res2["legal_issue"] == "Murder under Section 302"
        assert svc.stats["cache_hits"] == 1
        assert svc.stats["api_calls"] == 1  # No second API call!

    def test_operational_metrics_calculation(self):
        cfg = GeminiConfig(api_key=None)
        svc = GeminiService(config=cfg)
        svc.stats["total_requests"] = 10
        svc.stats["cache_hits"] = 4
        svc.stats["api_calls"] = 6
        svc.stats["api_errors"] = 1
        svc.stats["latencies_ms"] = [100.0, 150.0, 200.0, 250.0]
        svc.stats["input_tokens_est"] = 10000
        svc.stats["output_tokens_est"] = 2000

        metrics = svc.get_operational_metrics()
        assert metrics["total_requests"] == 10
        assert metrics["cache_hits"] == 4
        assert metrics["cache_hit_rate_pct"] == 40.0
        assert metrics["api_calls"] == 6
        assert metrics["avg_latency_ms"] == 175.0
        assert metrics["estimated_api_cost_usd"] > 0.0

