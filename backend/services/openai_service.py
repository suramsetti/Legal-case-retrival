"""
OpenAI Service Layer for Legal Intelligence, Precedent Analysis, and Judgment Explanation.

Provides:
- Direct HTTP integration with OpenAI Chat Completions API (no heavy SDK dependency).
- Resilience: Exponential backoff with rate limit (429) retry handling.
- Structured JSON: Strict JSON mode (response_format: {"type": "json_object"}).
- Performance: Persistent disk-based caching keyed by prompt hash.
- Security: Zero key leakage; credentials remain server-side only.
- Metrics Tracking: Call count, token usage, latency (P50/P95), cache hits, failure rates.
"""

import os
import json
import time
import hashlib
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

import numpy as np

from backend.config.openai_config import OpenAIConfig

logger = logging.getLogger("OpenAIService")


class OpenAIService:
    """Manages robust communication with the OpenAI API."""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()
        self.endpoint = "https://api.openai.com/v1/chat/completions"

        # Operational metrics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "api_errors": 0,
            "rate_limit_hits": 0,
            "latencies_ms": [],
            "input_tokens_est": 0,
            "output_tokens_est": 0,
        }

        # Initialize cache directory
        if self.config.cache_enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.is_available:
            logger.info(f"OpenAIService initialized successfully with model {self.config.model_name}.")
        else:
            logger.warning("OpenAIService initialized without valid API key. Operating in fallback mode.")

    @property
    def is_available(self) -> bool:
        """Checks if service has a configured API key."""
        return self.config.is_configured

    def _get_cache_key(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Computes deterministic cache key from prompt and model name."""
        content = f"{self.config.model_name}::{system_instruction or ''}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Reads cached JSON response if exists."""
        if not self.config.cache_enabled:
            return None
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None

    def _write_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Persists JSON response to disk cache."""
        if not self.config.cache_enabled:
            return
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")

    def clean_json_response(self, text: str) -> str:
        """Strips markdown code blocks, backticks, and extraneous text to isolate raw JSON."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        schema_validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
        fallback_factory: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes an OpenAI API call with JSON mode, caching, and retry backoff.
        """
        self.stats["total_requests"] += 1
        prompt_hash = self._get_cache_key(prompt, system_instruction)

        # 1. Check persistent cache
        cached = self._read_cache(prompt_hash)
        if cached is not None:
            self.stats["cache_hits"] += 1
            if schema_validator is None or schema_validator(cached):
                return cached

        # 2. Check client availability
        if not self.is_available:
            logger.debug("OpenAI not available. Triggering fallback.")
            return fallback_factory() if fallback_factory else None

        # 3. Call API with exponential retry & rate-limit handling
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UnifiedLegalAI/1.0",
        }

        retries = 0
        backoff_sec = 2.0
        t0 = time.time()

        while retries <= self.config.max_retries:
            try:
                self.stats["api_calls"] += 1
                self.stats["input_tokens_est"] += len(prompt) // 4

                req = urllib.request.Request(self.endpoint, data=payload_bytes, headers=headers)
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    raw_body = response.read().decode("utf-8")
                    latency_ms = (time.time() - t0) * 1000.0
                    self.stats["latencies_ms"].append(latency_ms)

                    resp_json = json.loads(raw_body)
                    choices = resp_json.get("choices", [])
                    if not choices:
                        raise ValueError("OpenAI returned empty choices array.")

                    raw_content = choices[0]["message"]["content"]
                    self.stats["output_tokens_est"] += len(raw_content) // 4

                    # Parse JSON content
                    cleaned_text = self.clean_json_response(raw_content)
                    parsed = json.loads(cleaned_text)

                    # Validate schema if provided
                    if schema_validator and not schema_validator(parsed):
                        logger.warning(f"Response failed schema validation: {parsed.keys()}")
                        if retries < self.config.max_retries:
                            retries += 1
                            time.sleep(backoff_sec)
                            backoff_sec *= 2
                            continue
                        break

                    # Persist to disk cache
                    self._write_cache(prompt_hash, parsed)
                    return parsed

            except urllib.error.HTTPError as e:
                self.stats["api_errors"] += 1
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass

                logger.error(f"OpenAI HTTP error {e.code}: {err_body[:200]}")

                # Fast-break on authentication errors
                if e.code in (401, 403):
                    logger.warning("Invalid OpenAI API key or unauthorized. Skipping retries.")
                    break

                if e.code == 429:
                    self.stats["rate_limit_hits"] += 1
                    logger.warning(f"OpenAI 429 Rate Limit hit. Retrying in {backoff_sec:.1f}s...")
                    retries += 1
                    if retries <= self.config.max_retries:
                        time.sleep(backoff_sec)
                        backoff_sec *= 2
                        continue
                    break

                # Server errors (5xx)
                if e.code >= 500:
                    retries += 1
                    if retries <= self.config.max_retries:
                        time.sleep(backoff_sec)
                        backoff_sec *= 2
                        continue
                    break

                break

            except json.JSONDecodeError as e:
                self.stats["api_errors"] += 1
                logger.warning(f"Failed to parse JSON from OpenAI response: {e}")
                retries += 1
                if retries <= self.config.max_retries:
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    break

            except Exception as e:
                self.stats["api_errors"] += 1
                logger.error(f"OpenAI request failed ({type(e).__name__}): {e}")
                retries += 1
                if retries <= self.config.max_retries:
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    break

        return fallback_factory() if fallback_factory else None

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        fallback_text: str = "",
    ) -> str:
        """Executes a text completion query."""
        if not self.is_available:
            return fallback_text

        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UnifiedLegalAI/1.0",
        }

        try:
            req = urllib.request.Request(self.endpoint, data=payload_bytes, headers=headers)
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                return resp_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenAI generate_text error: {e}")
            return fallback_text

    def get_operational_metrics(self) -> Dict[str, Any]:
        """Returns structured performance and operational analytics."""
        lats = self.stats["latencies_ms"]
        p50 = float(np.percentile(lats, 50)) if lats else 0.0
        p95 = float(np.percentile(lats, 95)) if lats else 0.0
        return {
            "provider": "OpenAI",
            "model": self.config.model_name,
            "configured": self.is_available,
            "masked_key": self.config.masked_key,
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "api_calls": self.stats["api_calls"],
            "api_errors": self.stats["api_errors"],
            "rate_limit_hits": self.stats["rate_limit_hits"],
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "input_tokens_est": self.stats["input_tokens_est"],
            "output_tokens_est": self.stats["output_tokens_est"],
        }

