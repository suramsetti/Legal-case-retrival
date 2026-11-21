"""
Gemini Service Layer for Legal Document Similarity and Precedent Retrieval.

Provides:
- Multi-Provider Support: Prioritizes OpenAI (GPT-4.1 Mini) when OPENAI_API_KEY is configured in .env,
  with transparent fallback to Google Gemini when configured.
- Resilience: Exponential backoff with rate limit (429) retry handling.
- Graceful Fallback: Returns None or baseline fallback on quota exhaustion / failure.
- Performance: Persistent disk-based caching keyed by prompt hash.
- Security: Zero key leakage; credentials remain server-side only.
- Structured JSON: Strict JSON extraction and schema validation.
- Metrics Tracking: Call count, token usage, latency (P50/P95), cache hits, failure rates.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import numpy as np

# Apply compatibility patch for protobuf >= 5 with google-generativeai
try:
    import google.protobuf.json_format as jf
    _orig_m2d = getattr(jf, "MessageToDict", None)
    if _orig_m2d:
        def _patched_m2d(*args, **kwargs):
            kwargs.pop("float_precision", None)
            return _orig_m2d(*args, **kwargs)
        jf.MessageToDict = _patched_m2d
except Exception:
    pass

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    genai = None

    class _GoogleExceptions:
        class ResourceExhausted(Exception):
            pass

    google_exceptions = _GoogleExceptions()

from backend.config.gemini_config import GeminiConfig

logger = logging.getLogger("GeminiService")


class GeminiService:
    """Manages robust communication with LLM providers (OpenAI / Google Gemini)."""

    def __init__(self, config: Optional[GeminiConfig] = None, client: Optional[Any] = None):
        self.config = config or GeminiConfig()
        self.client = client
        self.model = None
        self.openai_service = None

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

        # Check for OpenAI service first
        try:
            from backend.services.openai_service import OpenAIService
            o_svc = OpenAIService()
            if o_svc.is_available:
                self.openai_service = o_svc
                logger.info(f"GeminiService: Active provider set to OpenAI ({o_svc.config.model_name}).")
        except Exception as e:
            logger.debug(f"OpenAI service initialization check: {e}")

        # Initialize Gemini if OpenAI is not active or as secondary
        if not self.openai_service or not self.openai_service.is_available:
            self._init_client()

    def _init_client(self) -> None:
        """Initializes the GenerativeModel client if configured."""
        if self.client is not None:
            self.model = self.client
            return

        if genai is None or not self.config.is_configured:
            logger.warning("Gemini API key unconfigured. Operating in fallback mode.")
            return

        try:
            genai.configure(api_key=self.config.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                }
            )
            logger.info(f"GeminiService configured successfully with model {self.config.model_name}.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        """Checks if service has an active LLM provider (OpenAI or Gemini)."""
        return (self.openai_service is not None and self.openai_service.is_available) or self.model is not None

    @property
    def active_provider(self) -> str:
        """Returns the active provider name."""
        if self.openai_service is not None and self.openai_service.is_available:
            return f"OpenAI ({self.openai_service.config.model_name})"
        if self.model is not None:
            return f"Gemini ({self.config.model_name})"
        return "Unconfigured"

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
                    data = json.load(f)
                return data
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
        Executes an LLM call with caching, exponential retry backoff,
        and JSON structure parsing. Prioritizes OpenAI if available.
        """
        # 1. Delegate to OpenAI if configured
        if self.openai_service and self.openai_service.is_available:
            return self.openai_service.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                schema_validator=schema_validator,
                fallback_factory=fallback_factory,
            )

        self.stats["total_requests"] += 1
        prompt_hash = self._get_cache_key(prompt, system_instruction)

        # 2. Check persistent cache
        cached = self._read_cache(prompt_hash)
        if cached is not None:
            self.stats["cache_hits"] += 1
            if schema_validator is None or schema_validator(cached):
                return cached

        # 3. Check client availability
        if not self.is_available:
            logger.debug("Gemini not available. Triggering fallback.")
            return fallback_factory() if fallback_factory else None

        # 4. Call API with exponential retry & rate-limit handling
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        retries = 0
        backoff_sec = 2.0
        t0 = time.time()

        while retries <= self.config.max_retries:
            try:
                self.stats["api_calls"] += 1
                self.stats["input_tokens_est"] += len(full_prompt) // 4

                response = self.model.generate_content(full_prompt)
                latency_ms = (time.time() - t0) * 1000.0
                self.stats["latencies_ms"].append(latency_ms)

                raw_text = response.text if hasattr(response, "text") else str(response)
                self.stats["output_tokens_est"] += len(raw_text) // 4

                # Parse JSON
                cleaned_text = self.clean_json_response(raw_text)
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

            except google_exceptions.ResourceExhausted as e:
                self.stats["rate_limit_hits"] += 1
                self.stats["api_errors"] += 1
                retry_wait = backoff_sec
                err_msg = str(e)
                if "retry in" in err_msg:
                    try:
                        part = err_msg.split("retry in")[1].split("s")[0].strip()
                        retry_wait = min(float(part) + 1.0, 45.0)
                    except Exception:
                        pass
                logger.warning(f"Gemini 429 Rate Limit hit. Waiting {retry_wait:.1f}s before retry ({retries}/{self.config.max_retries})...")
                retries += 1
                if retries <= self.config.max_retries:
                    time.sleep(retry_wait)
                    backoff_sec = max(backoff_sec * 2, retry_wait)
                else:
                    logger.error(f"Exceeded max retries after rate limiting: {e}")
                    break

            except json.JSONDecodeError as e:
                self.stats["api_errors"] += 1
                logger.warning(f"Failed to parse JSON from Gemini response: {e}")
                retries += 1
                if retries <= self.config.max_retries:
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    break

            except Exception as e:
                self.stats["api_errors"] += 1
                logger.error(f"Gemini API request failed ({type(e).__name__}): {e}")
                err_str = str(e)
                if "API_KEY_INVALID" in err_str or "API key not valid" in err_str or "not found" in err_str.lower():
                    logger.warning("Non-retryable Gemini error. Skipping retries.")
                    break
                retries += 1
                if retries <= self.config.max_retries:
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    break

        # Fallback if API or parsing failed
        return fallback_factory() if fallback_factory else None

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        fallback_text: str = "",
    ) -> str:
        """Generates text completion with OpenAI priority."""
        if self.openai_service and self.openai_service.is_available:
            return self.openai_service.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                fallback_text=fallback_text,
            )

        if not self.is_available or self.model is None:
            return fallback_text

        try:
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            resp = self.model.generate_content(full_prompt)
            return resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:
            logger.error(f"Gemini generate_text error: {e}")
            return fallback_text

    def get_operational_metrics(self) -> Dict[str, Any]:
        """Returns structured performance and cost analytics."""
        if self.openai_service and self.openai_service.is_available:
            return self.openai_service.get_operational_metrics()

        lats = self.stats["latencies_ms"]
        avg_lat = float(np.mean(lats)) if lats else 0.0
        p95_lat = float(np.percentile(lats, 95)) if lats else 0.0

        cost_input = (self.stats["input_tokens_est"] / 1_000_000.0) * 0.075
        cost_output = (self.stats["output_tokens_est"] / 1_000_000.0) * 0.30
        total_cost = cost_input + cost_output

        cache_rate = (self.stats["cache_hits"] / max(self.stats["total_requests"], 1)) * 100.0
        error_rate = (self.stats["api_errors"] / max(self.stats["api_calls"], 1)) * 100.0

        return {
            "provider": "Google Gemini",
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate_pct": round(cache_rate, 2),
            "api_calls": self.stats["api_calls"],
            "api_errors": self.stats["api_errors"],
            "api_error_rate_pct": round(error_rate, 2),
            "rate_limit_hits": self.stats["rate_limit_hits"],
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "input_tokens_estimated": self.stats["input_tokens_est"],
            "output_tokens_estimated": self.stats["output_tokens_est"],
            "estimated_api_cost_usd": round(total_cost, 6),
            "model_configured": self.config.model_name,
        }
