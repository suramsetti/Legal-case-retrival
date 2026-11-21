"""
Gemini API Configuration Module.
Handles environment variables, defaults, rate limits, and safe key masking.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Automatically load .env file from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


@dataclass
class GeminiConfig:
    """Configuration container for Google Gemini API integration."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    model_name: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    max_retries: int = field(default_factory=lambda: int(os.getenv("GEMINI_MAX_RETRIES", "3")))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30")))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")))
    temperature: float = field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.1")))
    cache_enabled: bool = field(default_factory=lambda: os.getenv("GEMINI_CACHE_ENABLED", "true").lower() in ("true", "1", "yes"))
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("GEMINI_CACHE_DIR", "data/cache/gemini"))
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("GEMINI_RPM_LIMIT", "5")))

    @property
    def is_configured(self) -> bool:
        """Returns True if a non-empty API key is set."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    @property
    def masked_key(self) -> str:
        """Returns safe masked key for logging without leaking secrets."""
        if not self.api_key:
            return "UNCONFIGURED"
        k = self.api_key.strip()
        if len(k) <= 8:
            return "****"
        return f"{k[:4]}...{k[-4:]}"

    def to_dict(self) -> dict:
        """Returns safe dictionary for diagnostic logging."""
        return {
            "configured": self.is_configured,
            "masked_key": self.masked_key,
            "model_name": self.model_name,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "cache_enabled": self.cache_enabled,
            "cache_dir": str(self.cache_dir),
            "requests_per_minute": self.requests_per_minute,
        }

