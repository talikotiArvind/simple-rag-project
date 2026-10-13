"""Typed, side-effect-free settings loaded from environment variables."""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from rag_core.exceptions import ConfigError

REDACTED = "***"
VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
)
MIN_TOP_K = 1
MAX_TOP_K = 100
DEFAULT_INDEX_NAME = "rag-index"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5
DEFAULT_LOG_LEVEL = "INFO"
SECRET_ENV_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "RERANK_API_KEY",
)


class Secret:
    """Holds a sensitive value; repr/str are redacted, reveal() exposes it."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """Return the raw secret value (explicit opt-in)."""
        return self._value

    def __repr__(self) -> str:
        return f"Secret({REDACTED})"

    def __str__(self) -> str:
        return REDACTED

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    index_name: str = DEFAULT_INDEX_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K
    log_level: str = DEFAULT_LOG_LEVEL
    openai_api_key: Secret | None = field(default=None, repr=False)
    anthropic_api_key: Secret | None = field(default=None, repr=False)
    rerank_api_key: Secret | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.index_name.strip():
            raise ConfigError("index_name must not be empty")
        if not self.embedding_model.strip():
            raise ConfigError("embedding_model must not be empty")
        if not MIN_TOP_K <= self.top_k <= MAX_TOP_K:
            raise ConfigError(
                f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}, got {self.top_k}"
            )
        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigError(f"unknown log_level {self.log_level!r}")

    @property
    def log_level_no(self) -> int:
        """Numeric logging level for the configured log_level."""
        return int(getattr(logging, self.log_level))

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        require_secrets: bool = False,
    ) -> Settings:
        """Build Settings from env (defaults to os.environ); never loads .env."""
        source = os.environ if env is None else env
        secrets = cls._read_secrets(source)
        if require_secrets:
            missing = [n for n in SECRET_ENV_NAMES if n not in secrets]
            if missing:
                raise ConfigError(f"missing required secrets: {', '.join(missing)}")
        return cls(
            index_name=source.get("RAG_INDEX_NAME", DEFAULT_INDEX_NAME),
            embedding_model=source.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            top_k=cls._parse_top_k(source.get("RAG_TOP_K")),
            log_level=source.get("RAG_LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper(),
            openai_api_key=secrets.get("OPENAI_API_KEY"),
            anthropic_api_key=secrets.get("ANTHROPIC_API_KEY"),
            rerank_api_key=secrets.get("RERANK_API_KEY"),
        )

    @staticmethod
    def _read_secrets(source: Mapping[str, str]) -> dict[str, Secret]:
        found: dict[str, Secret] = {}
        for name in SECRET_ENV_NAMES:
            value = source.get(name, "").strip()
            if value:
                found[name] = Secret(value)
        return found

    @staticmethod
    def _parse_top_k(raw: str | None) -> int:
        if raw is None or not raw.strip():
            return DEFAULT_TOP_K
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"RAG_TOP_K must be an integer, got {raw!r}") from exc
