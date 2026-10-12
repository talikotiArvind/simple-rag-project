"""Exception hierarchy for rag_core."""
from __future__ import annotations


class RagCoreError(Exception):
    """Base class for all rag_core errors."""


class ValidationError(RagCoreError, ValueError):
    """Raised when a model is constructed with invalid input."""


class AuthorizationError(RagCoreError):
    """Raised when a caller is not permitted to access a tenant or resource."""


class NotFoundError(RagCoreError):
    """Raised for a missing resource or one the caller's tenant may not see (indistinguishable)."""


class ConfigError(RagCoreError):
    """Raised when required configuration is missing or invalid."""
