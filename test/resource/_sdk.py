"""Shared skip conditions for provider-SDK-dependent tests."""

import importlib.util

ANTHROPIC_MISSING = importlib.util.find_spec("anthropic") is None
ANTHROPIC_SKIP_REASON = "anthropic SDK not installed (pip install -r requirements.txt)"
