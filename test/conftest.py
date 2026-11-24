"""
Shared pytest configuration for the whole suite.

Only dummy credentials live here. Provider-SDK fakes are scoped to
``test/resource/conftest.py`` so ``rag_core``/``evals``/``tools`` tests never
import against a faked third-party package.
"""

import os

# Dummy credentials — must be set before src modules are imported
os.environ.setdefault("OPENAI_API_KEY", "sk-test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
