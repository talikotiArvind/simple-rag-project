"""
Provider-SDK fakes for the ``src/resource`` base-agent tests.

The OpenAI ``agents`` SDK and ``google.generativeai`` are replaced with
MagicMocks only when they are not installed, so these tests collect on a
clean install. A real installed package is never shadowed. ``anthropic`` is
not faked: Claude tests skip with a reason when it is missing (see
``ANTHROPIC_MISSING``).
"""

import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _is_installed(module_name: str) -> bool:
    """Return True when ``module_name`` can be imported from site-packages."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


if not _is_installed("agents") and "agents" not in sys.modules:
    sys.modules["agents"] = MagicMock()

if not _is_installed("google.generativeai") and "google.generativeai" not in sys.modules:
    # Fake only the generativeai submodule; a namespace stub stands in for
    # ``google`` solely when no real ``google`` package exists.
    if "google" not in sys.modules and not _is_installed("google"):
        sys.modules["google"] = ModuleType("google")
    fake_genai = MagicMock()
    sys.modules["google.generativeai"] = fake_genai
    sys.modules["google.generativeai.protos"] = fake_genai.protos
    sys.modules["google"].generativeai = fake_genai  # type: ignore[attr-defined]
