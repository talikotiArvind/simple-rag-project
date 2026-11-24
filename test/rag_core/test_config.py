"""Tests for rag_core/config.py and the .env.example / .gitignore secret hygiene."""
from __future__ import annotations

import dataclasses
import importlib
import io
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_core.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TOP_K,
    SECRET_ENV_NAMES,
    Secret,
    Settings,
)
from rag_core.exceptions import ConfigError

SENTINEL = "sk-test-SENTINEL"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _full_env(**overrides: str) -> dict[str, str]:
    env = {
        "OPENAI_API_KEY": SENTINEL + "-openai",
        "ANTHROPIC_API_KEY": SENTINEL + "-anthropic",
        "RERANK_API_KEY": SENTINEL + "-rerank",
    }
    env.update(overrides)
    return env


def _captured_log(*calls: tuple[str, tuple[object, ...]]) -> str:
    stream = io.StringIO()
    logger = logging.getLogger("test_config.capture")
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        for fmt, args in calls:
            logger.info(fmt, *args)
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


class TestSecret(unittest.TestCase):
    def setUp(self) -> None:
        self.secret = Secret(SENTINEL)

    def test_repr_does_not_reveal_value(self) -> None:
        self.assertNotIn(SENTINEL, repr(self.secret))

    def test_str_does_not_reveal_value(self) -> None:
        self.assertNotIn(SENTINEL, str(self.secret))

    def test_format_and_fstring_do_not_reveal_value(self) -> None:
        self.assertNotIn(SENTINEL, f"{self.secret}")
        self.assertNotIn(SENTINEL, "{}".format(self.secret))  # noqa: UP032
        self.assertNotIn(SENTINEL, f"{self.secret!r}")
        self.assertNotIn(SENTINEL, "%s %r" % (self.secret, self.secret))  # noqa: UP031

    def test_reveal_returns_value(self) -> None:
        self.assertEqual(self.secret.reveal(), SENTINEL)

    def test_container_repr_does_not_reveal_value(self) -> None:
        self.assertNotIn(SENTINEL, repr([self.secret]))
        self.assertNotIn(SENTINEL, repr({"k": self.secret}))

    def test_logging_does_not_reveal_value(self) -> None:
        out = _captured_log(("key=%s repr=%r", (self.secret, self.secret)))
        self.assertNotIn(SENTINEL, out)
        self.assertIn("key=", out)


class TestSettingsRedaction(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_env(_full_env(), require_secrets=True)

    def test_repr_and_str_do_not_leak(self) -> None:
        self.assertNotIn(SENTINEL, repr(self.settings))
        self.assertNotIn(SENTINEL, str(self.settings))
        self.assertNotIn(SENTINEL, f"{self.settings}")

    def test_asdict_display_does_not_leak(self) -> None:
        self.assertNotIn(SENTINEL, repr(dataclasses.asdict(self.settings)))
        self.assertNotIn(SENTINEL, str(dataclasses.asdict(self.settings)))

    def test_vars_display_does_not_leak(self) -> None:
        self.assertNotIn(SENTINEL, repr(vars(self.settings)))

    def test_logging_does_not_leak(self) -> None:
        out = _captured_log(
            ("settings=%s %r", (self.settings, self.settings)),
            ("dict=%s", (dataclasses.asdict(self.settings),)),
        )
        self.assertNotIn(SENTINEL, out)

    def test_secret_values_are_accessible_via_reveal(self) -> None:
        assert self.settings.openai_api_key is not None
        self.assertEqual(self.settings.openai_api_key.reveal(), SENTINEL + "-openai")

    def test_settings_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.settings.top_k = 7  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.settings.openai_api_key = None  # type: ignore[misc]


class TestFromEnv(unittest.TestCase):
    def test_defaults_with_empty_mapping(self) -> None:
        s = Settings.from_env({})
        self.assertEqual(s.index_name, DEFAULT_INDEX_NAME)
        self.assertEqual(s.embedding_model, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(s.top_k, DEFAULT_TOP_K)
        self.assertEqual(s.log_level, DEFAULT_LOG_LEVEL)
        self.assertIsNone(s.openai_api_key)
        self.assertIsNone(s.anthropic_api_key)
        self.assertIsNone(s.rerank_api_key)

    def test_explicit_mapping_values_are_used(self) -> None:
        s = Settings.from_env(
            {
                "RAG_INDEX_NAME": "my-index",
                "RAG_EMBEDDING_MODEL": "fake-model",
                "RAG_TOP_K": "12",
                "RAG_LOG_LEVEL": "DEBUG",
            }
        )
        self.assertEqual(
            (s.index_name, s.embedding_model, s.top_k, s.log_level),
            ("my-index", "fake-model", 12, "DEBUG"),
        )

    def test_explicit_mapping_ignores_real_environ(self) -> None:
        with patch.dict(os.environ, {"RAG_INDEX_NAME": "from-os", "OPENAI_API_KEY": "x"}):
            s = Settings.from_env({})
        self.assertEqual(s.index_name, DEFAULT_INDEX_NAME)
        self.assertIsNone(s.openai_api_key)

    def test_reads_os_environ_when_no_mapping_given(self) -> None:
        env = {"RAG_INDEX_NAME": "from-os", "RAG_TOP_K": "9", "OPENAI_API_KEY": SENTINEL}
        with patch.dict(os.environ, env, clear=True):
            s = Settings.from_env()
        self.assertEqual(s.index_name, "from-os")
        self.assertEqual(s.top_k, 9)
        assert s.openai_api_key is not None
        self.assertEqual(s.openai_api_key.reveal(), SENTINEL)

    def test_blank_top_k_uses_default(self) -> None:
        self.assertEqual(Settings.from_env({"RAG_TOP_K": "  "}).top_k, DEFAULT_TOP_K)

    def test_top_k_bounds_accepted(self) -> None:
        self.assertEqual(Settings.from_env({"RAG_TOP_K": "1"}).top_k, 1)
        self.assertEqual(Settings.from_env({"RAG_TOP_K": "100"}).top_k, 100)

    def test_top_k_out_of_bounds_rejected(self) -> None:
        for raw in ("0", "-1", "101", "100000"):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                Settings.from_env({"RAG_TOP_K": raw})

    def test_top_k_non_integer_rejected(self) -> None:
        for raw in ("abc", "5.5", "1e2", "five"):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                Settings.from_env({"RAG_TOP_K": raw})

    def test_unknown_log_level_rejected(self) -> None:
        for raw in ("VERBOSE", "TRACE", "nonsense"):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                Settings.from_env({"RAG_LOG_LEVEL": raw})

    def test_lowercase_log_level_normalised(self) -> None:
        s = Settings.from_env({"RAG_LOG_LEVEL": " debug "})
        self.assertEqual(s.log_level, "DEBUG")
        self.assertEqual(s.log_level_no, logging.DEBUG)

    def test_empty_index_name_rejected(self) -> None:
        for raw in ("", "   "):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                Settings.from_env({"RAG_INDEX_NAME": raw})

    def test_empty_embedding_model_rejected(self) -> None:
        for raw in ("", "   "):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                Settings.from_env({"RAG_EMBEDDING_MODEL": raw})

    def test_direct_construction_validates(self) -> None:
        with self.assertRaises(ConfigError):
            Settings(top_k=0)
        with self.assertRaises(ConfigError):
            Settings(log_level="NOPE")
        with self.assertRaises(ConfigError):
            Settings(index_name="")


class TestSecretsRequirement(unittest.TestCase):
    def test_missing_secrets_ok_when_not_required(self) -> None:
        s = Settings.from_env({}, require_secrets=False)
        self.assertIsNone(s.openai_api_key)

    def test_require_secrets_defaults_to_false(self) -> None:
        self.assertIsInstance(Settings.from_env({}), Settings)

    def test_missing_secrets_raise_naming_variables(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env({}, require_secrets=True)
        for name in SECRET_ENV_NAMES:
            self.assertIn(name, str(ctx.exception))

    def test_error_names_only_missing_and_never_values(self) -> None:
        env = {"OPENAI_API_KEY": SENTINEL, "ANTHROPIC_API_KEY": SENTINEL + "-2"}
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env(env, require_secrets=True)
        message = str(ctx.exception)
        self.assertIn("RERANK_API_KEY", message)
        self.assertNotIn("OPENAI_API_KEY", message)
        self.assertNotIn("ANTHROPIC_API_KEY", message)
        self.assertNotIn(SENTINEL, message)

    def test_blank_secret_counts_as_missing(self) -> None:
        env = _full_env(OPENAI_API_KEY="   ")
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env(env, require_secrets=True)
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_all_secrets_present_succeeds(self) -> None:
        s = Settings.from_env(_full_env(), require_secrets=True)
        self.assertIsInstance(s.rerank_api_key, Secret)


class TestNoImportTimeSideEffects(unittest.TestCase):
    def test_reimport_does_not_call_load_dotenv(self) -> None:
        import rag_core.config as cfg

        with patch("dotenv.load_dotenv") as mock_load:
            importlib.reload(cfg)
        mock_load.assert_not_called()

    def test_reimport_does_not_open_files(self) -> None:
        import rag_core.config as cfg

        with patch("builtins.open") as mock_open:
            importlib.reload(cfg)
        mock_open.assert_not_called()

    def test_fresh_import_does_not_load_dotenv_module(self) -> None:
        code = "import sys, rag_core.config; sys.exit(1 if 'dotenv' in sys.modules else 0)"
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_from_env_does_not_call_load_dotenv(self) -> None:
        with patch("dotenv.load_dotenv") as mock_load:
            Settings.from_env({})
        mock_load.assert_not_called()


class TestEnvExampleAndGitignore(unittest.TestCase):
    def _env_example_values(self) -> dict[str, str]:
        text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        parsed: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            parsed[name.strip()] = value.strip()
        return parsed

    def _gitignore_lines(self) -> set[str]:
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        return {ln.strip() for ln in text.splitlines()}

    def test_env_example_exists(self) -> None:
        self.assertTrue((REPO_ROOT / ".env.example").is_file())

    def test_env_example_lists_secret_names(self) -> None:
        parsed = self._env_example_values()
        for name in SECRET_ENV_NAMES:
            self.assertIn(name, parsed)

    def test_env_example_secret_values_are_empty(self) -> None:
        parsed = self._env_example_values()
        for name in SECRET_ENV_NAMES:
            self.assertEqual(parsed[name], "", name)

    def test_env_example_has_no_real_looking_values(self) -> None:
        text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("=sk-", text)
        self.assertNotIn("sk-ant-", text)

    def test_env_is_gitignored(self) -> None:
        lines = self._gitignore_lines()
        self.assertTrue({".env", "/.env", ".env*"} & lines)
        self.assertNotIn("!.env", lines)

    def test_env_example_is_not_gitignored(self) -> None:
        self.assertNotIn(".env.example", self._gitignore_lines())


if __name__ == "__main__":
    unittest.main()
