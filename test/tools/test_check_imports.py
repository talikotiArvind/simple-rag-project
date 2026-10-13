"""Tests for tools/check_imports.py (D-005 import-boundary checker)."""
from __future__ import annotations

import io
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools.check_imports import main

REPO_ROOT = Path(__file__).resolve().parents[2]
VIOLATION_LINE = re.compile(r"^\S+\.py:\d+ \S")


class CheckerTestCase(unittest.TestCase):
    """Builds a fake repo root with a rag_core/ tree and runs the CLI on it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "rag_core").mkdir()

    def write(self, rel: str, source: str) -> None:
        path = self.root / "rag_core" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def reset(self) -> None:
        shutil.rmtree(self.root / "rag_core")
        (self.root / "rag_core").mkdir()

    def run_cli(self) -> tuple[int, list[str]]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--root", str(self.root)])
        return code, out.getvalue().splitlines()

    def assert_violation(self, rel: str, source: str, line: int | None = None) -> None:
        self.reset()
        self.write(rel, source)
        code, lines = self.run_cli()
        self.assertEqual(code, 1, f"expected violation for {source!r} in {rel}")
        self.assertTrue(lines)
        self.assertTrue(all(s.startswith(f"rag_core/{rel}:") for s in lines), lines)
        if line is not None:
            self.assertTrue(any(f"{rel}:{line} " in s for s in lines), lines)

    def assert_clean(self, rel: str, source: str) -> None:
        self.reset()
        self.write(rel, source)
        code, lines = self.run_cli()
        self.assertEqual((code, lines), (0, []), f"unexpected violation for {source!r}")


class TestCleanTree(CheckerTestCase):
    def test_empty_package_passes(self) -> None:
        self.assertEqual(self.run_cli(), (0, []))

    def test_clean_strict_module_passes(self) -> None:
        self.assert_clean(
            "models.py",
            "from __future__ import annotations\n"
            "import dataclasses\nimport os.path\nfrom rag_core.interfaces import X\n"
            "from dataclasses import dataclass\n",
        )

    def test_missing_package_dir_is_error_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["--root", empty])
        self.assertNotEqual(code, 0)


class TestForbiddenEverywhere(CheckerTestCase):
    NAMES = (
        "agents", "openai_agents", "mcp", "fastmcp", "google.adk",
        "src.resource.foo", "adapters", "mcp_server",
    )

    def test_each_forbidden_name_fires_via_import(self) -> None:
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assert_violation("service.py", f"import {name}\n", line=1)

    def test_each_forbidden_name_fires_via_from_import(self) -> None:
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assert_violation("service.py", f"from {name} import thing\n", line=1)

    def test_forbidden_submodule_fires(self) -> None:
        self.assert_violation("service.py", "import mcp.server.fastmcp\n")

    def test_forbidden_fires_in_store_too(self) -> None:
        self.assert_violation("store/backend.py", "import agents\n")

    def test_forbidden_fires_when_lazy(self) -> None:
        self.assert_violation("service.py", "def f():\n    import mcp\n", line=2)

    def test_near_miss_names_do_not_fire(self) -> None:
        for source in (
            "import my_agents\n",
            "import google.protobuf\n",
            "from google.protobuf import message\n",
            "import mcpx\n",
        ):
            with self.subTest(source=source):
                self.assert_clean("service.py", source)


class TestProviderSdks(CheckerTestCase):
    SDKS = (
        "openai", "anthropic", "cohere", "voyageai", "boto3", "opensearchpy",
        "google.generativeai", "google.genai",
    )

    def test_sdk_outside_store_fires(self) -> None:
        for sdk in self.SDKS:
            with self.subTest(sdk=sdk):
                self.assert_violation("service.py", f"import {sdk}\n", line=1)

    def test_lazy_sdk_outside_store_still_fires(self) -> None:
        self.assert_violation("service.py", "def f():\n    import openai\n", line=2)

    def test_sdk_in_other_subpackage_fires(self) -> None:
        self.assert_violation("other/mod.py", "import anthropic\n")

    def test_module_level_sdk_in_store_fires(self) -> None:
        for sdk in self.SDKS:
            with self.subTest(sdk=sdk):
                self.assert_violation("store/emb.py", f"import {sdk}\n", line=1)

    def test_module_level_from_import_in_store_fires(self) -> None:
        self.assert_violation("store/emb.py", "from openai import OpenAI\n", line=1)

    def test_function_level_sdk_in_store_is_fine(self) -> None:
        for sdk in self.SDKS:
            with self.subTest(sdk=sdk):
                self.assert_clean(
                    "store/emb.py", f"def build():\n    import {sdk}\n    return 1\n"
                )

    def test_method_level_sdk_in_store_is_fine(self) -> None:
        self.assert_clean(
            "store/emb.py",
            "class E:\n    def go(self):\n        from anthropic import Anthropic\n",
        )

    def test_type_checking_only_sdk_in_store_is_fine(self) -> None:
        self.assert_clean(
            "store/emb.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import openai\n",
        )

    def test_sdk_in_else_of_type_checking_is_module_level_violation(self) -> None:
        self.assert_violation(
            "store/emb.py",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    pass\nelse:\n    import openai\n",
            line=5,
        )

    def test_near_miss_sdk_names_do_not_fire(self) -> None:
        self.assert_clean("service.py", "import openai_compat\nimport boto\n")


class TestStrictZones(CheckerTestCase):
    FILES = (
        "ingest/chunk.py", "okf/graph.py", "ranking/rerank.py", "retrieve.py",
        "ingest_pipeline.py", "models.py", "interfaces.py",
    )

    def test_third_party_flagged_in_each_strict_zone(self) -> None:
        for rel in self.FILES:
            with self.subTest(rel=rel):
                self.assert_violation(rel, "import requests\n", line=1)

    def test_third_party_from_import_flagged(self) -> None:
        self.assert_violation("models.py", "from requests import get\n", line=1)

    def test_lazy_third_party_still_flagged(self) -> None:
        self.assert_violation("retrieve.py", "def f():\n    import requests\n", line=2)

    def test_stdlib_and_rag_core_pass_in_each_strict_zone(self) -> None:
        for rel in self.FILES:
            with self.subTest(rel=rel):
                self.assert_clean(
                    rel,
                    "from __future__ import annotations\nimport json\n"
                    "from dataclasses import dataclass\nimport rag_core.models\n"
                    "from rag_core.models import Chunk\n",
                )

    def test_third_party_allowed_in_non_strict_module(self) -> None:
        self.assert_clean("helpers.py", "import requests\n")

    def test_relative_import_within_package_passes(self) -> None:
        self.assert_clean("ingest/chunk.py", "from ..models import Chunk\nfrom . import x\n")
        self.assert_clean("models.py", "from . import interfaces\nfrom .interfaces import P\n")

    def test_relative_import_escaping_package_flagged(self) -> None:
        self.assert_violation("models.py", "from .. import tools\n", line=1)
        self.assert_violation("ingest/chunk.py", "from ...pkg import tools\n", line=1)

    def test_relative_escape_flagged_outside_strict_zone_too(self) -> None:
        self.assert_violation("helpers.py", "from .. import tools\n", line=1)


class TestDynamicImports(CheckerTestCase):
    def test_import_module_literal_provider_flagged(self) -> None:
        self.assert_violation(
            "service.py", "import importlib\nimportlib.import_module('openai')\n", line=2
        )

    def test_dunder_import_literal_provider_flagged(self) -> None:
        self.assert_violation("service.py", "__import__('openai')\n", line=1)

    def test_dynamic_import_of_forbidden_name_flagged(self) -> None:
        self.assert_violation("service.py", "__import__('mcp')\n", line=1)

    def test_dynamic_import_in_strict_zone_third_party_flagged(self) -> None:
        self.assert_violation(
            "models.py",
            "from importlib import import_module\nimport_module('requests')\n",
            line=2,
        )

    def test_dynamic_provider_in_store_lazy_is_fine(self) -> None:
        self.assert_clean(
            "store/emb.py",
            "import importlib\ndef f():\n    return importlib.import_module('openai')\n",
        )

    def test_dynamic_stdlib_import_is_fine(self) -> None:
        self.assert_clean("models.py", "__import__('json')\n")

    def test_non_literal_dynamic_import_is_not_flagged(self) -> None:
        self.assert_clean("service.py", "import importlib\nimportlib.import_module(name)\n")

    def test_relative_literal_dynamic_import_flagged(self) -> None:
        self.assert_violation(
            "service.py", "import importlib\nimportlib.import_module('.x')\n", line=2
        )


class TestReportingAndExitCodes(CheckerTestCase):
    def test_unparsable_file_is_reported(self) -> None:
        self.write("broken.py", "def (:\n")
        code, lines = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("rag_core/broken.py:1 "), lines)

    def test_output_format_is_file_line_message(self) -> None:
        self.write("a.py", "import mcp\n")
        self.write("b.py", "x = 1\nimport agents\n")
        code, lines = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertRegex(line, VIOLATION_LINE)
        self.assertTrue(lines[0].startswith("rag_core/a.py:1 "))
        self.assertTrue(lines[1].startswith("rag_core/b.py:2 "))

    def test_violations_in_nested_dirs_use_posix_relative_paths(self) -> None:
        self.write("store/sub/deep.py", "import agents\n")
        _, lines = self.run_cli()
        self.assertTrue(lines[0].startswith("rag_core/store/sub/deep.py:1 "), lines)

    def test_output_is_deterministic(self) -> None:
        for name in ("z.py", "a.py", "m.py"):
            self.write(name, "import mcp\n")
        first = self.run_cli()
        self.assertEqual(first, self.run_cli())
        self.assertEqual(first[1], sorted(first[1]))

    def test_files_outside_rag_core_are_ignored(self) -> None:
        (self.root / "other").mkdir()
        (self.root / "other" / "x.py").write_text("import mcp\n", encoding="utf-8")
        self.assertEqual(self.run_cli(), (0, []))

    def test_clean_exit_zero_after_violation_removed(self) -> None:
        self.write("a.py", "import mcp\n")
        self.assertEqual(self.run_cli()[0], 1)
        self.write("a.py", "import json\n")
        self.assertEqual(self.run_cli(), (0, []))


class TestRealRepository(unittest.TestCase):
    def test_real_rag_core_is_clean(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--root", str(REPO_ROOT)])
        self.assertEqual((code, out.getvalue()), (0, ""))


if __name__ == "__main__":
    unittest.main()
