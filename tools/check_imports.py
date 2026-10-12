"""AST-based D-005 import-boundary checker for rag_core (stdlib only)."""
from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# --- Rules table (data): extend here for later phases --------------------------------
PACKAGE = "rag_core"
# Rule 1: forbidden anywhere under PACKAGE (agent/tool frameworks and upper layers).
FORBIDDEN_EVERYWHERE: tuple[str, ...] = (
    "agents",
    "openai_agents",
    "mcp",
    "fastmcp",
    "google.adk",
    "src",
    "adapters",
    "mcp_server",
)
# Rule 2: provider SDKs, allowed only under PROVIDER_ZONE and only as lazy imports.
PROVIDER_SDKS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "cohere",
    "voyageai",
    "boto3",
    "opensearchpy",
    "google.generativeai",
    "google.genai",
)
PROVIDER_ZONE = "store"
# Rule 3: first path component under PACKAGE (dir or file) limited to stdlib + PACKAGE.
STRICT_ZONES: frozenset[str] = frozenset(
    {
        "ingest",
        "okf",
        "ranking",
        "retrieve.py",
        "ingest_pipeline.py",
        "models.py",
        "interfaces.py",
        "config.py",
        "exceptions.py",
    }
)
DYNAMIC_IMPORT_FUNCS: frozenset[str] = frozenset({"__import__", "import_module"})
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One boundary violation, rendered as ``FILE:LINE message``."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.message}"


@dataclass(frozen=True)
class ImportRef:
    """An import statement or literal dynamic import found in a module."""

    name: str  # absolute dotted name, or "" when a relative import escapes the package
    line: int
    lazy: bool  # inside a function body
    type_checking: bool  # inside an ``if TYPE_CHECKING:`` block
    escapes: bool = False  # relative import resolving outside PACKAGE


def matches(name: str, prefix: str) -> bool:
    """Return True if dotted ``name`` equals ``prefix`` or is a submodule of it."""
    return name == prefix or name.startswith(prefix + ".")


def _is_type_checking_test(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    return isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"


class ImportCollector(ast.NodeVisitor):
    """Collects every import (static, relative, literal dynamic) with its context."""

    def __init__(self, package_parts: Sequence[str]) -> None:
        self._package_parts = list(package_parts)
        self._func_depth = 0
        self._tc_depth = 0
        self.refs: list[ImportRef] = []

    def _add(self, name: str, line: int, escapes: bool = False) -> None:
        self.refs.append(
            ImportRef(name, line, self._func_depth > 0, self._tc_depth > 0, escapes)
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._add(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.level == 0:
            self._add(node.module or "", node.lineno)
            return
        keep = len(self._package_parts) - (node.level - 1)
        if keep < 1:
            self._add("", node.lineno, escapes=True)
            return
        base = self._package_parts[:keep] + ([node.module] if node.module else [])
        self._add(".".join(base), node.lineno)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if called in DYNAMIC_IMPORT_FUNCS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                relative = first.value.startswith(".")
                self._add("" if relative else first.value, node.lineno, escapes=relative)
        self.generic_visit(node)

    def _visit_function(self, node: ast.AST) -> None:
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    visit_FunctionDef = _visit_function  # noqa: N815
    visit_AsyncFunctionDef = _visit_function  # noqa: N815
    visit_Lambda = _visit_function  # noqa: N815

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if not _is_type_checking_test(node.test):
            self.generic_visit(node)
            return
        self._tc_depth += 1
        for child in node.body:
            self.visit(child)
        self._tc_depth -= 1
        for child in node.orelse:
            self.visit(child)


class ImportBoundaryChecker:
    """Applies the D-005 import-boundary rules to a repository tree."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._stdlib = frozenset(sys.stdlib_module_names)

    def check(self) -> list[Violation]:
        """Check every module under ``<root>/rag_core``; return sorted violations."""
        package_dir = self._root / PACKAGE
        violations: list[Violation] = []
        for path in sorted(package_dir.rglob("*.py")):
            violations.extend(self.check_file(path))
        return sorted(violations, key=lambda v: (v.path, v.line, v.message))

    def check_file(self, path: Path) -> list[Violation]:
        """Check a single module file located under ``<root>/rag_core``."""
        rel = path.relative_to(self._root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            return [Violation(rel, exc.lineno or 1, f"cannot parse: {exc.msg}")]
        inside = path.relative_to(self._root / PACKAGE).parts
        collector = ImportCollector(self._package_parts(inside))
        collector.visit(tree)
        return [v for ref in collector.refs for v in self._check_ref(rel, inside, ref)]

    @staticmethod
    def _package_parts(inside: tuple[str, ...]) -> list[str]:
        return [PACKAGE, *inside[:-1]]

    def _check_ref(
        self, rel: str, inside: tuple[str, ...], ref: ImportRef
    ) -> list[Violation]:
        found: list[str] = []
        if ref.escapes:
            found.append("relative import escapes rag_core")
        found.extend(self._forbidden(ref))
        found.extend(self._provider(inside, ref))
        found.extend(self._strict(inside, ref))
        return [Violation(rel, ref.line, msg) for msg in found]

    @staticmethod
    def _forbidden(ref: ImportRef) -> list[str]:
        hits = [p for p in FORBIDDEN_EVERYWHERE if matches(ref.name, p)]
        return [f"D-005: rag_core must not import '{ref.name}' ({p})" for p in hits[:1]]

    @staticmethod
    def _provider(inside: tuple[str, ...], ref: ImportRef) -> list[str]:
        if not any(matches(ref.name, p) for p in PROVIDER_SDKS):
            return []
        if inside[0] != PROVIDER_ZONE:
            return [f"D-005: provider SDK '{ref.name}' allowed only in rag_core/store/"]
        if not ref.lazy and not ref.type_checking:
            return [f"D-005: provider SDK '{ref.name}' must be a function-level (lazy) import"]
        return []

    def _strict(self, inside: tuple[str, ...], ref: ImportRef) -> list[str]:
        if inside[0] not in STRICT_ZONES or not ref.name:
            return []
        top = ref.name.split(".", 1)[0]
        if top in self._stdlib or top == PACKAGE:
            return []
        return [f"D-005: '{ref.name}' is not stdlib or rag_core.* (strict module)"]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: print violations, return 1 if any, else 0."""
    parser = argparse.ArgumentParser(description="Check D-005 import boundaries.")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="repository root containing rag_core/ (default: this repo)",
    )
    args = parser.parse_args(argv)
    if not (args.root / PACKAGE).is_dir():
        print(f"error: {args.root / PACKAGE} is not a directory", file=sys.stderr)
        return 2
    violations = ImportBoundaryChecker(args.root).check()
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
