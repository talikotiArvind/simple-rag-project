"""CLI gate validating the dev and held-out eval sets (size, type mix, disjointness, tenants)."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evals.schema import EvalSet, assert_disjoint, load_manifest_doc_ids
from rag_core.exceptions import ValidationError

DEFAULT_DEV_SIZE = 50
DEFAULT_MIN_HELDOUT = 30
MIN_TENANTS = 2


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a validation run: failure lines, warning lines and a printable summary."""

    problems: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    summary: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when no failures were found (skipped files only produce warnings)."""
        return not self.problems

    @property
    def gate_satisfied(self) -> bool:
        """True only when nothing failed and nothing was skipped."""
        return not self.problems and not self.warnings


class SetValidator:
    """Validates the dev and held-out JSONL sets against the Phase 0 gate rules."""

    def __init__(
        self,
        dev_path: Path,
        heldout_path: Path,
        manifest_path: Path,
        dev_size: int = DEFAULT_DEV_SIZE,
        min_heldout: int = DEFAULT_MIN_HELDOUT,
        allow_missing: bool = False,
    ) -> None:
        if dev_size < 1:
            raise ValueError(f"dev_size must be positive, got {dev_size}")
        if min_heldout < 1:
            raise ValueError(f"min_heldout must be positive, got {min_heldout}")
        self._dev_path = dev_path
        self._heldout_path = heldout_path
        self._manifest_path = manifest_path
        self._dev_size = dev_size
        self._min_heldout = min_heldout
        self._allow_missing = allow_missing
        self._problems: list[str] = []
        self._warnings: list[str] = []

    def validate(self) -> ValidationReport:
        """Run all checks and return the report; never raises for bad input files."""
        self._problems, self._warnings = [], []
        doc_ids = self._load_manifest()
        dev = self._load_set("dev", self._dev_path, doc_ids)
        heldout = self._load_set("held-out", self._heldout_path, doc_ids)
        self._check_sizes(dev, heldout)
        self._check_type_mix("dev", dev)
        self._check_type_mix("held-out", heldout)
        self._check_disjoint(dev, heldout)
        self._check_tenants([s for s in (dev, heldout) if s is not None])
        return ValidationReport(
            problems=tuple(self._problems),
            warnings=tuple(self._warnings),
            summary=tuple(self._summarize(dev, heldout)),
        )

    def _missing(self, label: str, path: Path) -> None:
        if self._allow_missing:
            self._warnings.append(
                f"WARNING: {label} file {path} is missing; skipped (--allow-missing). "
                "The gate is NOT satisfied."
            )
        else:
            self._problems.append(f"{label} file not found: {path}")

    def _load_manifest(self) -> frozenset[str] | None:
        if not self._manifest_path.is_file():
            self._missing("manifest", self._manifest_path)
            return None
        try:
            return load_manifest_doc_ids(self._manifest_path)
        except (ValidationError, OSError) as exc:
            self._problems.append(f"manifest: {exc}")
            return None

    def _load_set(
        self, label: str, path: Path, doc_ids: frozenset[str] | None
    ) -> EvalSet | None:
        if not path.is_file():
            self._missing(label, path)
            return None
        try:
            return EvalSet.load_jsonl(path, manifest_doc_ids=doc_ids)
        except (ValidationError, OSError, UnicodeDecodeError) as exc:
            self._problems.append(f"{label} set {path}: {exc}")
            return None

    def _check_sizes(self, dev: EvalSet | None, heldout: EvalSet | None) -> None:
        if dev is not None and len(dev.questions) != self._dev_size:
            self._problems.append(
                f"dev set has {len(dev.questions)} questions, expected exactly {self._dev_size}"
            )
        if heldout is not None and len(heldout.questions) < self._min_heldout:
            self._problems.append(
                f"held-out set has {len(heldout.questions)} questions, "
                f"expected at least {self._min_heldout}"
            )

    def _check_type_mix(self, label: str, eval_set: EvalSet | None) -> None:
        if eval_set is None:
            return
        for qtype, count in eval_set.type_counts().items():
            if count < 1:
                self._problems.append(f"{label} set has no {qtype} questions")

    def _check_disjoint(self, dev: EvalSet | None, heldout: EvalSet | None) -> None:
        if dev is None or heldout is None:
            return
        try:
            assert_disjoint(dev, heldout)
        except ValidationError as exc:
            self._problems.append(str(exc))

    def _check_tenants(self, sets: list[EvalSet]) -> None:
        tenants = {q.tenant_id for s in sets for q in s.questions}
        if sets and len(tenants) < MIN_TENANTS and not self._warnings:
            self._problems.append(
                f"need at least {MIN_TENANTS} distinct tenant_ids across sets, "
                f"found {len(tenants)}"
            )

    @staticmethod
    def _summarize(dev: EvalSet | None, heldout: EvalSet | None) -> list[str]:
        lines = []
        for label, eval_set in (("dev", dev), ("held-out", heldout)):
            if eval_set is None:
                lines.append(f"{label}: not validated")
                continue
            mix = ", ".join(f"{k}={v}" for k, v in eval_set.type_counts().items())
            lines.append(f"{label}: {len(eval_set.questions)} questions ({mix})")
        return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.validate_sets", description=__doc__)
    parser.add_argument("--dev", type=Path, required=True, help="dev set JSONL")
    parser.add_argument("--heldout", type=Path, required=True, help="held-out set JSONL")
    parser.add_argument("--manifest", type=Path, required=True, help="corpus manifest JSON")
    parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE)
    parser.add_argument("--min-heldout", type=int, default=DEFAULT_MIN_HELDOUT)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip missing files (only before the human-authored sets land)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: print the summary and failures; return 0 only if the gate passes."""
    args = _build_parser().parse_args(argv)
    validator = SetValidator(
        args.dev,
        args.heldout,
        args.manifest,
        dev_size=args.dev_size,
        min_heldout=args.min_heldout,
        allow_missing=args.allow_missing,
    )
    report = validator.validate()
    for line in report.summary:
        print(line)
    for warning in report.warnings:
        print(warning)
    for problem in report.problems:
        print(f"FAIL: {problem}")
    if not report.ok:
        print("RESULT: FAIL")
    elif not report.gate_satisfied:
        print("RESULT: PASS WITH WARNINGS - the gate is NOT satisfied")
    else:
        print("RESULT: PASS")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
