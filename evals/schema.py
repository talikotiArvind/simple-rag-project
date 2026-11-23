"""Eval question schema, JSONL set loader, manifest reader and dev/held-out disjointness check."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rag_core.exceptions import ValidationError

MIN_MULTI_HOP_DOCS = 2
_FIELDS = frozenset(
    {
        "question_id",
        "question",
        "qtype",
        "expected_answer",
        "relevant_doc_ids",
        "tenant_id",
        "notes",
    }
)
_REQUIRED = _FIELDS - {"notes"}


class QuestionType(StrEnum):
    """Kind of eval question."""

    SINGLE_FACT = "single_fact"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


def normalize_text(text: str) -> str:
    """Casefold and collapse whitespace so equivalent questions compare equal."""
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class EvalQuestion:
    """One labelled eval question; relevance rules are enforced on construction."""

    question_id: str
    question: str
    qtype: QuestionType
    expected_answer: str
    tenant_id: str
    relevant_doc_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValidationError("question_id must not be empty")
        if not self.question.strip():
            raise ValidationError("question must not be empty")
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty")
        self._check_relevance()

    def _check_relevance(self) -> None:
        count = len(self.relevant_doc_ids)
        if any(not doc_id.strip() for doc_id in self.relevant_doc_ids):
            raise ValidationError("relevant_doc_ids must not contain empty ids")
        if self.qtype is QuestionType.UNANSWERABLE:
            if count:
                raise ValidationError("unanswerable question must have no relevant_doc_ids")
        elif count < 1:
            raise ValidationError("answerable question needs at least 1 relevant doc id")
        elif self.qtype is QuestionType.MULTI_HOP and count < MIN_MULTI_HOP_DOCS:
            raise ValidationError(
                f"multi_hop question needs at least {MIN_MULTI_HOP_DOCS} relevant doc ids"
            )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> EvalQuestion:
        """Build a question from a parsed JSON object, rejecting unknown or missing keys."""
        unknown = sorted(set(raw) - _FIELDS)
        if unknown:
            raise ValidationError(f"unknown keys: {', '.join(unknown)}")
        missing = sorted(_REQUIRED - set(raw))
        if missing:
            raise ValidationError(f"missing keys: {', '.join(missing)}")
        return cls(
            question_id=_as_str(raw, "question_id"),
            question=_as_str(raw, "question"),
            qtype=_as_qtype(raw["qtype"]),
            expected_answer=_as_str(raw, "expected_answer"),
            tenant_id=_as_str(raw, "tenant_id"),
            relevant_doc_ids=_as_doc_ids(raw["relevant_doc_ids"]),
            notes=_as_str(raw, "notes") if "notes" in raw else "",
        )


def _as_str(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise ValidationError(f"{key} must be a string")
    return value


def _as_qtype(value: object) -> QuestionType:
    try:
        return QuestionType(str(value))
    except ValueError:
        allowed = ", ".join(q.value for q in QuestionType)
        raise ValidationError(f"qtype must be one of: {allowed}") from None


def _as_doc_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValidationError("relevant_doc_ids must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class EvalSet:
    """Named, ordered collection of eval questions."""

    name: str
    questions: tuple[EvalQuestion, ...] = field(default_factory=tuple)

    @classmethod
    def load_jsonl(
        cls,
        path: Path | str,
        manifest_doc_ids: Collection[str] | None = None,
    ) -> EvalSet:
        """Load and validate a JSONL file; the set name is the file stem."""
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        return cls.loads(text, name=file_path.stem, manifest_doc_ids=manifest_doc_ids)

    @classmethod
    def loads(
        cls,
        text: str,
        name: str = "eval_set",
        manifest_doc_ids: Collection[str] | None = None,
    ) -> EvalSet:
        """Validate JSONL text; errors carry 1-based line numbers. File order is kept."""
        known = None if manifest_doc_ids is None else frozenset(manifest_doc_ids)
        seen: dict[str, int] = {}
        questions: list[EvalQuestion] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            question = _parse_line(line, line_no)
            _check_unique(question, line_no, seen)
            _check_manifest(question, line_no, known)
            questions.append(question)
        return cls(name=name, questions=tuple(questions))

    def type_counts(self) -> dict[str, int]:
        """Return question counts per type, including zero counts, in enum order."""
        counts = Counter(q.qtype for q in self.questions)
        return {qtype.value: counts.get(qtype, 0) for qtype in QuestionType}


def _parse_line(line: str, line_no: int) -> EvalQuestion:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValidationError(f"line {line_no}: expected a JSON object")
    try:
        return EvalQuestion.from_mapping(raw)
    except ValidationError as exc:
        raise ValidationError(f"line {line_no}: {exc}") from exc


def _check_unique(question: EvalQuestion, line_no: int, seen: dict[str, int]) -> None:
    first = seen.get(question.question_id)
    if first is not None:
        raise ValidationError(
            f"line {line_no}: duplicate question_id {question.question_id!r} "
            f"(first at line {first})"
        )
    seen[question.question_id] = line_no


def _check_manifest(
    question: EvalQuestion, line_no: int, known: frozenset[str] | None
) -> None:
    if known is None:
        return
    missing = sorted(set(question.relevant_doc_ids) - known)
    if missing:
        raise ValidationError(
            f"line {line_no}: relevant doc ids not in manifest: {', '.join(missing)}"
        )


def load_manifest_doc_ids(path: Path | str) -> frozenset[str]:
    """Read doc ids from a manifest: a list of objects, or an object with a 'documents' list."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"manifest {path}: invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    entries = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValidationError("manifest must be a list or an object with a 'documents' list")
    return frozenset(_manifest_doc_id(entry, index) for index, entry in enumerate(entries))


def _manifest_doc_id(entry: object, index: int) -> str:
    doc_id = entry.get("doc_id") if isinstance(entry, dict) else None
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValidationError(f"manifest entry {index}: missing non-empty string doc_id")
    return doc_id


def assert_disjoint(dev: EvalSet, heldout: EvalSet) -> None:
    """Raise if the sets share a normalized question text or a question_id."""
    shared_text = _shared(
        (normalize_text(q.question) for q in dev.questions),
        (normalize_text(q.question) for q in heldout.questions),
    )
    shared_ids = _shared(
        (q.question_id for q in dev.questions), (q.question_id for q in heldout.questions)
    )
    problems = []
    if shared_text:
        problems.append(f"shared question text: {shared_text}")
    if shared_ids:
        problems.append(f"shared question_id: {shared_ids}")
    if problems:
        raise ValidationError("dev and held-out sets overlap; " + "; ".join(problems))


def _shared(left: Iterable[str], right: Iterable[str]) -> list[str]:
    return sorted(set(left) & set(right))
