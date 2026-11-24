"""Tests for evals/schema.py (loader, validation rules, manifest reader, disjointness)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from evals.schema import (
    EvalQuestion,
    EvalSet,
    QuestionType,
    assert_disjoint,
    load_manifest_doc_ids,
    normalize_text,
)
from rag_core.exceptions import ValidationError


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": "q1",
        "question": "What is the warranty period?",
        "qtype": "single_fact",
        "expected_answer": "Two years",
        "relevant_doc_ids": ["doc-1"],
        "tenant_id": "tenant-a",
    }
    row.update(overrides)
    return row


def _jsonl(*rows: dict[str, Any]) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


class TestEvalQuestionRules(unittest.TestCase):
    """Relevance and emptiness rules."""

    def test_valid_single_fact_loads(self) -> None:
        q = EvalQuestion.from_mapping(_row())
        self.assertEqual(q.qtype, QuestionType.SINGLE_FACT)
        self.assertEqual(q.relevant_doc_ids, ("doc-1",))
        self.assertEqual(q.notes, "")

    def test_notes_are_optional_and_kept(self) -> None:
        self.assertEqual(EvalQuestion.from_mapping(_row(notes="hi")).notes, "hi")

    def test_unanswerable_with_relevant_docs_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvalQuestion.from_mapping(_row(qtype="unanswerable"))

    def test_unanswerable_without_relevant_docs_accepted(self) -> None:
        q = EvalQuestion.from_mapping(_row(qtype="unanswerable", relevant_doc_ids=[]))
        self.assertEqual(q.relevant_doc_ids, ())

    def test_answerable_without_relevant_docs_rejected(self) -> None:
        for qtype in ("single_fact", "multi_hop"):
            with self.subTest(qtype=qtype), self.assertRaises(ValidationError):
                EvalQuestion.from_mapping(_row(qtype=qtype, relevant_doc_ids=[]))

    def test_multi_hop_with_one_doc_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvalQuestion.from_mapping(_row(qtype="multi_hop", relevant_doc_ids=["a"]))

    def test_multi_hop_with_two_docs_accepted(self) -> None:
        q = EvalQuestion.from_mapping(_row(qtype="multi_hop", relevant_doc_ids=["a", "b"]))
        self.assertEqual(q.relevant_doc_ids, ("a", "b"))

    def test_empty_or_blank_question_rejected(self) -> None:
        for text in ("", "   "):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                EvalQuestion.from_mapping(_row(question=text))

    def test_empty_or_blank_tenant_rejected(self) -> None:
        for tenant in ("", "  "):
            with self.subTest(tenant=tenant), self.assertRaises(ValidationError):
                EvalQuestion.from_mapping(_row(tenant_id=tenant))

    def test_unknown_key_rejected_and_named(self) -> None:
        with self.assertRaisesRegex(ValidationError, "bogus"):
            EvalQuestion.from_mapping(_row(bogus=1))

    def test_each_missing_required_key_rejected_and_named(self) -> None:
        for key in (
            "question_id",
            "question",
            "qtype",
            "expected_answer",
            "relevant_doc_ids",
            "tenant_id",
        ):
            row = _row()
            del row[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, key):
                EvalQuestion.from_mapping(row)

    def test_unknown_qtype_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvalQuestion.from_mapping(_row(qtype="trivia"))

    def test_relevant_doc_ids_must_be_list_of_strings(self) -> None:
        for bad in ("doc-1", [1, 2], None):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                EvalQuestion.from_mapping(_row(relevant_doc_ids=bad))


class TestEvalSetLoading(unittest.TestCase):
    """JSONL loading, uniqueness, manifest membership, error line numbers."""

    def test_loads_questions_in_file_order(self) -> None:
        text = _jsonl(_row(question_id="b", question="two"), _row(question_id="a", question="one"))
        loaded = EvalSet.loads(text)
        self.assertEqual([q.question_id for q in loaded.questions], ["b", "a"])

    def test_blank_lines_are_skipped(self) -> None:
        text = json.dumps(_row(question_id="a")) + "\n\n   \n" + json.dumps(_row(question_id="b"))
        self.assertEqual(len(EvalSet.loads(text).questions), 2)

    def test_load_jsonl_uses_file_stem_as_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev_set.jsonl"
            path.write_text(_jsonl(_row()), encoding="utf-8")
            loaded = EvalSet.load_jsonl(path)
        self.assertEqual(loaded.name, "dev_set")
        self.assertEqual(len(loaded.questions), 1)

    def test_duplicate_question_id_rejected_with_line_number(self) -> None:
        text = _jsonl(_row(), _row(question="another"))
        with self.assertRaisesRegex(ValidationError, "line 2.*duplicate"):
            EvalSet.loads(text)

    def test_bad_json_error_names_the_line_number(self) -> None:
        text = json.dumps(_row()) + "\n" + "{not json}\n"
        with self.assertRaisesRegex(ValidationError, "line 2"):
            EvalSet.loads(text)

    def test_non_object_line_rejected_with_line_number(self) -> None:
        with self.assertRaisesRegex(ValidationError, "line 1"):
            EvalSet.loads("[1, 2]\n")

    def test_rule_violation_error_names_the_line_number(self) -> None:
        text = _jsonl(_row(), _row(question_id="q2", qtype="multi_hop", relevant_doc_ids=["a"]))
        with self.assertRaisesRegex(ValidationError, "line 2"):
            EvalSet.loads(text)

    def test_doc_id_outside_manifest_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "doc-1"):
            EvalSet.loads(_jsonl(_row()), manifest_doc_ids={"doc-2"})

    def test_doc_id_inside_manifest_accepted(self) -> None:
        loaded = EvalSet.loads(_jsonl(_row()), manifest_doc_ids=["doc-1", "doc-2"])
        self.assertEqual(len(loaded.questions), 1)

    def test_manifest_not_enforced_when_not_supplied(self) -> None:
        self.assertEqual(len(EvalSet.loads(_jsonl(_row(relevant_doc_ids=["ghost"]))).questions), 1)

    def test_type_counts_include_zeros_in_enum_order(self) -> None:
        loaded = EvalSet.loads(_jsonl(_row()))
        counts = loaded.type_counts()
        self.assertEqual(counts, {"single_fact": 1, "multi_hop": 0, "unanswerable": 0})
        self.assertEqual(list(counts), ["single_fact", "multi_hop", "unanswerable"])

    def test_type_counts_of_empty_set_all_zero(self) -> None:
        self.assertEqual(
            EvalSet(name="x").type_counts(),
            {"single_fact": 0, "multi_hop": 0, "unanswerable": 0},
        )


class TestLoadManifest(unittest.TestCase):
    """Both accepted manifest shapes plus failures."""

    def _load(self, payload: str) -> frozenset[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(payload, encoding="utf-8")
            return load_manifest_doc_ids(path)

    def test_list_of_objects_shape(self) -> None:
        self.assertEqual(self._load('[{"doc_id": "a"}, {"doc_id": "b"}]'), frozenset({"a", "b"}))

    def test_object_with_documents_shape(self) -> None:
        payload = '{"documents": [{"doc_id": "a", "title": "x"}]}'
        self.assertEqual(self._load(payload), frozenset({"a"}))

    def test_entry_without_doc_id_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self._load('[{"title": "x"}]')

    def test_wrong_top_level_shape_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self._load('{"docs": []}')

    def test_invalid_json_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self._load("{oops")


class TestAssertDisjoint(unittest.TestCase):
    """Dev / held-out leakage detection."""

    def _set(self, name: str, **row: Any) -> EvalSet:
        return EvalSet.loads(_jsonl(_row(**row)), name=name)

    def test_disjoint_sets_pass(self) -> None:
        assert_disjoint(
            self._set("dev", question_id="d1", question="alpha"),
            self._set("held", question_id="h1", question="beta"),
        )

    def test_shared_question_id_caught(self) -> None:
        with self.assertRaisesRegex(ValidationError, "shared question_id"):
            assert_disjoint(
                self._set("dev", question_id="same", question="alpha"),
                self._set("held", question_id="same", question="beta"),
            )

    def test_shared_text_differing_in_case_and_whitespace_caught(self) -> None:
        with self.assertRaisesRegex(ValidationError, "shared question text"):
            assert_disjoint(
                self._set("dev", question_id="d1", question="What  is the   Warranty?"),
                self._set("held", question_id="h1", question="  what is the warranty? "),
            )

    def test_normalize_text_casefolds_and_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  Hello \t World\n"), "hello world")


if __name__ == "__main__":
    unittest.main()
