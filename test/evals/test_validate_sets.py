"""Tests for evals/validate_sets.py (Phase 0 gate: size, type mix, disjointness, tenants)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evals.validate_sets import SetValidator, main

_DOCS = ["doc-1", "doc-2", "doc-3"]


def _line(qid: str, text: str, qtype: str, tenant: str) -> str:
    relevant = {"single_fact": ["doc-1"], "multi_hop": ["doc-1", "doc-2"], "unanswerable": []}
    return json.dumps(
        {
            "question_id": qid,
            "question": text,
            "qtype": qtype,
            "expected_answer": "x",
            "relevant_doc_ids": relevant[qtype],
            "tenant_id": tenant,
        }
    )


def _make_set(
    prefix: str, mix: dict[str, int], tenants: tuple[str, ...] = ("tenant-a", "tenant-b")
) -> str:
    lines: list[str] = []
    for qtype, count in mix.items():
        for _ in range(count):
            n = len(lines)
            lines.append(
                _line(f"{prefix}-{n}", f"{prefix} question {n}", qtype, tenants[n % len(tenants)])
            )
    return "\n".join(lines) + "\n"


DEV_MIX = {"single_fact": 20, "multi_hop": 20, "unanswerable": 10}  # 50
HELD_MIX = {"single_fact": 12, "multi_hop": 12, "unanswerable": 6}  # 30


class _ValidatorCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.dev = root / "dev.jsonl"
        self.held = root / "held.jsonl"
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps([{"doc_id": d} for d in _DOCS]), encoding="utf-8")
        self.dev.write_text(_make_set("d", DEV_MIX), encoding="utf-8")
        self.held.write_text(_make_set("h", HELD_MIX), encoding="utf-8")

    def run_main(self, *extra: str) -> tuple[int, str]:
        argv = [
            "--dev",
            str(self.dev),
            "--heldout",
            str(self.held),
            "--manifest",
            str(self.manifest),
            *extra,
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def validator(self, **kwargs: object) -> SetValidator:
        return SetValidator(self.dev, self.held, self.manifest, **kwargs)  # type: ignore[arg-type]


class TestValidSets(_ValidatorCase):
    """Happy path."""

    def test_valid_sets_exit_zero_and_pass(self) -> None:
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("RESULT: PASS", out)
        self.assertNotIn("WARNING", out)

    def test_summary_lists_counts(self) -> None:
        _, out = self.run_main()
        self.assertIn("dev: 50 questions", out)
        self.assertIn("held-out: 30 questions", out)

    def test_report_gate_satisfied(self) -> None:
        report = self.validator().validate()
        self.assertTrue(report.ok)
        self.assertTrue(report.gate_satisfied)

    def test_heldout_larger_than_minimum_accepted(self) -> None:
        self.held.write_text(
            _make_set("h", {"single_fact": 20, "multi_hop": 15, "unanswerable": 5})
        )
        self.assertEqual(self.run_main()[0], 0)


class TestSizeRules(_ValidatorCase):
    """Dev exactly 50, held-out at least 30."""

    def test_dev_with_49_fails(self) -> None:
        self.dev.write_text(_make_set("d", {"single_fact": 20, "multi_hop": 20, "unanswerable": 9}))
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("expected exactly 50", out)

    def test_dev_with_51_fails(self) -> None:
        self.dev.write_text(
            _make_set("d", {"single_fact": 21, "multi_hop": 20, "unanswerable": 10})
        )
        self.assertEqual(self.run_main()[0], 1)

    def test_heldout_with_29_fails(self) -> None:
        self.held.write_text(
            _make_set("h", {"single_fact": 12, "multi_hop": 12, "unanswerable": 5})
        )
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("at least 30", out)

    def test_heldout_with_exactly_30_passes(self) -> None:
        self.assertEqual(self.run_main()[0], 0)

    def test_configurable_sizes(self) -> None:
        self.assertEqual(self.run_main("--dev-size", "49")[0], 1)
        self.assertEqual(self.run_main("--min-heldout", "31")[0], 1)
        self.held.write_text(_make_set("h", {"single_fact": 2, "multi_hop": 2, "unanswerable": 1}))
        self.assertEqual(self.run_main("--min-heldout", "5")[0], 0)


class TestTypeMix(_ValidatorCase):
    """At least one of each type in both sets."""

    def test_dev_missing_a_type_fails(self) -> None:
        self.dev.write_text(_make_set("d", {"single_fact": 30, "multi_hop": 20, "unanswerable": 0}))
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("dev set has no unanswerable", out)

    def test_heldout_missing_a_type_fails(self) -> None:
        self.held.write_text(
            _make_set("h", {"single_fact": 15, "multi_hop": 0, "unanswerable": 15})
        )
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("held-out set has no multi_hop", out)


class TestDisjointnessAndTenants(_ValidatorCase):
    """Leakage and tenant diversity."""

    def test_shared_text_between_sets_fails(self) -> None:
        shared = _line("h-0", "d question 0", "single_fact", "tenant-a")
        rest = self.held.read_text().splitlines()[1:]
        self.held.write_text("\n".join([shared, *rest]) + "\n")
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("overlap", out)

    def test_shared_id_between_sets_fails(self) -> None:
        first = self.held.read_text().splitlines()[0].replace('"h-0"', '"d-0"')
        rest = self.held.read_text().splitlines()[1:]
        self.held.write_text("\n".join([first, *rest]) + "\n")
        self.assertEqual(self.run_main()[0], 1)

    def test_single_tenant_across_sets_fails(self) -> None:
        self.dev.write_text(_make_set("d", DEV_MIX, tenants=("tenant-a",)))
        self.held.write_text(_make_set("h", HELD_MIX, tenants=("tenant-a",)))
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("at least 2 distinct tenant_ids", out)

    def test_two_tenants_split_across_sets_passes(self) -> None:
        self.dev.write_text(_make_set("d", DEV_MIX, tenants=("tenant-a",)))
        self.held.write_text(_make_set("h", HELD_MIX, tenants=("tenant-b",)))
        self.assertEqual(self.run_main()[0], 0)


class TestInvalidContent(_ValidatorCase):
    """Schema errors surface as failures, not exceptions."""

    def test_malformed_dev_json_fails_with_line_number(self) -> None:
        self.dev.write_text("{bad\n")
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("line 1", out)

    def test_doc_id_missing_from_manifest_fails(self) -> None:
        self.manifest.write_text(json.dumps([{"doc_id": "doc-1"}]))
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("doc-2", out)

    def test_invalid_manifest_fails(self) -> None:
        self.manifest.write_text("{bad")
        self.assertEqual(self.run_main()[0], 1)


class TestAllowMissing(_ValidatorCase):
    """Missing files fail by default; --allow-missing warns and never satisfies the gate."""

    def test_missing_dev_fails_without_flag(self) -> None:
        self.dev.unlink()
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("not found", out)

    def test_missing_heldout_fails_without_flag(self) -> None:
        self.held.unlink()
        self.assertEqual(self.run_main()[0], 1)

    def test_missing_manifest_fails_without_flag(self) -> None:
        self.manifest.unlink()
        self.assertEqual(self.run_main()[0], 1)

    def test_allow_missing_prints_warning_and_exits_zero(self) -> None:
        self.held.unlink()
        code, out = self.run_main("--allow-missing")
        self.assertEqual(code, 0)
        self.assertIn("WARNING", out)
        self.assertIn("NOT satisfied", out)
        self.assertNotIn("RESULT: PASS\n", out)

    def test_allow_missing_report_is_ok_but_gate_not_satisfied(self) -> None:
        self.dev.unlink()
        self.held.unlink()
        report = self.validator(allow_missing=True).validate()
        self.assertTrue(report.ok)
        self.assertFalse(report.gate_satisfied)
        self.assertEqual(len(report.warnings), 2)

    def test_allow_missing_still_validates_existing_files(self) -> None:
        self.held.unlink()
        self.dev.write_text(_make_set("d", {"single_fact": 20, "multi_hop": 20, "unanswerable": 9}))
        code, out = self.run_main("--allow-missing")
        self.assertEqual(code, 1)
        self.assertIn("WARNING", out)
        self.assertIn("expected exactly 50", out)

    def test_allow_missing_still_rejects_bad_existing_schema(self) -> None:
        self.dev.unlink()
        self.held.write_text("{bad\n")
        self.assertEqual(self.run_main("--allow-missing")[0], 1)


class TestConstructorValidation(_ValidatorCase):
    """Fail-fast on nonsensical thresholds."""

    def test_nonpositive_dev_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.validator(dev_size=0)

    def test_nonpositive_min_heldout_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.validator(min_heldout=0)

    def test_validate_is_repeatable(self) -> None:
        validator = self.validator()
        self.assertEqual(validator.validate(), validator.validate())


if __name__ == "__main__":
    unittest.main()
