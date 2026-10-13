"""Pure retrieval-eval metrics: recall@k, MRR, abstention accuracy and a citation-validity stub.

Definitions are fixed in requirements.md ("Metric definitions"); do not redefine them here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from evals.schema import EvalQuestion, QuestionType
from rag_core.exceptions import ValidationError

DEFAULT_K = 5


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValidationError(f"k must be positive, got {k}")


def _check_relevant(relevant_doc_ids: Collection[str]) -> frozenset[str]:
    relevant = frozenset(relevant_doc_ids)
    if not relevant:
        raise ValidationError("relevant_doc_ids must not be empty for an answerable question")
    return relevant


def dedupe_preserving_rank(ranked_doc_ids: Iterable[str]) -> list[str]:
    """Drop repeated doc ids (duplicate chunks of one document), keeping first-seen order."""
    return list(dict.fromkeys(ranked_doc_ids))


def _top_k_unique(ranked_doc_ids: Iterable[str], k: int) -> list[str]:
    # Dedupe BEFORE truncating, so duplicate chunks never consume top-k slots.
    return dedupe_preserving_rank(ranked_doc_ids)[:k]


def recall_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int = DEFAULT_K
) -> float:
    """Fraction of relevant doc ids present in the top-k de-duplicated ranking."""
    _check_k(k)
    relevant = _check_relevant(relevant_doc_ids)
    found = relevant.intersection(_top_k_unique(ranked_doc_ids, k))
    return len(found) / len(relevant)


def reciprocal_rank(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int = DEFAULT_K
) -> float:
    """1 / rank of the first relevant doc in the de-duplicated top k; 0.0 if none."""
    _check_k(k)
    relevant = _check_relevant(relevant_doc_ids)
    for rank, doc_id in enumerate(_top_k_unique(ranked_doc_ids, k), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ValidationError(f"cannot compute {label} over zero questions")
    return sum(values) / len(values)


def mean_recall(per_question_recall: Sequence[float]) -> float:
    """Average of per-question recall values (answerable questions only)."""
    return _mean(per_question_recall, "mean recall")


def mrr(per_question_rr: Sequence[float]) -> float:
    """Mean reciprocal rank over per-question reciprocal ranks."""
    return _mean(per_question_rr, "MRR")


def abstention_correct(top_score: float | None, threshold: float) -> bool:
    """True when nothing clears the threshold (no result, or top score below it)."""
    return top_score is None or top_score < threshold


def abstention_accuracy(outcomes: Sequence[bool]) -> float:
    """Fraction of unanswerable questions answered with a correct abstention."""
    return _mean([1.0 if ok else 0.0 for ok in outcomes], "abstention accuracy")


@dataclass(frozen=True)
class QuestionResult:
    """Retrieval output for one question: ranked doc ids (chunks mapped to docs) and top score."""

    question_id: str
    qtype: QuestionType
    ranked_doc_ids: tuple[str, ...] = field(default_factory=tuple)
    top_score: float | None = None


@dataclass(frozen=True)
class TypeMetrics:
    """Recall and MRR for the answerable questions of one type."""

    qtype: QuestionType
    count: int
    recall: float
    mrr: float


@dataclass(frozen=True)
class MetricsReport:
    """Overall and per-type metrics; unanswerable questions only feed abstention."""

    k: int
    answerable_count: int
    unanswerable_count: int
    recall: float | None
    mrr: float | None
    abstention_accuracy: float | None
    by_type: tuple[TypeMetrics, ...] = field(default_factory=tuple)


def _index_results(
    questions: Sequence[EvalQuestion], results: Sequence[QuestionResult]
) -> dict[str, QuestionResult]:
    dupes = sorted(qid for qid, n in Counter(r.question_id for r in results).items() if n > 1)
    if dupes:
        raise ValidationError(f"duplicate results for questions: {dupes}")
    by_id = {r.question_id: r for r in results}
    missing = sorted(q.question_id for q in questions if q.question_id not in by_id)
    if missing:
        raise ValidationError(f"missing results for questions: {missing}")
    return by_id


def _check_unanswerable(question: EvalQuestion) -> None:
    if question.qtype is QuestionType.UNANSWERABLE and question.relevant_doc_ids:
        raise ValidationError(f"unanswerable question {question.question_id} has relevant ids")


def _type_metrics(qtype: QuestionType, scores: list[tuple[float, float]]) -> TypeMetrics:
    return TypeMetrics(
        qtype=qtype,
        count=len(scores),
        recall=mean_recall([s[0] for s in scores]),
        mrr=mrr([s[1] for s in scores]),
    )


def _by_type(scores: dict[QuestionType, list[tuple[float, float]]]) -> tuple[TypeMetrics, ...]:
    ordered = sorted(scores, key=lambda t: t.value)
    return tuple(_type_metrics(qtype, scores[qtype]) for qtype in ordered)


def _optional_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    questions: Sequence[EvalQuestion],
    results: Sequence[QuestionResult],
    abstention_threshold: float,
    k: int = DEFAULT_K,
) -> MetricsReport:
    """Score every question and return overall and per-type metrics, deterministically."""
    _check_k(k)
    by_id = _index_results(questions, results)
    scores: dict[QuestionType, list[tuple[float, float]]] = {}
    abstentions: list[bool] = []
    for question in questions:
        _check_unanswerable(question)
        result = by_id[question.question_id]
        if question.qtype is QuestionType.UNANSWERABLE:
            abstentions.append(abstention_correct(result.top_score, abstention_threshold))
            continue
        ranked, relevant = result.ranked_doc_ids, question.relevant_doc_ids
        scores.setdefault(question.qtype, []).append(
            (recall_at_k(ranked, relevant, k), reciprocal_rank(ranked, relevant, k))
        )
    flat = [pair for pairs in scores.values() for pair in pairs]
    return MetricsReport(
        k=k,
        answerable_count=len(flat),
        unanswerable_count=len(abstentions),
        recall=_optional_mean([p[0] for p in flat]),
        mrr=_optional_mean([p[1] for p in flat]),
        abstention_accuracy=abstention_accuracy(abstentions) if abstentions else None,
        by_type=_by_type(scores),
    )


class CitationCondition(StrEnum):
    """The four claim-level citation conditions, in evaluation order."""

    EXISTS = "exists"
    AUTHORIZED = "authorized"
    VERSION_MATCHES = "version_matches"
    SUPPORTS = "supports"


@dataclass(frozen=True)
class CitationCheck:
    """Outcome of a citation check; `failed` names the first failed condition, if any."""

    citation: str
    failed: CitationCondition | None = None

    @property
    def valid(self) -> bool:
        """True when all four conditions held."""
        return self.failed is None


def citation_valid(
    citation: str,
    *,
    exists: Callable[[str], bool],
    authorized: Callable[[str], bool],
    version_matches: Callable[[str], bool],
    supports: Callable[[str], bool],
) -> CitationCheck:
    """Stub: apply injected lookups in order, short-circuiting on the first failure."""
    if not citation.strip():
        raise ValidationError("citation must not be empty")
    checks = (
        (CitationCondition.EXISTS, exists),
        (CitationCondition.AUTHORIZED, authorized),
        (CitationCondition.VERSION_MATCHES, version_matches),
        (CitationCondition.SUPPORTS, supports),
    )
    for condition, check in checks:
        if not check(citation):
            return CitationCheck(citation=citation, failed=condition)
    return CitationCheck(citation=citation)
