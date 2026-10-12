"""Returned-text envelope (NFR-03, R-17..R-19): retrieved text is data, never instructions."""
from __future__ import annotations

import re
from typing import Literal

from rag_core.exceptions import ValidationError

EnvelopeType = Literal["chunk", "concept", "document"]

_ID_PATTERN = re.compile(r"[A-Za-z0-9._:/-]+")
_TAG = "retrieved_content"


def escape_body(text: str) -> str:
    """Escape ``&`` first, then neutralise any opening or closing envelope tag (R-18)."""
    escaped = text.replace("&", "&amp;")
    escaped = escaped.replace(f"<{_TAG}", f"&lt;{_TAG}")
    return escaped.replace(f"</{_TAG}", f"&lt;/{_TAG}")


def validate_envelope_id(result_id: str) -> None:
    """Reject IDs that cannot be an envelope ``id`` attribute (R-17)."""
    if not _ID_PATTERN.fullmatch(result_id):
        raise ValidationError(f"id not allowed in envelope attribute: {result_id!r}")


def wrap_retrieved(result_id: str, result_type: EnvelopeType, text: str) -> str:
    """Return ``text`` escaped and wrapped in the single ``<retrieved_content>`` envelope."""
    validate_envelope_id(result_id)
    return f'<{_TAG} id="{result_id}" type="{result_type}">\n{escape_body(text)}\n</{_TAG}>'
