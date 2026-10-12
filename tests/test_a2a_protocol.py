"""Unit tests for A2A protocol models — A2AMessage, A2AResponse, A2APeer."""

from __future__ import annotations

import unittest
from datetime import timezone
from unittest.mock import MagicMock

from src.resource.a2a.protocol import A2AMessage, A2APeer, A2AResponse


class TestA2AMessage(unittest.TestCase):
    """Tests for A2AMessage default values and field assignment."""

    def test_defaults_populated(self) -> None:
        """id, trace_id, created_at, and metadata are auto-populated when omitted."""
        msg = A2AMessage(sender="agent-1", payload="hello")
        self.assertTrue(msg.id)
        self.assertTrue(msg.trace_id)
        self.assertIsNotNone(msg.created_at)
        self.assertEqual(msg.metadata, {})

    def test_explicit_fields(self) -> None:
        """Explicitly provided fields are stored unchanged on the instance."""
        msg = A2AMessage(
            sender="agent-1",
            payload="task",
            trace_id="trace-abc",
            metadata={"key": "val"},
        )
        self.assertEqual(msg.sender, "agent-1")
        self.assertEqual(msg.payload, "task")
        self.assertEqual(msg.trace_id, "trace-abc")
        self.assertEqual(msg.metadata["key"], "val")

    def test_unique_ids_per_instance(self) -> None:
        """Each new A2AMessage gets a unique id and trace_id."""
        m1 = A2AMessage(sender="a", payload="x")
        m2 = A2AMessage(sender="a", payload="x")
        self.assertNotEqual(m1.id, m2.id)
        self.assertNotEqual(m1.trace_id, m2.trace_id)

    def test_created_at_is_utc_aware(self) -> None:
        """created_at is timezone-aware and set to UTC."""
        msg = A2AMessage(sender="a", payload="x")
        self.assertEqual(msg.created_at.tzinfo, timezone.utc)


class TestA2AResponse(unittest.TestCase):
    """Tests for A2AResponse field storage and defaults."""

    def test_all_fields_stored(self) -> None:
        """All constructor arguments are stored unchanged and metadata defaults to empty."""
        resp = A2AResponse(
            message_id="msg-1",
            sender="agent-2",
            output="result text",
            trace_id="trace-xyz",
            duration_ms=42.5,
        )
        self.assertEqual(resp.message_id, "msg-1")
        self.assertEqual(resp.sender, "agent-2")
        self.assertEqual(resp.output, "result text")
        self.assertEqual(resp.trace_id, "trace-xyz")
        self.assertAlmostEqual(resp.duration_ms, 42.5)
        self.assertEqual(resp.metadata, {})

    def test_metadata_optional(self) -> None:
        """A custom metadata dict is stored unchanged on the response."""
        resp = A2AResponse(
            message_id="m",
            sender="s",
            output="o",
            trace_id="t",
            duration_ms=1.0,
            metadata={"source": "gemini"},
        )
        self.assertEqual(resp.metadata["source"], "gemini")


class TestA2APeer(unittest.TestCase):
    """Tests for A2APeer field storage and defaults."""

    def test_peer_stores_agent_reference(self) -> None:
        """Peer stores its name, description, and the exact agent instance given."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "mock-id"
        peer = A2APeer(name="fact_checker", description="Checks facts", agent=mock_agent)
        self.assertEqual(peer.name, "fact_checker")
        self.assertEqual(peer.description, "Checks facts")
        self.assertIs(peer.agent, mock_agent)

    def test_description_defaults_empty(self) -> None:
        """description defaults to an empty string when omitted."""
        mock_agent = MagicMock()
        peer = A2APeer(name="reviewer", agent=mock_agent)
        self.assertEqual(peer.description, "")

    def test_peer_name_uniqueness_is_caller_responsibility(self) -> None:
        """Two peers may share the same name; the protocol layer does not enforce uniqueness."""
        # protocol layer does not enforce uniqueness — router does
        mock_a = MagicMock()
        mock_b = MagicMock()
        p1 = A2APeer(name="dup", agent=mock_a)
        p2 = A2APeer(name="dup", agent=mock_b)
        self.assertEqual(p1.name, p2.name)


if __name__ == "__main__":
    unittest.main()
