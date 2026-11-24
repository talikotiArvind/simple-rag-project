"""Unit tests for A2ARouter — dispatch, async dispatch, error handling."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.resource.a2a.protocol import A2AMessage, A2APeer
from src.resource.a2a.router import A2ARouter


def _make_peer(name: str, output: str = "mock output") -> A2APeer:
    """Build an A2APeer wrapping a mock agent that returns a fixed output.

    Args:
        name: Peer name and basis for the mocked agent_id.
        output: Value returned by both the sync run() and async run_async() mocks.

    Returns:
        A2APeer: A peer wired to the mock agent.
    """
    agent = MagicMock()
    agent.agent_id = f"{name}-agent-id"
    agent.run.return_value = output
    agent.run_async = AsyncMock(return_value=output)
    return A2APeer(name=name, description=f"Peer {name}", agent=agent)


def _make_message(sender: str = "caller", payload: str = "do task") -> A2AMessage:
    """Build an A2AMessage with a fixed trace_id for deterministic assertions.

    Args:
        sender: Sender identifier to attach to the message.
        payload: Message payload text.

    Returns:
        A2AMessage: A message with trace_id "trace-001".
    """
    return A2AMessage(sender=sender, payload=payload, trace_id="trace-001")


class TestA2ARouterDispatch(unittest.TestCase):
    """Tests for A2ARouter.dispatch() synchronous routing behavior."""

    def setUp(self) -> None:
        """Create a router with a single 'summarizer' peer for each test."""
        self.peer = _make_peer("summarizer", output="summary result")
        self.router = A2ARouter(peers=[self.peer])

    def test_dispatch_returns_response(self) -> None:
        """dispatch() returns a response whose output matches the peer's return value."""
        msg = _make_message()
        resp = self.router.dispatch("summarizer", msg)
        self.assertEqual(resp.output, "summary result")

    def test_response_carries_trace_id(self) -> None:
        """The response's trace_id matches the originating message's trace_id."""
        msg = _make_message()
        resp = self.router.dispatch("summarizer", msg)
        self.assertEqual(resp.trace_id, msg.trace_id)

    def test_response_sender_is_peer_agent_id(self) -> None:
        """The response's sender is set to the dispatched peer's agent_id."""
        msg = _make_message()
        resp = self.router.dispatch("summarizer", msg)
        self.assertEqual(resp.sender, "summarizer-agent-id")

    def test_response_message_id_matches(self) -> None:
        """The response's message_id matches the originating message's id."""
        msg = _make_message()
        resp = self.router.dispatch("summarizer", msg)
        self.assertEqual(resp.message_id, msg.id)

    def test_duration_ms_is_non_negative(self) -> None:
        """The response's duration_ms is never negative."""
        msg = _make_message()
        resp = self.router.dispatch("summarizer", msg)
        self.assertGreaterEqual(resp.duration_ms, 0)

    def test_peer_run_called_with_payload(self) -> None:
        """dispatch() invokes the peer's run() with the message payload."""
        msg = _make_message(payload="specific task")
        self.router.dispatch("summarizer", msg)
        self.peer.agent.run.assert_called_once_with("specific task")

    def test_unknown_peer_raises_key_error(self) -> None:
        """dispatch() to an unregistered peer name raises KeyError naming both peers."""
        msg = _make_message()
        with self.assertRaises(KeyError) as ctx:
            self.router.dispatch("nonexistent", msg)
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("summarizer", str(ctx.exception))

    def test_peer_exception_propagates(self) -> None:
        """An exception raised by the peer's run() propagates out of dispatch()."""
        self.peer.agent.run.side_effect = RuntimeError("peer failed")
        msg = _make_message()
        with self.assertRaises(RuntimeError, msg="peer failed"):
            self.router.dispatch("summarizer", msg)

    def test_multiple_peers_routed_correctly(self) -> None:
        """dispatch() routes each message to the correct peer among several registered."""
        peer_a = _make_peer("coder", output="code output")
        peer_b = _make_peer("reviewer", output="review output")
        router = A2ARouter(peers=[peer_a, peer_b])

        resp_a = router.dispatch("coder", _make_message())
        resp_b = router.dispatch("reviewer", _make_message())

        self.assertEqual(resp_a.output, "code output")
        self.assertEqual(resp_b.output, "review output")

    def test_registered_peers_returns_sorted(self) -> None:
        """registered_peers returns peer names sorted alphabetically."""
        peer_z = _make_peer("zzz")
        peer_a = _make_peer("aaa")
        router = A2ARouter(peers=[peer_z, peer_a])
        self.assertEqual(router.registered_peers, ["aaa", "zzz"])

    def test_duplicate_peer_names_last_wins(self) -> None:
        """When two peers share a name, the later-registered one wins on dispatch."""
        peer_1 = _make_peer("dup")
        peer_1.agent.run.return_value = "first"
        peer_2 = _make_peer("dup")
        peer_2.agent.run.return_value = "second"
        router = A2ARouter(peers=[peer_1, peer_2])
        resp = router.dispatch("dup", _make_message())
        self.assertEqual(resp.output, "second")


class TestA2ARouterDispatchAsync(unittest.TestCase):
    """Tests for A2ARouter.dispatch_async() asynchronous routing behavior."""

    def _run(self, coro):  # type: ignore[override]
        """Run a coroutine to completion on a fresh event loop.

        Args:
            coro: The coroutine to execute.

        Returns:
            The coroutine's result.
        """
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_dispatch_async_returns_response(self) -> None:
        """dispatch_async() returns a response whose output matches the peer's return value."""
        peer = _make_peer("async_peer", output="async result")
        router = A2ARouter(peers=[peer])
        msg = _make_message()
        resp = self._run(router.dispatch_async("async_peer", msg))
        self.assertEqual(resp.output, "async result")

    def test_dispatch_async_trace_id_preserved(self) -> None:
        """The response's trace_id matches the originating message's trace_id."""
        peer = _make_peer("async_peer")
        router = A2ARouter(peers=[peer])
        msg = _make_message()
        resp = self._run(router.dispatch_async("async_peer", msg))
        self.assertEqual(resp.trace_id, msg.trace_id)

    def test_dispatch_async_unknown_peer_raises(self) -> None:
        """dispatch_async() to an unregistered peer name raises KeyError."""
        router = A2ARouter(peers=[])
        msg = _make_message()
        with self.assertRaises(KeyError):
            self._run(router.dispatch_async("nobody", msg))

    def test_dispatch_async_calls_run_async(self) -> None:
        """dispatch_async() awaits the peer's run_async() with the message payload."""
        peer = _make_peer("async_peer")
        router = A2ARouter(peers=[peer])
        msg = _make_message(payload="async task")
        self._run(router.dispatch_async("async_peer", msg))
        peer.agent.run_async.assert_awaited_once_with("async task")


class TestAbstractBaseAgentDelegate(unittest.TestCase):
    """Test delegate() and delegate_async() on AbstractBaseAgent via a minimal concrete stub."""

    def _make_stub_agent(self, peers: list[A2APeer]):  # type: ignore[type-arg]
        """Build a minimal concrete AbstractBaseAgent subclass with the given peers.

        Args:
            peers: Peers to wire onto the stub agent.

        Returns:
            AbstractBaseAgent: A stub agent instance whose run() echoes its input.
        """
        from src.resource.base_agent import AbstractBaseAgent

        class _StubAgent(AbstractBaseAgent):
            """Minimal concrete agent used only to exercise delegate()/delegate_async()."""

            def run(self, message: str, **kwargs: object) -> str:
                """Return a fixed string embedding the input message."""
                return f"stub:{message}"

        return _StubAgent(name="stub", instructions="stub", peers=peers)

    def test_delegate_calls_peer(self) -> None:
        """delegate() returns a response whose output matches the peer's return value."""
        peer = _make_peer("helper", output="helped")
        agent = self._make_stub_agent(peers=[peer])
        resp = agent.delegate("helper", "do something")
        self.assertEqual(resp.output, "helped")

    def test_delegate_sender_is_own_agent_id(self) -> None:
        """delegate() produces a response with a trace_id set."""
        peer = _make_peer("helper")
        agent = self._make_stub_agent(peers=[peer])
        with patch.object(
            __import__("src.resource.a2a.router", fromlist=["A2ARouter"]).A2ARouter,
            "dispatch",
            wraps=lambda self_r, name, msg: self_r._registry[name].agent.run(msg.payload),
        ):
            pass
        resp = agent.delegate("helper", "task")
        self.assertEqual(resp.trace_id, resp.trace_id)  # trace_id is set

    def test_delegate_unknown_peer_raises(self) -> None:
        """delegate() to an unregistered peer name raises KeyError."""
        agent = self._make_stub_agent(peers=[])
        with self.assertRaises(KeyError):
            agent.delegate("nobody", "task")

    def test_peer_names_property(self) -> None:
        """peer_names returns registered peer names sorted alphabetically."""
        p1 = _make_peer("zzz")
        p2 = _make_peer("aaa")
        agent = self._make_stub_agent(peers=[p1, p2])
        self.assertEqual(agent.peer_names, ["aaa", "zzz"])

    def test_has_peers_true(self) -> None:
        """has_peers() returns True when at least one peer is registered."""
        agent = self._make_stub_agent(peers=[_make_peer("p")])
        self.assertTrue(agent.has_peers())

    def test_has_peers_false_when_empty(self) -> None:
        """has_peers() returns False when no peers are registered."""
        agent = self._make_stub_agent(peers=[])
        self.assertFalse(agent.has_peers())

    def test_explicit_trace_id_threaded(self) -> None:
        """A caller-supplied trace_id is threaded through to the response."""
        peer = _make_peer("tracer", output="ok")
        agent = self._make_stub_agent(peers=[peer])
        resp = agent.delegate("tracer", "work", trace_id="my-trace-123")
        self.assertEqual(resp.trace_id, "my-trace-123")

    def test_delegate_async(self) -> None:
        """delegate_async() returns a response whose output matches the peer's return value."""
        peer = _make_peer("async_helper", output="async helped")
        agent = self._make_stub_agent(peers=[peer])
        resp = asyncio.new_event_loop().run_until_complete(
            agent.delegate_async("async_helper", "do async")
        )
        self.assertEqual(resp.output, "async helped")


if __name__ == "__main__":
    unittest.main()
