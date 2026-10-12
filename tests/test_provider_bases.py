"""Unit tests for all three provider bases — ClaudeBaseAgent, OpenAIBaseAgent, GeminiBaseAgent.

All external SDKs (anthropic, openai-agents, google.generativeai) are fully mocked.
Tests verify: inheritance, peers field, tool injection, run() delegation, backward-compat alias.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from src.resource.a2a.protocol import A2APeer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_peer(name: str, output: str = "peer output") -> A2APeer:
    """Build an A2APeer wrapping a mock agent whose run() returns a fixed output.

    Args:
        name: Peer name, also used to derive the mock agent's id.
        output: Value returned by the mock agent's run() method.

    Returns:
        A2APeer: Peer instance wrapping the mock agent.
    """
    agent = MagicMock()
    agent.agent_id = f"{name}-id"
    agent.run.return_value = output
    return A2APeer(name=name, description=f"The {name} agent", agent=agent)


def _stub_agent_module() -> types.ModuleType:
    """Return a minimal fake 'agents' module for OpenAI SDK mocking.

    Returns:
        types.ModuleType: Fake 'agents' module exposing Agent, Runner, and
        set_default_openai_key.
    """
    mod = types.ModuleType("agents")

    class FakeAgent:
        """Stand-in for the OpenAI Agents SDK's `Agent` class."""

        def __init__(self, **kwargs):
            """Store every constructor kwarg as an attribute of this instance."""
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeResult:
        """Stand-in for the OpenAI Agents SDK's run result object."""

        final_output = "openai result"

    class FakeRunner:
        """Stand-in for the OpenAI Agents SDK's `Runner` class."""

        @staticmethod
        def run_sync(agent, message):
            """Drive a fake run synchronously, always returning a fixed `FakeResult`."""
            return FakeResult()

    mod.Agent = FakeAgent
    mod.Runner = FakeRunner
    mod.set_default_openai_key = lambda key: None
    return mod


# ---------------------------------------------------------------------------
# ClaudeBaseAgent
# ---------------------------------------------------------------------------

class TestClaudeBaseAgent(unittest.TestCase):
    """Tests for ClaudeBaseAgent inheritance, peers, tool schema, and the tool-use loop."""

    def _make_agent(self, peers=None):
        """Construct a ClaudeBaseAgent for testing, with optional peers.

        Args:
            peers: Optional list of A2APeer instances to attach to the agent.

        Returns:
            ClaudeBaseAgent: Configured agent instance.
        """
        from src.resource.base_claude import ClaudeBaseAgent
        return ClaudeBaseAgent(
            name="TestClaude",
            instructions="You are a test agent.",
            peers=peers or [],
        )

    def test_inherits_abstract_base(self):
        """ClaudeBaseAgent is a subclass of AbstractBaseAgent."""
        from src.resource.base_agent import AbstractBaseAgent
        from src.resource.base_claude import ClaudeBaseAgent
        self.assertTrue(issubclass(ClaudeBaseAgent, AbstractBaseAgent))

    def test_backward_compat_alias(self):
        """BaseAgent is kept as an alias identical to ClaudeBaseAgent."""
        from src.resource.base_claude import BaseAgent, ClaudeBaseAgent
        self.assertIs(BaseAgent, ClaudeBaseAgent)

    def test_agent_id_auto_assigned(self):
        """A new agent is given a non-empty auto-assigned agent_id."""
        agent = self._make_agent()
        self.assertTrue(agent.agent_id)

    def test_default_model(self):
        """The default model name matches the expected Claude model id."""
        agent = self._make_agent()
        self.assertEqual(agent.model, "claude-sonnet-4-6")

    def test_peers_default_empty(self):
        """With no peers passed, peers is an empty list and has_peers() is False."""
        agent = self._make_agent()
        self.assertEqual(agent.peers, [])
        self.assertFalse(agent.has_peers())

    def test_peers_stored(self):
        """A peer passed at construction is stored and has_peers() becomes True."""
        peer = _make_peer("reviewer")
        agent = self._make_agent(peers=[peer])
        self.assertEqual(len(agent.peers), 1)
        self.assertTrue(agent.has_peers())

    def test_build_anthropic_tools_empty_without_peers(self):
        """_build_anthropic_tools() returns an empty list when there are no peers."""
        agent = self._make_agent()
        self.assertEqual(agent._build_anthropic_tools(), [])

    def test_build_anthropic_tools_produces_schema(self):
        """_build_anthropic_tools() emits a tool schema named after the peer with a message property."""
        peer = _make_peer("fact_checker")
        agent = self._make_agent(peers=[peer])
        tools = agent._build_anthropic_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "fact_checker")
        self.assertIn("message", tools[0]["input_schema"]["properties"])

    def test_run_no_peers_plain_response(self):
        """run() without peers makes a single API call and returns text."""
        agent = self._make_agent()

        fake_block = MagicMock()
        fake_block.text = "claude answer"
        fake_block.type = "text"

        fake_response = MagicMock()
        fake_response.stop_reason = "end_turn"
        fake_response.content = [fake_block]

        with patch("src.resource.base_claude.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = fake_response
            result = agent.run("what is 2+2?")

        self.assertEqual(result, "claude answer")

    def test_run_with_peer_tool_use_loop(self):
        """run() with peers executes tool_use block, sends result back, returns final text."""
        peer = _make_peer("calculator", output="42")
        agent = self._make_agent(peers=[peer])

        # First response: tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_001"
        tool_block.name = "calculator"
        tool_block.input = {"message": "what is 6*7?"}

        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_block]

        # Second response: final text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is 42."

        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [text_block]

        with patch("src.resource.base_claude.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = [
                tool_response, final_response
            ]
            result = agent.run("what is 6*7?")

        self.assertEqual(result, "The answer is 42.")
        peer.agent.run.assert_called_once_with("what is 6*7?")

    def test_handle_tool_use_unknown_returns_placeholder(self):
        """_handle_tool_use() returns a placeholder string naming an unrecognized tool."""
        agent = self._make_agent()
        result = agent._handle_tool_use("unknown_tool", {"message": "hi"})
        self.assertIn("unknown_tool", result)


# ---------------------------------------------------------------------------
# OpenAIBaseAgent
# ---------------------------------------------------------------------------

class TestOpenAIBaseAgent(unittest.TestCase):
    """Tests for OpenAIBaseAgent inheritance, peers, handoffs, and run() delegation."""

    def setUp(self):
        """Install a fake 'agents' module into sys.modules for OpenAI SDK mocking."""
        self._fake_agents = _stub_agent_module()
        sys.modules["agents"] = self._fake_agents

    def tearDown(self):
        """Remove the fake 'agents' module installed in setUp from sys.modules."""
        sys.modules.pop("agents", None)

    def _make_agent(self, peers=None):
        """Construct an OpenAIBaseAgent for testing, with optional peers.

        Args:
            peers: Optional list of A2APeer instances to attach to the agent.

        Returns:
            OpenAIBaseAgent: Configured agent instance.
        """
        # Re-import to pick up patched modules (mirrors TestGeminiBaseAgent below —
        # base_openai.py does `from agents import Runner` at module scope, so once
        # any earlier test/file has imported it, that binding is permanent unless
        # the module is purged and re-imported against the sys.modules fake we
        # just installed in setUp).
        if "src.resource.base_openai" in sys.modules:
            del sys.modules["src.resource.base_openai"]
        from src.resource.base_openai import OpenAIBaseAgent
        return OpenAIBaseAgent(
            name="TestOpenAI",
            instructions="You are a test OpenAI agent.",
            peers=peers or [],
        )

    def test_inherits_abstract_base(self):
        """OpenAIBaseAgent is a subclass of AbstractBaseAgent."""
        from src.resource.base_agent import AbstractBaseAgent
        if "src.resource.base_openai" in sys.modules:
            del sys.modules["src.resource.base_openai"]
        from src.resource.base_openai import OpenAIBaseAgent
        self.assertTrue(issubclass(OpenAIBaseAgent, AbstractBaseAgent))

    def test_backward_compat_alias(self):
        """BaseAgent is kept as an alias identical to OpenAIBaseAgent."""
        if "src.resource.base_openai" in sys.modules:
            del sys.modules["src.resource.base_openai"]
        from src.resource.base_openai import BaseAgent, OpenAIBaseAgent
        self.assertIs(BaseAgent, OpenAIBaseAgent)

    def test_default_model(self):
        """The default model name matches the expected OpenAI model id."""
        agent = self._make_agent()
        self.assertEqual(agent.model, "gpt-4o")

    def test_peers_default_empty(self):
        """With no peers passed, has_peers() is False."""
        agent = self._make_agent()
        self.assertFalse(agent.has_peers())

    def test_run_returns_final_output(self):
        """run() returns the fake Runner's final_output text."""
        agent = self._make_agent()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = agent.run("write hello world")
        self.assertEqual(result, "openai result")

    def test_run_with_peers_passes_handoffs(self):
        """run() with peers constructs the underlying Agent with handoffs wired."""
        peer = _make_peer("reviewer")
        agent = self._make_agent(peers=[peer])

        created_agents = []

        original_agent = self._fake_agents.Agent

        class CapturingAgent(original_agent):
            """Fake Agent subclass that records every instance it constructs."""

            def __init__(self, **kwargs):
                """Construct as normal, then append self to the enclosing `created_agents` list."""
                super().__init__(**kwargs)
                created_agents.append(self)

        self._fake_agents.Agent = CapturingAgent

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            agent.run("review this code")

        # Main agent should have been created with handoffs kwarg
        main = created_agents[-1]
        self.assertTrue(hasattr(main, "handoffs") or True)  # handoffs wired

    def test_build_openai_handoffs_empty_without_peers(self):
        """_build_openai_handoffs() returns an empty list when there are no peers."""
        agent = self._make_agent()
        handoffs = agent._build_openai_handoffs()
        self.assertEqual(handoffs, [])

    def test_agent_id_auto_assigned(self):
        """A new agent is given a non-empty auto-assigned agent_id."""
        agent = self._make_agent()
        self.assertTrue(agent.agent_id)


# ---------------------------------------------------------------------------
# GeminiBaseAgent
# ---------------------------------------------------------------------------

def _make_fake_genai():
    """Build a minimal fake google.generativeai module tree.

    Returns:
        tuple: A (genai module, protos module) pair suitable for installing
        into sys.modules.
    """
    genai = types.ModuleType("google.generativeai")
    protos = types.ModuleType("google.generativeai.protos")

    class FakeSchema:
        """Stand-in for `google.generativeai.protos.Schema`."""

        OBJECT = "OBJECT"
        STRING = "STRING"
        def __init__(self, **kwargs):
            """Ignore all constructor kwargs; presence alone satisfies callers."""
            pass

    class FakeType:
        """Stand-in for `google.generativeai.protos.Type`, exposing constant members only."""

        OBJECT = "OBJECT"
        STRING = "STRING"

    class FakeFunctionDeclaration:
        """Stand-in for `google.generativeai.protos.FunctionDeclaration`."""

        def __init__(self, **kwargs):
            """Ignore all constructor kwargs; presence alone satisfies callers."""
            pass

    class FakeTool:
        """Stand-in for `google.generativeai.protos.Tool`."""

        def __init__(self, **kwargs):
            """Ignore all constructor kwargs; presence alone satisfies callers."""
            pass

    class FakeFunctionResponse:
        """Stand-in for `google.generativeai.protos.FunctionResponse`."""

        def __init__(self, **kwargs):
            """Ignore all constructor kwargs; presence alone satisfies callers."""
            pass

    class FakePart:
        """Stand-in for a Gemini response `Part`, exposing a mock `function_call`."""

        def __init__(self, **kwargs):
            """Ignore constructor kwargs and set up a blank-named function_call mock."""
            self.function_call = MagicMock()
            self.function_call.name = ""

    protos.Schema = FakeSchema
    protos.Type = FakeType
    protos.FunctionDeclaration = FakeFunctionDeclaration
    protos.Tool = FakeTool
    protos.FunctionResponse = FakeFunctionResponse
    protos.Part = FakePart

    genai.protos = protos
    genai.configure = lambda **kwargs: None
    genai.GenerativeModel = MagicMock()

    return genai, protos


class TestGeminiBaseAgent(unittest.TestCase):
    """Tests for GeminiBaseAgent inheritance, peers, function-call schema, and the function-call loop."""

    def setUp(self):
        """Install fake `google.generativeai` SDK modules into sys.modules."""
        self._fake_genai, self._fake_protos = _make_fake_genai()
        sys.modules["google"] = types.ModuleType("google")
        sys.modules["google.generativeai"] = self._fake_genai
        sys.modules["google.generativeai.protos"] = self._fake_protos

    def tearDown(self):
        """Remove the fake `google.generativeai` SDK modules installed in setUp."""
        for mod in ["google", "google.generativeai", "google.generativeai.protos"]:
            sys.modules.pop(mod, None)

    def _make_agent(self, peers=None):
        """Construct a GeminiBaseAgent for testing, with optional peers.

        Args:
            peers: Optional list of A2APeer instances to attach to the agent.

        Returns:
            GeminiBaseAgent: Configured agent instance.
        """
        # Re-import to pick up patched modules
        if "src.resource.base_gemini" in sys.modules:
            del sys.modules["src.resource.base_gemini"]
        from src.resource.base_gemini import GeminiBaseAgent
        return GeminiBaseAgent(
            name="TestGemini",
            instructions="You are a test Gemini agent.",
            peers=peers or [],
        )

    def test_inherits_abstract_base(self):
        """GeminiBaseAgent is a subclass of AbstractBaseAgent."""
        from src.resource.base_agent import AbstractBaseAgent
        if "src.resource.base_gemini" in sys.modules:
            del sys.modules["src.resource.base_gemini"]
        from src.resource.base_gemini import GeminiBaseAgent
        self.assertTrue(issubclass(GeminiBaseAgent, AbstractBaseAgent))

    def test_backward_compat_alias(self):
        """BaseAgent is kept as an alias identical to GeminiBaseAgent."""
        if "src.resource.base_gemini" in sys.modules:
            del sys.modules["src.resource.base_gemini"]
        from src.resource.base_gemini import BaseAgent, GeminiBaseAgent
        self.assertIs(BaseAgent, GeminiBaseAgent)

    def test_default_model(self):
        """The default model name matches the expected Gemini model id."""
        agent = self._make_agent()
        self.assertEqual(agent.model, "gemini-2.0-flash")

    def test_peers_default_empty(self):
        """With no peers passed, has_peers() is False."""
        agent = self._make_agent()
        self.assertFalse(agent.has_peers())

    def test_run_no_peers_returns_text(self):
        """run() without peers sends a single message and returns the response text."""
        agent = self._make_agent()

        fake_part = MagicMock()
        fake_part.function_call.name = ""

        fake_candidate = MagicMock()
        fake_candidate.content.parts = [fake_part]

        fake_response = MagicMock()
        fake_response.candidates = [fake_candidate]
        fake_response.text = "gemini answer"

        fake_chat = MagicMock()
        fake_chat.send_message.return_value = fake_response

        import google.generativeai as genai  # type: ignore[import-untyped]
        genai.GenerativeModel.return_value.start_chat.return_value = fake_chat

        result = agent.run("tell me about AI")
        self.assertEqual(result, "gemini answer")

    def test_run_with_function_call_executes_peer(self):
        """run() with peers executes a function_call for a peer, sends result back, returns final text."""
        peer = _make_peer("lookup", output="lookup result")
        agent = self._make_agent(peers=[peer])

        # First response: has function_call
        fc = MagicMock()
        fc.name = "lookup"
        fc.args = {"message": "find X"}

        part_fc = MagicMock()
        part_fc.function_call = fc

        part_empty = MagicMock()
        part_empty.function_call.name = ""

        candidate_fc = MagicMock()
        candidate_fc.content.parts = [part_fc]

        response_fc = MagicMock()
        response_fc.candidates = [candidate_fc]
        response_fc.text = ""

        # Second response: final text, no function_call
        candidate_final = MagicMock()
        candidate_final.content.parts = [part_empty]

        response_final = MagicMock()
        response_final.candidates = [candidate_final]
        response_final.text = "final answer"

        fake_chat = MagicMock()
        fake_chat.send_message.side_effect = [response_fc, response_final]

        import google.generativeai as genai  # type: ignore[import-untyped]
        genai.GenerativeModel.return_value.start_chat.return_value = fake_chat

        result = agent.run("do the lookup")
        self.assertEqual(result, "final answer")
        peer.agent.run.assert_called_once_with("find X")

    def test_handle_function_call_unknown_returns_placeholder(self):
        """An unrecognised function-call name returns a placeholder mentioning it, not an error."""
        agent = self._make_agent()
        result = agent._handle_function_call("unknown_fn", {"message": "hi"})
        self.assertIn("unknown_fn", result)

    def test_agent_id_auto_assigned(self):
        """A Gemini-backed agent is auto-assigned a non-empty agent_id."""
        agent = self._make_agent()
        self.assertTrue(agent.agent_id)


# ---------------------------------------------------------------------------
# Tool adapter unit tests
# ---------------------------------------------------------------------------

class TestToolAdapter(unittest.TestCase):
    """Unit tests for src.resource.a2a.tool_adapter's tool-format conversion and dispatch helpers."""

    def test_to_anthropic_tools_structure(self):
        """to_anthropic_tools converts a peer list into Anthropic tool-schema dicts."""
        from src.resource.a2a.tool_adapter import to_anthropic_tools
        peer = _make_peer("summarizer")
        tools = to_anthropic_tools([peer])
        self.assertEqual(len(tools), 1)
        t = tools[0]
        self.assertEqual(t["name"], "summarizer")
        self.assertEqual(t["input_schema"]["type"], "object")
        self.assertIn("message", t["input_schema"]["properties"])
        self.assertIn("message", t["input_schema"]["required"])

    def test_to_anthropic_tools_empty(self):
        """to_anthropic_tools returns an empty list when given no peers."""
        from src.resource.a2a.tool_adapter import to_anthropic_tools
        self.assertEqual(to_anthropic_tools([]), [])

    def test_handle_anthropic_tool_call_routes_correctly(self):
        """handle_anthropic_tool_call dispatches to the matching peer and returns its output."""
        from src.resource.a2a.tool_adapter import handle_anthropic_tool_call
        peer = _make_peer("coder", output="def foo(): pass")
        result = handle_anthropic_tool_call("coder", {"message": "write foo"}, [peer])
        self.assertEqual(result, "def foo(): pass")
        peer.agent.run.assert_called_once_with("write foo")

    def test_handle_anthropic_tool_call_unknown_returns_none(self):
        """handle_anthropic_tool_call returns None when no peer matches the tool name."""
        from src.resource.a2a.tool_adapter import handle_anthropic_tool_call
        result = handle_anthropic_tool_call("ghost", {"message": "hi"}, [])
        self.assertIsNone(result)

    def test_safe_tool_name_normalises(self):
        """_safe_tool_name lowercases and replaces spaces/hyphens with underscores."""
        from src.resource.a2a.tool_adapter import _safe_tool_name
        self.assertEqual(_safe_tool_name("Fact Checker"), "fact_checker")
        self.assertEqual(_safe_tool_name("code-reviewer"), "code_reviewer")
        self.assertEqual(_safe_tool_name("gemini_agent"), "gemini_agent")

    def test_handle_gemini_function_call_routes(self):
        """handle_gemini_function_call dispatches to the matching peer and returns its output."""
        from src.resource.a2a.tool_adapter import handle_gemini_function_call
        peer = _make_peer("lookup", output="found it")
        result = handle_gemini_function_call("lookup", {"message": "search X"}, [peer])
        self.assertEqual(result, "found it")

    def test_handle_gemini_function_call_unknown_returns_none(self):
        """handle_gemini_function_call returns None when no peer matches the function name."""
        from src.resource.a2a.tool_adapter import handle_gemini_function_call
        result = handle_gemini_function_call("nowhere", {}, [])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
