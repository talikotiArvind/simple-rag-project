# Code Index

Auto-updated by `.claude/hooks/update_code_index.py` when source files are edited.

**AGENTS: Read this file first before reading any source file.**
Use it to understand what exists before loading full files.
Only load a full source file when you need to inspect specific logic.
Keys are repo-relative paths.

---

<!-- Entries are auto-inserted below by the hook -->

<!-- BEGIN: src/__init__.py -->
## src/__init__.py
*Updated: 2026-10-12 14:29*

**Purpose**: Top-level package for the multi-provider agent application.
<!-- END: src/__init__.py -->

<!-- BEGIN: src/resource/__init__.py -->
## src/resource/__init__.py
*Updated: 2026-10-12 14:29*

**Purpose**: Resource layer: provider base classes and A2A infrastructure.
<!-- END: src/resource/__init__.py -->

<!-- BEGIN: src/resource/a2a/__init__.py -->
## src/resource/a2a/__init__.py
*Updated: 2026-10-12 14:29*

**Purpose**: A2A (Agent-to-Agent) communication layer.
<!-- END: src/resource/a2a/__init__.py -->

<!-- BEGIN: src/resource/a2a/protocol.py -->
## src/resource/a2a/protocol.py
*Updated: 2026-10-12 14:29*

**Purpose**: Typed protocol models for A2A (Agent-to-Agent) communication.
**Classes**:
  - `A2AMessage(BaseModel)`: Envelope sent from one agent to a peer agent.
  - `A2AResponse(BaseModel)`: Response returned by a peer agent after handling an A2AMessage.
  - `A2APeer(BaseModel)`: Declares a peer agent reachable by name via delegate().
<!-- END: src/resource/a2a/protocol.py -->

<!-- BEGIN: src/resource/a2a/router.py -->
## src/resource/a2a/router.py
*Updated: 2026-10-12 14:29*

**Purpose**: A2ARouter — peer-local dispatch with trace logging and error boundary.
**Classes**:
  - `A2ARouter`: Dispatch A2AMessages to registered peer agents.
    - `__init__(self, peers: list[A2APeer]) -> None`: Initialize the router with a peer-local registry.
    - `dispatch(self, peer_name: str, message: A2AMessage) -> A2AResponse`: Route a message to the named peer and return its response.
    - `async dispatch_async(self, peer_name: str, message: A2AMessage) -> A2AResponse`: Route a message to the named peer asynchronously.
    - `_resolve(self, peer_name: str) -> A2APeer`: Look up a registered peer by name.
    - `@property registered_peers(self) -> list[str]`: Return the sorted list of registered peer names.
<!-- END: src/resource/a2a/router.py -->

<!-- BEGIN: src/resource/a2a/tool_adapter.py -->
## src/resource/a2a/tool_adapter.py
*Updated: 2026-10-12 14:29*

**Purpose**: Per-provider adapters that expose A2APeer agents as native tool definitions.
**Functions**:
- `to_anthropic_tools(peers: list[A2APeer]) -> list[dict[str, Any]]`: Convert peers to Anthropic tool definitions.
- `handle_anthropic_tool_call(tool_name: str, tool_input: dict[str, Any], peers: list[A2APeer]) -> str | None`: Execute the peer agent for an Anthropic tool_use block.
- `to_openai_handoffs(peers: list[A2APeer]) -> list[Any]`: Convert peers to openai-agents SDK Agent objects for use as handoffs.
- `to_gemini_tools(peers: list[A2APeer]) -> list[Any]`: Convert peers to Gemini FunctionDeclaration tools.
- `handle_gemini_function_call(function_name: str, function_args: dict[str, Any], peers: list[A2APeer]) -> str | None`: Execute the peer agent for a Gemini function_call part.
- `_safe_tool_name(name: str) -> str`: Normalize a peer name to a valid tool identifier.
<!-- END: src/resource/a2a/tool_adapter.py -->

<!-- BEGIN: src/resource/base_agent.py -->
## src/resource/base_agent.py
*Updated: 2026-10-12 14:29*

**Purpose**: Shared abstract base for all provider-specific agent implementations.
**Classes**:
  - `AbstractBaseAgent(BaseModel, ABC)`: Provider-agnostic contract shared by ClaudeBaseAgent, OpenAIBaseAgent, GeminiBaseAgent.
    - `@abstractmethod run(self, message: str, **kwargs: object) -> str`: Execute the agent synchronously.
    - `async run_async(self, message: str, **kwargs: object) -> str`: Execute the agent asynchronously.
    - `delegate(self, peer_name: str, message: str, trace_id: str | None = None) -> A2AResponse`: Send a sub-task to a named peer agent and return its response.
    - `async delegate_async(self, peer_name: str, message: str, trace_id: str | None = None) -> A2AResponse`: Send a sub-task to a named peer agent asynchronously.
    - `@property peer_names(self) -> list[str]`: Return the sorted list of declared peer names.
    - `has_peers(self) -> bool`: Check whether this agent has any declared peers.
<!-- END: src/resource/base_agent.py -->

<!-- BEGIN: src/resource/base_claude.py -->
## src/resource/base_claude.py
*Updated: 2026-10-12 14:29*

**Purpose**: Anthropic Claude provider base — inherits AbstractBaseAgent.
**Classes**:
  - `ClaudeBaseAgent(AbstractBaseAgent)`: Base for all Claude-backed agents.
    - `_get_vault_context(self) -> str`: Fetch relevant Second Brain notes for this agent's role topic.
    - `_write_to_vault(self, output: str, project_id: str) -> None`: Persist the agent's output as a note in the Second Brain vault.
    - `_build_anthropic_tools(self) -> list[dict[str, Any]]`: Build Anthropic tool definitions for this agent's declared peers.
    - `_handle_tool_use(self, tool_name: str, tool_input: dict[str, Any]) -> str`: Execute the peer agent targeted by an Anthropic tool_use block.
    - `run(self, message: str, project_id: str = 'default', **kwargs: object) -> str`: Execute the agent synchronously via the Anthropic Messages API.
<!-- END: src/resource/base_claude.py -->

<!-- BEGIN: src/resource/base_gemini.py -->
## src/resource/base_gemini.py
*Updated: 2026-10-12 14:30*

**Purpose**: Google Gemini provider base — inherits AbstractBaseAgent.
**Classes**:
  - `GeminiBaseAgent(AbstractBaseAgent)`: Base for all Gemini-backed agents.
    - `model_post_init(self, __context: object) -> None`: Configure the genai client once per instance.
    - `_build_gemini_tools(self) -> list[Any]`: Build Gemini Tool objects for this agent's declared peers.
    - `_handle_function_call(self, function_name: str, function_args: dict[str, Any]) -> str`: Execute the peer agent targeted by a Gemini function_call part.
    - `run(self, message: str, **kwargs: object) -> str`: Execute the agent synchronously via a Gemini GenerativeModel chat.
<!-- END: src/resource/base_gemini.py -->

<!-- BEGIN: src/resource/base_openai.py -->
## src/resource/base_openai.py
*Updated: 2026-10-12 14:30*

**Purpose**: OpenAI provider base — inherits AbstractBaseAgent.
**Classes**:
  - `OpenAIBaseAgent(AbstractBaseAgent)`: Base for all OpenAI-backed agents.
    - `_build_openai_handoffs(self) -> list[Any]`: Build openai-agents handoff objects for this agent's declared peers.
    - `run(self, message: str, **kwargs: object) -> str`: Execute the agent synchronously via the openai-agents Runner.
<!-- END: src/resource/base_openai.py -->
