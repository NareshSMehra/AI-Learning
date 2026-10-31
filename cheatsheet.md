---
title: "Cheat Sheet & Troubleshooting"
nav_order: 6
---

# Cheat Sheet, Troubleshooting & Glossary
{: .no_toc }

<details open markdown="block">
<summary>Contents</summary>
{: .text-delta }
- TOC
{:toc}
</details>

---

## Imports you'll use all the time

```python
from dotenv import load_dotenv; load_dotenv()

# Models & agents
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState

# Messages
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage

# Tools
from langchain.tools import tool, ToolRuntime

# Memory / control flow
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# Middleware
from langchain.agents.middleware import (
    SummarizationMiddleware, HumanInTheLoopMiddleware,
    before_model, after_model, wrap_model_call, dynamic_prompt,
    ModelRequest, ModelResponse,
)

# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient
```

---

## The whole course in one `create_agent` call

```python
agent = create_agent(
    model="gpt-5-nano",               # 1.1  which LLM (string or model object)
    system_prompt="You are ...",      # 1.1  job description
    tools=[web_search, ...],          # 1.2  what it can do   (2.1: + MCP tools)
    response_format=MyPydanticModel,  # 1.1  structured output (optional)
    checkpointer=InMemorySaver(),     # 1.3  short-term memory (needed for HITL)
    context_schema=MyContext,         # 2.2  read-only per-call data
    state_schema=MyState,             # 2.2  extra remembered fields
    middleware=[...],                 # 3.x  summarise, HITL, dynamic model/prompt/tools
)

agent.invoke(
    {"messages": [HumanMessage("...")]},                 # input state
    {"configurable": {"thread_id": "1"}},                # which conversation
    context=MyContext(...),                              # runtime context
)
```

---

## Calling patterns

| Want | Code |
|---|---|
| Final answer text | `r["messages"][-1].content` |
| Watch every step | `for s in agent.stream(inp, cfg, stream_mode="values"): s["messages"][-1].pretty_print()` |
| Token-by-token | `stream_mode="messages"` |
| Async | `await agent.ainvoke(...)` / `async for ... in agent.astream(...)` |
| Structured result | `r["structured_response"]` |
| Paused for approval? | `"__interrupt__" in r` |
| Resume | `agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)` |
| Inspect saved state | `agent.get_state(cfg).values` |

---

## Message types

| Class | Who produces it | Notable fields |
|---|---|---|
| `SystemMessage` | You (via `system_prompt`) | `content` |
| `HumanMessage` | The user | `content` (str or list of blocks) |
| `AIMessage` | The model | `content`, `tool_calls`, `usage_metadata` |
| `ToolMessage` | A tool's result | `content`, `tool_call_id` (must match the call) |
| `RemoveMessage` | Middleware | `id` (or `REMOVE_ALL_MESSAGES`) |

---

## Troubleshooting

### `ImportError` when running `env_utils.py`
- You're probably running a different Python than the venv. Use `uv run python env_utils.py`, or activate `.venv` first.
- Re-run `uv sync` (or `pip install -r requirements.txt`).

### Wrong Python version
- Required: **≥ 3.12 and < 3.14**. With uv, `uv sync` installs a compatible one into `.venv`. With pip, use pyenv (or similar) to install 3.12/3.13 and recreate the venv.
- In Jupyter/VS Code, make sure the **kernel** points at `.venv`.

### Environment variable conflicts
- A key set in your system/shell overrides `.env`, because `load_dotenv()` doesn't override existing variables.
- Fix: remove/unset the global variable, or call `load_dotenv(override=True)`.

### LangSmith tracing errors (401 / auth warnings)
- `LANGSMITH_TRACING=true` is set but `LANGSMITH_API_KEY` is missing or invalid. Add a valid key or comment tracing out.
- EU account? Un-comment `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.

### MCP / `uvx` problems
- `uvx: command not found` → install uv, then restart the terminal and Jupyter.
- First launch is slow (download). Hanging? Run the same `uvx ...` command in a terminal to see its errors.

### HITL never pauses
- No `checkpointer` → interrupts can't be saved.
- The model answered without calling the tool → make the request explicit or force a tool call.
- The tool name in `interrupt_on` doesn't match the tool's actual `name`.

### "Tool call without matching ToolMessage" errors
- A tool returning `Command` must include a `ToolMessage(..., tool_call_id=runtime.tool_call_id)`.
- Message trimming split an AI tool-call message from its tool results.

### Agent forgets everything
- Missing `checkpointer`, or not passing the **same** `thread_id` config on every call.

---

## Glossary

| Term | Meaning |
|---|---|
| **Agent** | A model running in a loop that can call tools until it can answer. |
| **Tool** | A Python function (name + typed args + docstring) the model can request. |
| **State** | The dict an agent carries through a run; always has `messages`. |
| **Checkpointer** | Saves state after each step so conversations (and pauses) persist. |
| **Thread** | One conversation, identified by `thread_id`. |
| **Runtime context** | Read-only data passed at `invoke` time; hidden from the model. |
| **`ToolRuntime`** | Injected tool parameter giving access to `context`, `state`, `tool_call_id`. |
| **`Command`** | Return value that updates state (or resumes a paused run). |
| **MCP** | Model Context Protocol — a standard for exposing tools to AI apps. |
| **Middleware** | Hooks that run before/after/around model and tool calls. |
| **HITL** | Human-in-the-loop: pause for a person to approve/edit/reject an action. |
| **Interrupt** | The paused point in a run, shown as `__interrupt__`. |
| **LangSmith** | Tracing, debugging and evaluation platform for LLM apps. |
| **Studio** | Visual UI for running/debugging agents served by `langgraph dev`. |

---

## Official references
- Course repo — [langchain-ai/lca-lc-foundations](https://github.com/langchain-ai/lca-lc-foundations)
- LangChain Python docs — [docs.langchain.com](https://docs.langchain.com/oss/python/langchain/overview)
- Chat model providers — [integrations list](https://docs.langchain.com/oss/python/integrations/providers/all_providers)
- LangSmith — [docs](https://docs.langchain.com/langsmith/home)
- uv — [docs.astral.sh/uv](https://docs.astral.sh/uv/)
