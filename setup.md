---
title: "0 · Setup"
nav_order: 2
---

# Chapter 0 — Setup
{: .no_toc }

Get a working environment in about 10 minutes. Every step says **what to run** and **why**.
{: .fs-5 .fw-300 }

<details open markdown="block">
<summary>Contents</summary>
{: .text-delta }
- TOC
{:toc}
</details>

---

## 0.1 Prerequisites

| Tool | Required? | Why you need it |
|---|---|---|
| **Python ≥ 3.12, < 3.14** | Yes | The course packages target this range. `uv` installs a matching Python for you. |
| **uv** | Strongly recommended | Fast package/project manager. Also needed in **Module 2, Lesson 1** because the MCP server is launched with `uvx`. |
| pip | Alternative | Works, but you manage Python versions yourself. |
| git | Recommended | To clone the course repo. |
| Chrome | Recommended | Best experience for LangSmith Studio and Agent Chat UI. |

{: .tip }
If you install only one thing, install **uv**. It creates the virtual environment, picks the right Python, and runs commands inside it with `uv run …`.

---

## 0.2 Get the code

```bash
# Clone only the latest commit (faster download)
git clone --depth 1 https://github.com/langchain-ai/lca-lc-foundations.git
cd lca-lc-foundations
```

- `--depth 1` → skips the full history; you only need the current files.
- `cd` → every later command assumes you are in the repo root.

---

## 0.3 Add your API keys

```bash
cp example.env .env     # Windows PowerShell: Copy-Item example.env .env
```

Open `.env` and fill it in:

```bash
# ---- Required ----
OPENAI_API_KEY='sk-...'          # the main model (gpt-5-nano) used in almost every lesson
TAVILY_API_KEY='tvly-...'        # web-search tool (Personal Chef, Wedding Planner, ...)

# ---- Optional: only used once in Module 1, Lesson 1 ----
ANTHROPIC_API_KEY='...'          # to compare a Claude model
GOOGLE_API_KEY='...'             # to compare a Gemini model

# ---- Optional: tracing & evaluation ----
LANGSMITH_API_KEY='lsv2_...'
#LANGSMITH_TRACING=true          # un-comment ONLY after you set a valid key
LANGSMITH_PROJECT=lca-lc-foundation
#LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU accounts only
```

**What each key unlocks**

| Key | Service | Cost note |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI models | Needs a paid account; the course mostly uses the inexpensive `gpt-5-nano`. |
| `TAVILY_API_KEY` | Tavily search — returns results formatted for LLMs | Generous free tier. |
| `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | Claude / Gemini | Only to see that `create_agent` works the same with other providers. |
| `LANGSMITH_API_KEY` | LangSmith tracing — see every step the agent took | Free developer tier. |

{: .warning }
> **Never commit `.env`.** It is already in `.gitignore`. If you fork the repo, double-check before pushing.

### How the keys reach your code

Every notebook starts with:

```python
from dotenv import load_dotenv
load_dotenv()          # reads .env and puts each KEY=value into os.environ
```

LangChain integrations (OpenAI, Tavily, LangSmith…) automatically look in `os.environ` for their key, so you never pass keys in code.

{: .note }
`load_dotenv()` does **not** override variables that already exist in your system environment. If you have an old `OPENAI_API_KEY` set globally, it wins. `env_utils.py` warns you about this conflict.

---

## 0.4 Create the virtual environment & install packages

**With uv (recommended)**

```bash
uv sync
```

`uv sync` reads `pyproject.toml` + `uv.lock`, downloads a compatible Python if needed, creates `.venv/`, and installs the exact pinned versions.

**With pip**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 0.5 Verify the setup

```bash
uv run python env_utils.py       # pip users: activate .venv, then  python env_utils.py
```

The script checks:

- ✅ Python location and version (≥ 3.12, < 3.14)
- ✅ The virtual environment is active
- ✅ Required packages and versions are installed
- ✅ Packages live in *this* Python's `site-packages`
- ✅ API keys are configured — and warns if tracing is on without a valid LangSmith key, or if system variables conflict with `.env`

See [Troubleshooting](cheatsheet.html#troubleshooting) if anything is flagged.

---

## 0.6 Run the notebooks

```bash
uv run jupyter lab               # pip: activate .venv, then  jupyter lab
```

Prefer an editor? Open the folder in VS Code / Cursor / Windsurf and select the `.venv` interpreter as the notebook kernel.

---

## 0.7 (Optional) Run LangSmith Studio

Studio is a visual debugger: you see the agent graph, each model call, each tool call, and the state at each step.

```bash
cd notebooks/module-1            # or notebooks/module-3
uv run langgraph dev
```

`langgraph dev` reads the `langgraph.json` file in that folder. It looks roughly like this:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./1.5_personal_chef.py:agent"
  },
  "env": "../../.env"
}
```

| Field | Meaning |
|---|---|
| `graphs` | Name → `file.py:variable`. Studio imports that file and serves the `agent` object. |
| `env` | Where to load API keys from. |
| `dependencies` | Local code to make importable. |

This is why the projects have a **`.py` version** alongside the notebook: Studio needs an importable module, not a notebook.

{: .tip }
Turn on LangSmith tracing now. Seeing the trace of each run is the fastest way to understand what an agent is actually doing.
