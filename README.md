# 📘 LangChain Foundations — The Master Book

A plain-English, step-by-step study companion for LangChain Academy's **Introduction to LangChain (Python)** course ([course repo](https://github.com/langchain-ai/lca-lc-foundations)).

Every lesson is explained as **The idea → The code → Line-by-line → Gotchas**, so you understand *why* each line exists.

| Chapter | Covers |
|---|---|
| [0 · Setup](docs/setup.md) | Keys, `uv`/pip install, verification, Jupyter, Studio |
| [1 · Create Agent](docs/module-1.md) | Models, prompting, tools, web search, memory, multimodal, **Personal Chef** |
| [2 · Advanced Agent](docs/module-2.md) | MCP, runtime context, state, multi-agent, **Wedding Planner**, RAG/SQL bonus |
| [3 · Production-Ready Agent](docs/module-3.md) | Middleware, summarising/trimming, HITL, dynamic model/prompt/tools, **Email Assistant**, Agent Chat UI |
| [Cheat Sheet](docs/cheatsheet.md) | Imports, one-call summary, troubleshooting, glossary |

## Publish as a GitHub Pages site

1. Create a new GitHub repository and push this folder to it.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** = *Deploy from a branch*, **Branch** = `main`, **Folder** = `/docs`, then **Save**.
4. After a minute your book is live at `https://<your-username>.github.io/<repo-name>/` — with sidebar navigation, search and Mermaid diagrams (theme: [Just the Docs](https://just-the-docs.com)).

Preview locally (optional, needs Ruby):

```bash
cd docs
bundle install
bundle exec jekyll serve      # open http://localhost:4000
```

## Disclaimer

Unofficial study notes. Code samples are explanatory re-writes using LangChain v1 APIs; always compare with the notebooks in the official repo, which are the source of truth. Course content © LangChain, Inc.
