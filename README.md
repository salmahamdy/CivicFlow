# ⚙ CivicFlow

> AI-powered business registration assistant with a human-in-the-loop approval step.

CivicFlow automates the tedious parts of business registration — researching regulations, filling forms, validating data — while keeping a **human in control** before anything gets submitted.

---

## How it works

```
User Request → [Orchestrator] → [Researcher] → [Form Filler] → [Validator] → ⏸ Human Approval → [Submit]
```

1. **Orchestrator** — Plans the registration steps using a local LLM (via Ollama)
2. **Researcher** — Looks up relevant regulations and fees
3. **Form Filler** — Drafts the application automatically
4. **Validator** — Checks the form is complete
5. **⏸ Human Approval** — You review the form in the browser before anything is filed
6. **Submission** — Submits and returns a reference ID

---

## Stack

| Layer    | Tech                        |
|----------|-----------------------------|
| Backend  | FastAPI + Uvicorn           |
| AI Agent | LangGraph + LangChain       |
| LLM      | Ollama (phi3.5, local)      |
| Frontend | Vanilla HTML/CSS/JS         |

No databases. No Docker needed. No cloud dependencies.

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally

```bash
# Pull the model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

---

## Setup

```bash
# Clone
git clone https://github.com/yourname/civicflow.git
cd civicflow

# Install dependencies
pip install -r requirements.txt

# Run
cd backend
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Project Structure

```
civicflow/
├── backend/
│   ├── main.py       # FastAPI routes
│   └── agent.py      # LangGraph agent + session helpers
├── frontend/
│   └── index.html    # Single-file UI
├── requirements.txt
└── README.md
```


## License

MIT
