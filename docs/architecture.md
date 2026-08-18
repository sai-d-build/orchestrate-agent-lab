# Architecture

## Phase 0 architecture

```text
User Input
    ↓
Python application
    ↓
LLM API (future Phase 0 implementation)
    ↓
Structured Output
    ↓
Pydantic Validation
    ↓
Final Response
```

The repository starts intentionally small. We will add retrieval, tools, agents, multimodal processing, guardrails, and evaluation incrementally.

## Architecture principle

Use the simplest architecture that reliably solves the problem. Prefer deterministic code over an LLM when deterministic code is the better tool.
