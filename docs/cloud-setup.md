# Cloud Setup

The project is designed for browser/cloud development such as GitHub Codespaces or an equivalent environment.

## Python

Use Python 3.11+.

## Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Test

```bash
pytest -q
```

No local PostgreSQL, Redis, Kafka, vector database, GPU, or other infrastructure is required for Phase 0.
