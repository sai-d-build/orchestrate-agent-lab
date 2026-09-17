# Execution Guide

Step-by-step instructions for running the full inference pipeline on CSV input files.

---

## Prerequisites

### System Requirements
- Python 3.11+
- pip package manager
- Internet access for API calls

### Install Dependencies
```bash
cd /Users/sai-work/Documents/gitCode/new/orchestrate-agent-lab
pip install -r requirements.txt
```

**requirements.txt contents:**
```
pydantic>=2
PyYAML>=6
python-dotenv>=1
requests>=2
scikit-learn
pandas
openai
pytest>=8
```

---

## Environment Configuration

### 1. Copy Environment Template
```bash
cp .env.example .env
```

### 2. Edit `.env` with Your API Keys
```bash
# NVIDIA API key (for Nemotron models via https://integrate.api.nvidia.com/v1)
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter API key (for free model benchmarking)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx

# Model profile to use (see config/models.yaml for available profiles)
# Options: nvidia-ragset-inference, nvidia-ragset-validator, openrouter-free, gemma-4-free, gpt-oss-20b-free, gemini-fast, openai-default, etc.
MODEL_PROFILE=nvidia-ragset-inference

# Direct provider API keys (for later lessons)
GEMINI_API_KEY=your-gemini-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GROQ_API_KEY=your-groq-api-key-here
MISTRAL_API_KEY=your-mistral-api-key-here

# Application
LOG_LEVEL=INFO
```

### 3. Verify Required Files Exist
```bash
# Check train.csv exists at repository root
ls -la train.csv

# Check experiment structure
ls -la experiments/ragset_report_inference_experiment/
```

---

## Pipeline Commands

### Command Overview
```bash
# From repository root:

# 1. Analyze gold reports (one-time, or after data changes)
python -m experiments.ragset_report_inference_experiment.runners.analyze_gold

# 2. Create held-out validation set (required before inference)
python -m experiments.ragset_report_inference_experiment.runners.validate_agent

# 3. Run full inference pipeline
python -m experiments.ragset_report_inference_experiment.runners.run_inference [LIMIT] [--overwrite] [--resume]
```

---

## Step-by-Step Execution

### Step 1: Analyze Gold Reports
```bash
python -m experiments.ragset_report_inference_experiment.runners.analyze_gold
```

**What it does:**
- Loads `train.csv` from repository root
- Splits into gold (all 12 labels populated) and unlabeled sets
- Computes label profiles, evidence statistics, negation/uncertainty patterns
- Outputs analysis to console (used by inference pipeline)

**Expected output:**
```
Gold reports: 58
Unlabeled reports: 1247
Label profiles computed...
```

---

### Step 2: Create Held-Out Validation Set
```bash
python -m experiments.ragset_report_inference_experiment.runners.validate_agent
```

**What it does:**
- Splits gold reports into held-out subset
- Saves to `experiments/ragset_report_inference_experiment/data/validation/heldout.csv`
- Required before running inference (used for held-out evaluation)

**Expected output:**
```
Created held-out set: 10 reports
Saved to: experiments/ragset_report_inference_experiment/data/validation/heldout.csv
```

---

### Step 3: Run Inference Pipeline

#### Fresh Run (Overwrite Existing Results)
```bash
python -m experiments.ragset_report_inference_experiment.runners.run_inference --overwrite
```

#### Resume Previous Run (Skip Processed Reports)
```bash
python -m experiments.ragset_report_inference_experiment.runners.run_inference --resume
```

#### Limit Reports (Testing)
```bash
# Process only 5 reports
python -m experiments.ragset_report_inference_experiment.runners.run_inference 5 --overwrite

# Resume and process up to 10 more
python -m experiments.ragset_report_inference_experiment.runners.run_inference 10 --resume
```

---

## Command-Line Flags Reference

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `limit` | Positional (optional) | Maximum number of reports to process | All unlabeled |
| `--overwrite` | Flag | Delete existing results, start fresh | False |
| `--resume` | Flag | Skip already-processed reports from `predictions.jsonl` | False |

### Flag Combinations

| Command | Behavior |
|---------|----------|
| `run_inference --overwrite` | Fresh run, all reports |
| `run_inference --resume` | Resume, all remaining reports |
| `run_inference 5 --overwrite` | Fresh run, max 5 reports |
| `run_inference 10 --resume` | Resume, max 10 more reports |
| `run_inference` (no flags) | Fresh run, all reports (deletes existing) |

---

## Configuration Options

### Model Selection (via `config/models.yaml`)

**NVIDIA Nemotron (Current Default):**
```yaml
nvidia-ragset-inference:
  provider: nvidia
  model: nvidia/nemotron-3-super-120b-a12b
  api_key_env: NVIDIA_API_KEY
  parameters:
    temperature: 0
    max_output_tokens: 20000
    reasoning: false
```

**OpenRouter Free Models (Alternative):**
```yaml
ragset-inference:
  provider: openrouter
  model: meta-llama/llama-3.1-70b-instruct:free
  api_key_env: OPENROUTER_API_KEY
  parameters:
    temperature: 0
    max_output_tokens: 20000
    reasoning: true
```

**Switch Models:** Edit `config/experiment.yaml`:
```yaml
models:
  inference_profile: nvidia-ragset-inference  # or ragset-inference
  validator_profile: nvidia-ragset-validator  # or ragset-validator
```

### Pipeline Settings (`config/experiment.yaml`)

```yaml
source:
  path: train.csv
  id_column: StudyInstanceUID
  report_column: Report

gold:
  require_all_labels_populated: true

retrieval:
  enabled: true
  method: tfidf
  top_k: 8

models:
  inference_profile: nvidia-ragset-inference
  validator_profile: nvidia-ragset-validator
  temperature: 0

loop:
  max_attempts: 3

validation:
  require_validator_pass: true

runtime:
  output_dir: experiments/ragset_report_inference_experiment
  save_model_trace: true
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `retrieval.top_k` | Number of gold examples to retrieve | 8 |
| `loop.max_attempts` | Max inference/validation retries | 3 |
| `models.temperature` | Sampling temperature | 0 (deterministic) |
| `parameters.max_output_tokens` | Max tokens per response | 20000 |
| `parameters.reasoning` | Enable model reasoning | false (for JSON mode) |

---

## Expected Outputs

### Output Directory Structure
```
experiments/ragset_report_inference_experiment/
├── results/
│   └── inference/
│       ├── predictions.jsonl    # One JSON line per report
│       └── model_trace.jsonl    # One JSON line per model call
├── data/
│   └── validation/
│       └── heldout.csv          # Created by validate_agent.py
```

### `predictions.jsonl` Format
```json
{
  "report_id": "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260",
  "status": "passed",
  "review_reason": null,
  "attempts": [
    {
      "attempt": 1,
      "prediction": {
        "predictions": {
          "ACL": {"value": 0, "evidence": ""},
          "MCL": {"value": 0, "evidence": ""},
          "Medial_Meniscus": {"value": 1, "evidence": "Rotura de menisco interno."},
          "Lateral_Meniscus": {"value": 0, "evidence": ""},
          "Medial_OA": {"value": 1, "evidence": "Artrosis femorotibial medial."},
          "Lateral_OA": {"value": 0, "evidence": ""},
          "PF_OA": {"value": 0, "evidence": ""},
          "Effusion": {"value": 1, "evidence": "Derrame."},
          "Synovitis": {"value": 0, "evidence": ""},
          "Bakers": {"value": 0, "evidence": ""},
          "Contusion": {"value": 0, "evidence": ""},
          "Fracture": {"value": 0, "evidence": ""}
        }
      },
      "validation": {"status": "PASS", "issues": []},
      "elapsed_seconds": 5.78
    }
  ],
  "final_prediction": { ... }
}
```

**Status Values:**
- `passed` — All 12 labels validated by Model 2
- `needs_review` — Failed validation after max attempts

### `model_trace.jsonl` Format
```json
{
  "report_id": "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260",
  "stage": "inference",
  "model": "nvidia/nemotron-3-super-120b-a12b",
  "attempt": 1,
  "prompt": "...",
  "status": "success",
  "latency_seconds": 5.78
}
```

---

## Verification & Testing

### Run Test Suite
```bash
python -m pytest experiments/ragset_report_inference_experiment/tests/ -v
```

**Expected: 24 passed**

### Quick Validation (5 Reports)
```bash
python -m experiments.ragset_report_inference_experiment.runners.run_inference 5 --overwrite
```

**Expected output:**
```
Overwrite mode: starting fresh
No held-out set found. Run validate_agent.py first.
  Completed <report_id> (1 total) - status: passed
  Completed <report_id> (2 total) - status: needs_review
  ...
Inference complete: 5 reports processed
```

### Check Results
```bash
# Count processed reports
wc -l experiments/ragset_report_inference_experiment/results/inference/predictions.jsonl

# View last prediction
tail -1 experiments/ragset_report_inference_experiment/results/inference/predictions.jsonl | python -m json.tool

# Check trace
wc -l experiments/ragset_report_inference_experiment/results/inference/model_trace.jsonl
```

---

## Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `yaml.scanner.ScannerError` | Bad YAML indentation | Fix literal block scalar indentation (2 spaces after `\|`) |
| `ValidationError: evidence None` | Model returns `null` | Schema handles via validator (coerces to `""`) |
| `ValidationError: corrected None` | Validator returns `null` | Schema allows `Literal[0,1] \| None` |
| `503 Service Unavailable` | NVIDIA API overload | Retry; transient issue |
| `KeyError: gold_analysis` | Missing template variable | Ensure `gold_analysis` passed to validator |
| `OPENROUTER_API_KEY required` | Wrong profile | Use `nvidia-ragset-inference` or set `OPENROUTER_API_KEY` |
| `NVIDIA_API_KEY required` | Missing env var | Add to `.env` file |

### Debug Mode
```bash
# Enable verbose logging
LOG_LEVEL=DEBUG python -m experiments.ragset_report_inference_experiment.runners.run_inference 1 --overwrite
```

### Inspect Model Calls
```bash
# View trace for specific report
grep "report_id" experiments/ragset_report_inference_experiment/results/inference/model_trace.jsonl | head -5
```

---

## Performance Expectations

| Metric | Typical Value |
|--------|---------------|
| Latency per report | 5-6 seconds |
| Inference tokens | 2,000-4,000 |
| Validation tokens | 1,500-3,000 |
| Max attempts/report | 3 |
| Rate limit delay | 2 seconds |
| Reports/hour | ~300-400 |

---

## Resume Behavior

The `--resume` flag:
1. Reads existing `predictions.jsonl`
2. Extracts `report_id` from each line
3. Skips reports already in the file
4. Continues from next unprocessed report

**Use case:** Interrupt long runs, handle API outages, incremental processing.

---

## Held-Out Evaluation

Before full inference, the pipeline automatically runs held-out evaluation:
1. Loads `data/validation/heldout.csv`
2. For each held-out report: uses `retrieve_excluding()` (prevents data leakage)
3. Runs full inference/validation loop
4. Compares predictions vs gold labels
5. Prints evaluation metrics to console

**Requires:** `validate_agent.py` run first to create `heldout.csv`