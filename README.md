# Anonymous Submission Artifact

This repository contains the code needed to reproduce the simulation experiments
for a double-blind conference submission. It intentionally excludes generated
figures, generated tables, raw API logs, run artifacts, local caches, and API
keys.

## Contents

- `simulate.py`: main agent-based economic simulation runner.
- `llm_providers.py`: OpenAI, OpenRouter-compatible, and local OpenAI-compatible providers.
- `memory_module.py`: episodic memory, semantic rule memory, and reflection support.
- `market_sentiment.py`: endogenous regime/sentiment dynamics and event controls.
- `run_emnlp_experiments.py`: controlled experiment matrix runner.
- `export_emnlp_outputs.py`: normalizes raw simulation folders into the `runs/` schema.
- `run_gpt52_ls_baseline.sh`: large-scale text-only baseline helper.
- `run_openrouter_e1_lane.sh`: OpenRouter-compatible large-scale helper.
- `ai_economist/`: local simulation environment components.
- `data/profiles.json`: static agent profile fixture.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API keys in your shell. Do not commit them.

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Smoke Test

Run a tiny local-output smoke test before spending substantial API budget:

```bash
python simulate.py \
  --provider openai \
  --model gpt-5.2 \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 13 \
  --tag smoke \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

## Controlled Experiments

Print commands without running:

```bash
python run_emnlp_experiments.py \
  --exps E2,E3,E4,E5 \
  --models gpt52 \
  --seeds 13,21,42
```

Run a focused controlled experiment:

```bash
python run_emnlp_experiments.py --run \
  --exps E2 \
  --models gpt52 \
  --variants default,without-sentiment,without-reflection,without-semantic-memory,without-episodic-memory \
  --seeds 13,21,42 \
  --workers 8
```

## Large-Scale Helpers

Three-seed text-only baseline:

```bash
SEEDS="13 21 42" WORKERS=8 bash run_gpt52_ls_baseline.sh
```

OpenRouter-compatible diagnostic lane:

```bash
bash run_openrouter_e1_lane.sh \
  "Gemini-3-Flash" \
  "google/gemini-3-flash-preview" \
  13 \
  2
```

## Output Policy

Generated artifacts are intentionally ignored by Git:

- `data/*` except `data/profiles.json`
- `runs/`
- `logs/`
- `figs/`
- `*.pkl`, `*.pickle`, `*.zip`, `*.log`
- `.env` and any environment-specific secret files

Use `export_emnlp_outputs.py` to normalize completed runs locally. Do not submit
raw API traces unless the venue explicitly requests them.
