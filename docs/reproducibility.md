# Reproducibility Notes

This artifact is code-only. It is designed for double-blind review and should be
mirrored through an anonymization service before sharing with reviewers.

## Seed Sets

Controlled experiments may be run with three or five seeds depending on budget.
The scripts accept comma-separated seed lists for controlled runs and a
space-separated `SEEDS` environment variable for shell helpers.

Recommended compact seed set:

```text
13, 21, 42
```

Extended seed set:

```text
13, 21, 42, 87, 2026
```

## Required Environment Variables

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Output Layout

Normalized outputs use:

```text
runs/<exp_id>/<model>/<setting>/<variant>/seed_<seed>/
```

Each completed run should include:

```text
config.yaml
metrics_summary.csv
trajectory.csv
agent_state.csv
actions.jsonl
memory_retrieval.jsonl
semantic_rules.jsonl
api_errors.jsonl
event_log.csv
```

The `runs/` directory is ignored by Git in this artifact repository.
