# FineVo Anonymous Artifact

FineVo is an LLM-agent simulation framework for studying economic decision-making dynamics under agent memory, reflection, endogenous market sentiment, and controlled ablation settings. The artifact provides the executable simulation code, provider interfaces, experiment runners, and output-normalization utilities needed to reproduce the main experimental pipeline.

## Repository Contents

Core simulation code:

| File / Directory | Purpose |
| --- | --- |
| `simulate.py` | Main FineVo simulation entry point. Runs one simulation condition with a selected provider, model, seed, setting, and variant. |
| `llm_providers.py` | Provider abstraction for OpenAI, OpenRouter-compatible, and local OpenAI-compatible model endpoints. |
| `memory_module.py` | Episodic memory, semantic rule memory, and reflection support for FineVo agents. |
| `market_sentiment.py` | Endogenous market regime, sentiment dynamics, and event controls. |
| `ai_economist/` | Local economic simulation environment components. |
| `data/profiles.json` | Static anonymized agent profile fixture used by the simulations. |

Experiment orchestration:

| File | Purpose |
| --- | --- |
| `run_emnlp_experiments.py` | Controlled experiment matrix runner for FineVo experiments, models, variants, and seeds. |
| `export_emnlp_outputs.py` | Normalizes completed raw simulation folders into the expected `runs/` schema for downstream analysis. |
| `run_gpt52_ls_baseline.sh` | Large-scale GPT-5.2 text-only baseline helper. |
| `run_openrouter_e1_lane.sh` | OpenRouter-compatible helper for Gemini, Qwen, Llama, and other OpenRouter-routed models. |

Generated artifacts are not tracked by Git. See [Output Policy](#output-policy).

## Requirements

Python 3.11 is recommended.

The artifact can be run in three common modes:

| Mode | Typical Use | GPU Required |
| --- | --- | --- |
| API-only CPU mode | Hosted LLM experiments through OpenAI or OpenRouter | No |
| Apple Silicon MPS mode | Local neural components or local OpenAI-compatible backends on macOS | Optional |
| CUDA mode | Local GPU-backed models or high-throughput local components | Optional, recommended for local models |

For hosted API experiments, the most important requirements are Python dependencies, valid API keys, and provider access. A GPU is not required for API-only FineVo runs.

Install the base environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify that the main scripts are available:

```bash
python simulate.py --help
python run_emnlp_experiments.py --help
python export_emnlp_outputs.py --help
```

## Environment Setup

### 1. API-only CPU Setup

Use this setup when running FineVo with hosted LLM APIs such as OpenAI or OpenRouter.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This mode is sufficient for the standard API-backed FineVo experiments.

Confirm the environment:

```bash
python - <<'PY'
import sys
print("Python:", sys.version)
print("Environment import check passed.")
PY
```

### 2. Apple Silicon MPS Setup

Use this setup on macOS machines with Apple Silicon if local neural components or local model-serving backends use PyTorch/MPS.

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch is not already installed by `requirements.txt`, install the standard macOS wheel:

```bash
pip install torch torchvision torchaudio
```

Check MPS availability:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
```

Recommended MPS fallback setting:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Notes:

- API-only FineVo runs do not require MPS.
- MPS is only relevant when a local component uses PyTorch tensors or a local OpenAI-compatible backend runs on the same machine.
- If a local model/backend exposes its own device flag, configure the backend with `mps`; do not rely only on FineVo command-line flags.

### 3. CUDA Setup

Use this setup on Linux machines with NVIDIA GPUs when running local GPU-backed models, local OpenAI-compatible endpoints, or other CUDA-dependent components.

First check the NVIDIA driver:

```bash
nvidia-smi
```

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install repository dependencies:

```bash
pip install -r requirements.txt
```

If the environment requires an explicit CUDA PyTorch wheel and `requirements.txt` does not already pin the desired CUDA build, install the PyTorch wheel matching the CUDA runtime supported by your machine.

For CUDA 12.4 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.1 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA availability:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.get_device_name(0))
PY
```

Select GPU devices when needed:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Notes:

- CUDA is not required for hosted API-only runs.
- CUDA is useful when a local OpenAI-compatible model server is used behind FineVo’s provider abstraction.
- For multi-GPU local serving, configure device placement in the local serving backend, not only in FineVo.

## API Configuration

FineVo supports multiple LLM families through provider abstractions.

Do not commit API keys, `.env` files, raw traces, provider request/response logs, or machine-specific configuration files.

### OpenAI Models

Use the OpenAI provider for GPT-family models.

```bash
export OPENAI_API_KEY="..."
```

Example OpenAI smoke run:

```bash
python simulate.py \
  --provider openai \
  --model gpt-5.2 \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke_openai \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

### OpenRouter-Compatible Models

Use the OpenRouter-compatible provider for non-OpenAI hosted models used in the FineVo experiments, including Gemini, Qwen, Llama, and other OpenRouter-routed models.

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

In this artifact, Gemini, Qwen, Llama-3.3, Llama-4, and similar non-OpenAI hosted models should be run through the OpenRouter-compatible route unless the user explicitly replaces the provider backend.

Typical model families:

| Model Family | Provider Route | Example Identifier Pattern |
| --- | --- | --- |
| GPT-5.2 | OpenAI | `gpt-5.2` |
| Gemini-3-Flash | OpenRouter-compatible | `google/...` |
| Llama-3.3-70B | OpenRouter-compatible | `meta-llama/...` |
| Llama-4-Maverick | OpenRouter-compatible | `meta-llama/...` |
| Qwen3-235B | OpenRouter-compatible | `qwen/...` |
| Other OpenRouter models | OpenRouter-compatible | Provider-specific OpenRouter model ID |

Example Gemini-style run:

```bash
python simulate.py \
  --provider openrouter \
  --model "google/gemini-3-flash-preview" \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke_gemini \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

Example Qwen-style run:

```bash
python simulate.py \
  --provider openrouter \
  --model "<openrouter-qwen-model-id>" \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke_qwen \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

Example Llama-style run:

```bash
python simulate.py \
  --provider openrouter \
  --model "<openrouter-llama-model-id>" \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke_llama \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

Replace `<openrouter-qwen-model-id>` and `<openrouter-llama-model-id>` with the exact OpenRouter model identifiers used in the experiment matrix or helper scripts.

### Local OpenAI-Compatible Endpoints

FineVo can also be connected to local OpenAI-compatible model servers through the local provider path implemented in `llm_providers.py`.

This mode is useful for locally served models or self-hosted endpoints that expose OpenAI-style chat completion APIs.

General pattern:

```bash
python simulate.py \
  --provider <local-openai-compatible-provider-name> \
  --model <local-model-name> \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke_local \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

Check the available provider names, base-URL options, and model arguments with:

```bash
python simulate.py --help
```

## Model Families Used in FineVo Experiments

The controlled experiments compare multiple LLM families under the same FineVo simulation protocol.

| Model / Baseline | FineVo Condition | Typical Provider Route |
| --- | --- | --- |
| EconAgent baseline | Without FineVo | Local baseline implementation |
| GPT-5.2 | Without FineVo and with FineVo | OpenAI |
| Gemini-3-Flash | Without FineVo and with FineVo | OpenRouter-compatible |
| Llama-3.3-70B | Without FineVo and with FineVo | OpenRouter-compatible |
| Llama-4-Maverick | Without FineVo and with FineVo | OpenRouter-compatible |
| Qwen3-235B | Without FineVo and with FineVo | OpenRouter-compatible |

For reproducibility, always record the exact model identifier, not only the model family name. For OpenRouter-routed models, the exact model ID should be the OpenRouter model string used at runtime.

## Running a Smoke Test

Run a small smoke test before launching controlled or large-scale experiments. This prevents unnecessary API cost from authentication errors, dependency issues, or malformed output paths.

OpenAI smoke test:

```bash
python simulate.py \
  --provider openai \
  --model gpt-5.2 \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

OpenRouter smoke test:

```bash
python simulate.py \
  --provider openrouter \
  --model "<openrouter-model-id>" \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke_openrouter \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0
```

A successful smoke test should complete without provider authentication errors, rate-limit failures, import errors, or missing data-file errors.

Recommended checks after the smoke test:

```bash
find . -maxdepth 3 -type d \( -name "runs" -o -name "logs" -o -name "figs" \)
git status --short --ignored
```

The smoke test may create local run folders or logs depending on the configured output path. These generated files are intentionally ignored by Git.

## Experiment Overview

FineVo experiments can be run at two levels:

1. Single-condition simulation through `simulate.py`.
2. Controlled experiment matrices through `run_emnlp_experiments.py`.

The controlled runner is the preferred interface for reproducing paper-scale experiments because it makes experiment IDs, model aliases, variants, seeds, and worker settings explicit.

Main reproducibility dimensions:

| Dimension | Example Values |
| --- | --- |
| Experiment IDs | `E2`, `E3`, `E4`, `E5` |
| Model families | GPT, Gemini, Qwen, Llama, local OpenAI-compatible models |
| Large-scale seeds | `3`, `21`, `42`, `87`, `2026` |
| Variants | `default`, `without-sentiment`, `without-reflection`, `without-semantic-memory`, `without-episodic-memory` |
| Provider route | `openai`, `openrouter`, or local OpenAI-compatible provider |
| Decoding | `temperature`, `top_p` |
| Worker count | Controlled by `--workers` |

Hosted model APIs may change over time. Runs should record the provider, exact model identifier, seed, decoding parameters, date, worker count, and any provider-side model-version metadata available at runtime.

## Controlled Experiments

### Print Commands Without Running

Use dry-run mode first to inspect the exact commands that will be executed.

GPT-only dry run:

```bash
python run_emnlp_experiments.py \
  --exps E2,E3,E4,E5 \
  --models gpt52 \
  --seeds 3,21,42,87,2026
```

Multi-model dry run:

```bash
python run_emnlp_experiments.py \
  --exps E2,E3,E4,E5 \
  --models gpt52,<gemini_alias>,<qwen_alias>,<llama_alias> \
  --seeds 3,21,42,87,2026
```

Replace `<gemini_alias>`, `<qwen_alias>`, and `<llama_alias>` with the exact aliases defined in `run_emnlp_experiments.py`.

Check available aliases with:

```bash
python run_emnlp_experiments.py --help
```

or inspect the model alias mapping in:

```bash
run_emnlp_experiments.py
```

### Run a Focused Controlled Experiment

The following command runs one controlled FineVo experiment over the default condition and four ablation variants:

```bash
python run_emnlp_experiments.py --run \
  --exps E2 \
  --models gpt52 \
  --variants default,without-sentiment,without-reflection,without-semantic-memory,without-episodic-memory \
  --seeds 3,21,42,87,2026 \
  --workers 8
```

Ablation variants:

| Variant | Description |
| --- | --- |
| `default` | Full FineVo configuration. |
| `without-sentiment` | Removes endogenous market sentiment/regime component. |
| `without-reflection` | Removes agent reflection. |
| `without-semantic-memory` | Removes semantic rule memory. |
| `without-episodic-memory` | Removes episodic memory. |

Worker count controls parallel execution. Increase `--workers` only after confirming API budget, rate limits, and local CPU/GPU capacity.

## Single Simulation Entry Point

For debugging or running a custom condition, call `simulate.py` directly.

Template:

```bash
python simulate.py \
  --provider <provider> \
  --model <model_name> \
  --num_agents <N> \
  --episode_length <T> \
  --gap_fixes True \
  --workers <num_workers> \
  --seed <seed> \
  --tag <run_tag> \
  --exp_id <experiment_id> \
  --setting finevo \
  --variant <variant_name> \
  --temperature <temperature> \
  --top_p <top_p>
```

Example minimal run:

```bash
python simulate.py \
  --provider openai \
  --model gpt-5.2 \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag debug \
  --exp_id debug \
  --setting finevo \
  --variant default \
  --temperature 0.2 \
  --top_p 1.0
```

Recommended debugging strategy:

1. Start with `--num_agents 2` and `--episode_length 1`.
2. Use `--workers 1` until provider authentication and output writing are confirmed.
3. Increase agents, episode length, seeds, variants, and workers only after the small run succeeds.
4. Use explicit `--tag`, `--exp_id`, `--setting`, and `--variant` values so that downstream normalization can group runs correctly.

## Large-Scale Helpers

The large-scale FineVo runs use five seeds:

```text
3, 21, 42, 87, 2026
```

### Five-Seed GPT-5.2 Text-Only Baseline

```bash
SEEDS="3 21 42 87 2026" WORKERS=8 bash run_gpt52_ls_baseline.sh
```

Adjust `WORKERS` according to API rate limits and local resources.

### OpenRouter-Compatible Diagnostic Lane

Use this helper for OpenRouter-routed models, including Gemini, Qwen, Llama-3.3, Llama-4, and other non-OpenAI models.

Gemini example:

```bash
bash run_openrouter_e1_lane.sh \
  "Gemini-3-Flash" \
  "google/gemini-3-flash-preview" \
  3 \
  2
```

Qwen example:

```bash
bash run_openrouter_e1_lane.sh \
  "Qwen3-235B" \
  "<openrouter-qwen3-235b-model-id>" \
  3 \
  2
```

Llama-3.3 example:

```bash
bash run_openrouter_e1_lane.sh \
  "Llama-3.3-70B" \
  "<openrouter-llama-3.3-70b-model-id>" \
  3 \
  2
```

Llama-4 example:

```bash
bash run_openrouter_e1_lane.sh \
  "Llama-4-Maverick" \
  "<openrouter-llama-4-maverick-model-id>" \
  3 \
  2
```

Arguments:

| Position | Example | Meaning |
| --- | --- | --- |
| 1 | `"Qwen3-235B"` | Human-readable run/model label. |
| 2 | `"<openrouter-qwen3-235b-model-id>"` | Exact OpenRouter model identifier. |
| 3 | `3` | Seed. |
| 4 | `2` | Worker count or lane-specific parallelism setting. |

For five-seed OpenRouter runs, repeat the lane over the full seed set.

Gemini:

```bash
for SEED in 3 21 42 87 2026; do
  bash run_openrouter_e1_lane.sh \
    "Gemini-3-Flash" \
    "google/gemini-3-flash-preview" \
    "${SEED}" \
    2
done
```

Qwen:

```bash
for SEED in 3 21 42 87 2026; do
  bash run_openrouter_e1_lane.sh \
    "Qwen3-235B" \
    "<openrouter-qwen3-235b-model-id>" \
    "${SEED}" \
    2
done
```

Llama-3.3:

```bash
for SEED in 3 21 42 87 2026; do
  bash run_openrouter_e1_lane.sh \
    "Llama-3.3-70B" \
    "<openrouter-llama-3.3-70b-model-id>" \
    "${SEED}" \
    2
done
```

Llama-4:

```bash
for SEED in 3 21 42 87 2026; do
  bash run_openrouter_e1_lane.sh \
    "Llama-4-Maverick" \
    "<openrouter-llama-4-maverick-model-id>" \
    "${SEED}" \
    2
done
```

Replace placeholder OpenRouter model IDs with the exact IDs used in the submitted experiment configuration.

## Output Normalization

Raw simulation outputs are generated locally and are not committed to the anonymous repository. After simulations finish, use `export_emnlp_outputs.py` to normalize completed run folders into the expected `runs/` schema.

Check available export options:

```bash
python export_emnlp_outputs.py --help
```

Typical workflow:

```bash
# 1. Run controlled simulations.
python run_emnlp_experiments.py --run \
  --exps E2 \
  --models gpt52 \
  --variants default,without-sentiment,without-reflection,without-semantic-memory,without-episodic-memory \
  --seeds 3,21,42,87,2026 \
  --workers 8

# 2. Normalize completed outputs locally.
python export_emnlp_outputs.py --help

# 3. Inspect normalized run files.
find runs -maxdepth 3 -type f | head
```

The export script should be run on completed local outputs. It is not intended to recover missing or failed API calls.

## Reported Metrics

FineVo simulation outputs are used to compute economic and behavioral metrics such as:

| Metric | Direction Used in Reporting |
| --- | --- |
| Average wealth | Higher is better |
| Gini coefficient | Lower is better |
| Unemployment rate | Lower is better |
| Inflation rate | Reported as an economic stability indicator |

The repository does not include generated result tables or figures. Reproduce them from completed local runs using the normalized `runs/` schema.

## Reproducibility Notes

Use fixed seeds for controlled comparisons. The large-scale FineVo seed set is:

```text
3, 21, 42, 87, 2026
```

For hosted model APIs, exact bitwise reproducibility is not guaranteed across time because providers may update model snapshots, serving infrastructure, routing, moderation behavior, or decoding implementations.

Record the following metadata for every run:

| Item | Recommendation |
| --- | --- |
| Provider | Record `openai`, `openrouter`, or the local endpoint type. |
| Exact model ID | Record the exact model identifier, not only the model family. |
| Model family | Record GPT, Gemini, Qwen, Llama, or local model family. |
| Seeds | Use explicit seeds, preferably `3,21,42,87,2026` for large-scale runs. |
| Decoding | Record `temperature` and `top_p`. |
| Workers | Record worker count. |
| Date | Record run date for API-backed experiments. |
| Variant | Record `default`, `without-sentiment`, `without-reflection`, `without-semantic-memory`, or `without-episodic-memory`. |
| Output tag | Use stable tags to simplify export and grouping. |

For cost control, always run smoke tests before increasing `num_agents`, `episode_length`, number of seeds, number of variants, or worker count.

## Data

The anonymous artifact includes:

```text
data/profiles.json
```

This file is a static anonymized agent-profile fixture used by the simulations.

The repository does not include generated data, raw provider traces, local caches, large run outputs, generated figures, or generated tables.

## Output Policy

Generated artifacts are intentionally ignored by Git.

The following paths or file types should not be committed:

```text
data/* except data/profiles.json
runs/
logs/
figs/
*.pkl
*.pickle
*.zip
*.log
.env
```

Do not commit:

```text
API keys
raw API traces
provider request/response logs
local caches
generated figures
generated tables
temporary run folders
environment-specific secret files
machine-specific paths
non-anonymous metadata
```

Before committing, inspect the working tree:

```bash
git status --short
git status --ignored --short
```

If generated artifacts are required for review, package them separately according to the venue’s artifact policy rather than committing raw traces or provider logs to this repository.

## Troubleshooting

### Missing API Key

If a run fails immediately with an authentication error, confirm the relevant key is set:

```bash
echo "${OPENAI_API_KEY:+OPENAI_API_KEY is set}"
echo "${OPENROUTER_API_KEY:+OPENROUTER_API_KEY is set}"
echo "${OPENROUTER_BASE_URL:+OPENROUTER_BASE_URL is set}"
```

Do not print full keys in shared logs.

### OpenRouter Model Not Found

Confirm that the OpenRouter model identifier is exact. For example, Qwen and Llama model IDs should use the provider-specific OpenRouter format:

```text
qwen/...
meta-llama/...
```

Also confirm the base URL:

```bash
echo "$OPENROUTER_BASE_URL"
```

The expected base URL is:

```text
https://openrouter.ai/api/v1
```

### Rate Limits or Provider Errors

Reduce concurrency first:

```bash
--workers 1
```

Then retry the smoke test before resuming larger runs.

For large-scale experiments, increase concurrency gradually.

### Import Errors

Recreate the virtual environment and reinstall dependencies:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### MPS Is Not Available

Check that the machine is Apple Silicon and that PyTorch was installed correctly:

```bash
python - <<'PY'
import platform
import torch
print(platform.platform())
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
```

If MPS is unavailable, use CPU/API-only mode or a CUDA machine.

### CUDA Is Not Available

Check the NVIDIA driver:

```bash
nvidia-smi
```

Check PyTorch CUDA visibility:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

If `torch.cuda.is_available()` is `False`, reinstall the PyTorch wheel matching the CUDA runtime supported by the machine.

### Unexpected Generated Files in Git

Check ignored files:

```bash
git status --ignored --short
```

Generated outputs should remain untracked. Only source code, configuration required for reproducibility, and the static fixture `data/profiles.json` should be committed.

## Minimal Reproduction Checklist

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

python simulate.py \
  --provider openai \
  --model gpt-5.2 \
  --num_agents 2 \
  --episode_length 1 \
  --gap_fixes True \
  --workers 1 \
  --seed 3 \
  --tag smoke \
  --exp_id smoke \
  --setting finevo \
  --variant smoke \
  --temperature 0.2 \
  --top_p 1.0

python run_emnlp_experiments.py \
  --exps E2,E3,E4,E5 \
  --models gpt52 \
  --seeds 3,21,42,87,2026
```

After the smoke test and dry run succeed, launch the controlled experiment of interest with `--run`.
