# GIM: Evaluating models via tasks that integrate multiple cognitive domains

As LLM benchmarks saturate, the evaluation community has pursued two strategies to increase difficulty: escalating knowledge demands (GPQA, HLE) or removing knowledge entirely in favor of abstract reasoning (ARC-AGI). The first conflates memorization with capability; the second divorces reasoning from the practical contexts in which it matters. We take a different approach. The **Grounded Integration Measure (GIM)** is a benchmark of 820 original problems (615 public, 205 private) where difficulty comes from _integration_; individual problems require coordinating multiple cognitive operations (constraint satisfaction, state tracking, epistemic vigilance, audience calibration) over broadly accessible knowledge, so that reasoning stays grounded in realistic tasks without being gated on specialized expertise. Each problem is an original expert-authored composition, majority with rubric-decomposed scoring (median 6 independently judged criteria). A balanced public--private split provides built-in contamination diagnostic. We calibrate a continuous response 2-parameter logistic (2PL) IRT model over >200k prompt-response pairs across 28 models, producing robust ability estimates that correctly order test-configurations even when raw accuracy is distorted by errors or missing data, addressing a common challenge in benchmark reporting. Using this framework, we present a comprehensive leaderboard spanning 22 models and 47 test-configurations (unique model × thinking-level pairs), and conduct what is to our knowledge the most extensive published study of how test-time compute trades off against model capability on a fixed benchmark: 11 models swept across 35 test-configurations. We observe that within-family configuration choices, such as thinking budget and quantization, matter as much as model selection, and increasing thinking tokens has diminishing marginal returns. We release the evaluation framework, calibrated IRT parameters, and all public problems.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Install with dev and test extras
uv sync --group dev --group test
```

You will also need an API key for the model provider you want to evaluate (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

### Dataset

Download the dataset from HuggingFace into `data/`. This pulls the prompts and all attachments (images, PDFs) needed for evaluation.

```bash
# Install the HuggingFace CLI
# https://huggingface.co/docs/huggingface_hub/installation#install-the-hugging-face-cli
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login

hf download --repo-type dataset facebook/gim --local-dir data
```

## Running Evaluations

GIM is built on [Inspect AI](https://inspect.ai-safety-institute.org.uk/). Run evaluations with `inspect eval`:

Tasks are namespaced as `gim/<variant>`:

| Task | Description |
|---|---|
| `gim/v3` | GIM v3 evaluation (all modalities by default) |

Use the `modality` parameter (`-T modality=<value>`) to filter samples:

| Modality | Samples included |
|---|---|
| `all` (default) | All (text, image, document) |
| `text_only` | Text-only (no attachments) |
| `image` | Image-bearing samples only (no PDFs) |
| `docs` | Document (PDF) bearing samples only |
| `media` | All attachment-bearing samples (images + PDFs) |

```bash
# Text-only (works with all model providers)
uv run inspect eval gim/v3 --model anthropic/claude-sonnet-4-6 -T modality=text_only

# All modalities with a Google model
uv run inspect eval gim/v3 --model google/gemini-3-flash-preview
```

### Task Parameters

| Parameter | Default | Description |
|---|---|---|
| `modality` | `all` | Sample filter: `all`, `text_only`, `image`, `docs`, or `media` |
| `grader_model` | Same as eval model | Model used for LLM-as-judge scoring |
| `dataset_path` | `data` | Path to the HuggingFace dataset directory |
| `media_base` | Dataset directory | Base path/URI for attachments. Use `gs://bucket/prefix` for GCS (Google only) |
| `epochs` | `1` | Number of runs per sample (use 5 for paper results) |

### Thinking / Reasoning Effort

Control thinking level with the standard `--reasoning-effort` flag:

```bash
uv run inspect eval gim/v3 --model google/gemini-3-pro --reasoning-effort high
uv run inspect eval gim/v3 --model anthropic/claude-sonnet-4-6 --reasoning-effort medium
uv run inspect eval gim/v3 --model openai/gpt-5.4-mini --reasoning-effort low
```

## Testing

```bash
# Run all tests
uv run --group test pytest

# Verbose output
uv run --group test pytest -v

# Run a specific test file
uv run --group test pytest tests/test_scorers.py -v
```

### Coverage

<!-- Coverage: 92% — 207 tests passing (last updated: 2026-05-06) -->

```bash
# Run tests with coverage summary
uv run --group test pytest --cov

# Generate an HTML coverage report
uv run --group test pytest --cov --cov-report=html
# then open htmlcov/index.html
```

## Project Structure

```
src/gim/
  task.py              # Inspect AI @task entrypoint
  dataset.py           # HuggingFace dataset → Inspect Sample loader
  scorers.py           # LLM-as-judge scorers (rubric + exact-answer)
  metrics.py           # raw_mean and gim_per_label metrics
  system_registry.py   # Custom model API configs (e.g. thinking levels)
tests/                 # pytest suite
```

## License

This repository (code and data) is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
See [LICENSE](LICENSE) for the full license text.
