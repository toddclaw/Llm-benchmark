# llm-benchmark

A small, self-contained, **offline** benchmark for LLMs served behind an
OpenAI-compatible chat API (Open WebUI, Ollama, LM Studio, vLLM, llama.cpp
server, text-generation-webui, etc.).

It exists for two purposes:

1. **Detect degradation** in a model/deployment you use regularly, by
   re-running the same fixed question set over time and diffing the scores.
2. **Compare models** (a new local/open model vs. what you currently use)
   on the same terms.

Everything runs from the Python standard library — no `pip install`, no
internet access needed at runtime. The only network calls are to the LLM
endpoint you point it at.

## Requirements

- Python 3.8+
- Network access to your LLM's OpenAI-compatible endpoint (can be
  `localhost`, LAN, or fully air-gapped — no internet required)

## How it works

`questions.json` ships 60 questions across 5 categories (12 each):

| Category      | Tests |
|---------------|-------|
| `math`        | Arithmetic and basic word problems |
| `logic`       | Short deductive-reasoning puzzles |
| `code`        | Predicting exact output of small Python snippets |
| `factual`     | Stable, timeless general-knowledge facts |
| `instruction` | Strict format/instruction compliance (exact strings, JSON output) |

Every question has a single programmatically-checkable answer (numeric
match with optional tolerance, exact string match, regex, substring, or
JSON equality) — there's no LLM-as-judge and no subjective scoring, so
results are reproducible and comparable across runs and hardware.

Each run also measures **latency** and **tokens/sec** (from the API's
`usage` field when the backend reports it, otherwise a flagged estimate),
so you can tell whether a regression is in *quality* or *speed* — these
often degrade independently (e.g. quantization changes, a noisy-neighbor
GPU, a swapped backend) and you want to know which one to point at.

The headline number is **accuracy %** (correct / total). Speed is reported
alongside it, not blended in, so it stays a diagnosable signal instead of a
single opaque score.

## Quick start

```bash
# 1. Sanity-check connectivity/auth with a single request
python3 benchmark.py ping \
  --base-url http://your-llm-host:port/v1 \
  --api-key YOUR_API_KEY \
  --model your-model-name

# 2. Run the full benchmark
python3 benchmark.py run \
  --base-url http://your-llm-host:port/v1 \
  --api-key YOUR_API_KEY \
  --model your-model-name
```

`--base-url` is the root of the OpenAI-compatible API — the tool POSTs to
`<base-url>/chat/completions`. For Open WebUI, check **Settings → Account →
API Keys** for a key, and confirm the exact base path for your instance
with the `ping` command first (paths have varied across Open WebUI
versions — some expose `/api/chat/completions` directly, others an
`/v1`-prefixed OpenAI-compatible route). For Ollama it's typically
`http://host:11434/v1`; for llama.cpp's server, LM Studio, and vLLM it's
usually `http://host:port/v1`.

You can also set `LLM_BENCH_BASE_URL`, `LLM_BENCH_API_KEY`, and
`LLM_BENCH_MODEL` as environment variables instead of passing flags every
time (handy since it keeps the key out of shell history).

A results JSON file is saved to `results/` by default, named
`<model>_<timestamp>.json`. If a previous run exists for the same model and
the same question-set version, a comparison against it prints
automatically.

## Tracking degradation over time

Run it the same way on a schedule (cron, or manually) against your work
LLM. Each run is saved, so you build a history:

```bash
python3 benchmark.py history --model your-model-name
```

```
Timestamp                  Model                     Accuracy   Lat(mean)     Tok/s
-------------------------------------------------------------------------------------
2026-06-01 09:00:00        your-model-name              93.3%      1.120s      42.1
2026-06-15 09:00:00        your-model-name              91.7%      1.340s      38.9
2026-07-01 09:00:00        your-model-name              81.7%      2.910s      19.4
2026-07-17 09:00:00        your-model-name              80.0%      3.050s      18.2
```

That kind of trend line is exactly the evidence needed to escalate a
"the LLM feels worse lately" complaint into a data-backed one.

To compare two specific runs directly (e.g. before/after an infra change):

```bash
python3 benchmark.py compare results/model_20260601_090000.json results/model_20260717_090000.json
```

For automation, `run` supports exit codes for scripting/alerting:

```bash
# exit non-zero if accuracy drops below an absolute floor
python3 benchmark.py run ... --fail-under 85

# exit non-zero if accuracy dropped more than N points vs. the previous run
python3 benchmark.py run ... --fail-if-regression 5
```

## Comparing / evaluating new models

Just point `--model` (and `--base-url` if it's a different host) at the
candidate model and run the same command — results are saved separately by
model name, so `history` and `compare` work the same way across different
models as they do across time for one model.

```bash
python3 benchmark.py run --base-url http://localhost:11434/v1 --api-key ollama --model llama3.1:8b
python3 benchmark.py run --base-url http://localhost:11434/v1 --api-key ollama --model qwen2.5:14b
python3 benchmark.py compare results/llama3.1_8b_*.json results/qwen2.5_14b_*.json
```

## Other useful flags

- `--category math` — run only one category (fast smoke test)
- `--limit 10` — run only the first N matching questions
- `--concurrency 4` — send requests in parallel (throughput testing; default is sequential for clean latency numbers)
- `--max-tokens`, `--temperature`, `--timeout`, `--retries` — tune request behavior
- `--output ''` — don't save a results file (e.g. for one-off checks)
- `python3 benchmark.py categories` — list category names/counts in the question bank

## Extending the question bank

`questions.json` is plain JSON; add entries following the existing shape.
Grading types available:

- `numeric` — `{"type": "numeric", "answer": 42, "tolerance": 0}`
- `exact` — `{"type": "exact", "answer": "paris", "case_sensitive": false}`
- `contains` — `{"type": "contains", "answer": "substring"}`
- `regex` — `{"type": "regex", "pattern": "^hola$"}`
- `json` — `{"type": "json", "expected": {"a": 1, "b": 2}}`

Responses are cleaned before grading: `<think>...</think>` blocks (from
local reasoning models) and wrapping markdown code fences are stripped
first. If you change the question set, old and new runs get different
`questions_hash` values, so stale comparisons are flagged instead of
silently mixing scores from different question sets.

## Notes / limitations

- Not a general-purpose eval suite (no MMLU/HumanEval-scale coverage) — it's
  intentionally small so it runs in seconds/minutes and stays fully
  reproducible offline. Feel free to grow the question bank for your needs.
- Non-streaming requests only; token/sec figures come from the API's
  `usage.completion_tokens` when the backend reports it, otherwise a
  rough word-count-based estimate (flagged as `estimated: true` in output).
- `temperature 0` is requested by default for reproducibility, but not all
  backends honor it exactly — expect small run-to-run variance on some
  models/servers.
