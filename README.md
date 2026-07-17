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

- `--questions built-in my_questions.json` — merge the built-in bank with your own file(s); see "Adding your own questions" below
- `--category math` — run only one category (fast smoke test)
- `--limit 10` — run only the first N matching questions
- `--concurrency 4` — send requests in parallel (throughput testing; default is sequential for clean latency numbers)
- `--max-tokens`, `--temperature`, `--timeout`, `--retries` — tune request behavior
- `--output ''` — don't save a results file (e.g. for one-off checks)
- `python3 benchmark.py categories` — list category names/counts in the question bank

## Adding your own questions

Keep the built-in bank as-is and add your own questions in a separate
file — this keeps `git pull`-ing tool updates from clobbering your data,
and keeps a duplicate-id check between files instead of a silent merge.

1. Copy `custom_questions.example.json` to e.g. `my_questions.json` and
   edit the entries (same JSON shape as `questions.json`: `id`, `category`,
   `prompt`, `grading`).
2. Run with both files — `built-in` is shorthand for the bundled bank so
   you don't need its absolute path:

   ```bash
   python3 benchmark.py run --questions built-in my_questions.json \
     --base-url ... --api-key ... --model ...
   ```

   Or run only your own set: `--questions my_questions.json`.
3. `benchmark.py categories --questions built-in my_questions.json` lists
   category counts across the merged set so you can sanity-check it loaded.

Loading validates every question up front (unknown grading type, missing
fields, duplicate `id` across files) and fails with a specific error
message rather than crashing mid-run.

### Choosing a grading type

The core design constraint: grading is entirely programmatic (no LLM judge,
no human-in-the-loop), so every question needs an answer a script can check
unambiguously. Work through these in order:

1. **Single correct number** (a count, a port, a year) → `numeric`, with
   optional `tolerance` for approximate answers.
2. **Single correct short string** (a word, a code, a status) → `exact`
   (whole normalized response must match) or `contains` (must appear
   somewhere in the response).
3. **A few acceptable phrasings** ("hola" vs "¡Hola!") → `regex`.
4. **Structured output** (you asked for JSON) → `json`, compared by value.
5. **Longer or open-ended answers that must cover certain facts** — this is
   the common case for your own domain data: RAG-style Q&A over internal
   docs, "summarize this incident," "what's our policy on X" — use
   `keywords` (see below). It's the right tool whenever there's no single
   correct string, but there *is* a checklist of facts a correct answer has
   to hit.
6. **Genuinely open-ended/subjective** (creative writing, "is this a good
   response", tone/style) — out of scope for this tool's deterministic
   grading. Don't try to force it into `keywords`; either reframe the task
   so a correct answer is checkable (ask for a specific fact, a structured
   field, a yes/no judgment against a rubric you define), or accept that
   quality here needs a human or a separate LLM-judge pipeline, which this
   tool intentionally doesn't do (an LLM judging itself, or a possibly-also-
   degraded model, is a shaky source of truth for exactly the kind of
   regression you're trying to catch).

### `keywords` grading (partial credit)

```json
{
  "type": "keywords",
  "required": ["rollback", ["pagerduty", "on-call", "on call"]],
  "optional": ["logs", "runbook"],
  "min_required": 2,
  "case_sensitive": false
}
```

- `required` — terms the answer should cover. Each entry is either a
  string, or a list of alternative phrasings where any one counts (like
  `["pagerduty", "on-call", "on call"]` above — the model doesn't have to
  use your exact wording).
- `optional` — bonus terms that add to the score but aren't required to
  pass.
- `min_required` — how many of the `required` terms must be present to
  count as a pass/fail `correct` (default: all of them). Use this for "must
  mention at least N of these key points."
- `score` = (matched required + matched optional) / (total required +
  total optional) — this is what makes it *partial credit* rather than
  pass/fail: matching 2 of 3 required facts is a 0.67, not a 0.

This changes the run output: `correct`/`accuracy_pct` stays a strict
pass/fail count (did it hit `min_required`?), and a second **Avg score**
figure appears — the mean of the continuous `score` across questions that
have one. Watch both: accuracy can stay flat (still technically "passing")
while avg score quietly drops, which is often the earlier, more sensitive
signal of degradation than a hard pass/fail count — a good thing to trend
in `history` over time.

Responses are cleaned before grading regardless of type:
`<think>...</think>` blocks (from local reasoning models) and wrapping
markdown code fences are stripped first. If you change any question set,
old and new runs get different `questions_hash` values, so stale
comparisons are flagged instead of silently mixing scores from different
question sets.

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
