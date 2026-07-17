#!/usr/bin/env python3
"""
llm-benchmark: a small, self-contained, offline LLM benchmark tool.

Talks to any OpenAI-compatible chat-completions endpoint (Open WebUI, Ollama,
LM Studio, vLLM, llama.cpp server, text-generation-webui, ...) using only the
Python standard library -- no pip install required, so it works fully
disconnected from the internet.

Usage:
    python3 benchmark.py ping  --base-url URL --api-key KEY --model NAME
    python3 benchmark.py run   --base-url URL --api-key KEY --model NAME
    python3 benchmark.py compare run_a.json run_b.json
    python3 benchmark.py history --dir results [--model NAME]

Run `python3 benchmark.py <command> --help` for full options.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

SCHEMA_VERSION = 2
DEFAULT_QUESTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?|```$")


# --------------------------------------------------------------------------
# Response cleaning & grading
# --------------------------------------------------------------------------

def clean_response(text):
    """Strip <think>...</think> blocks and surrounding code fences before grading."""
    text = THINK_BLOCK_RE.sub("", text or "")
    text = text.strip()
    stripped = FENCE_RE.sub("", text).strip()
    if stripped:
        text = stripped
    return text.strip()


def normalize_text(s, case_sensitive=False):
    s = s.strip()
    s = s.strip("\"'` \t\n.")
    s = re.sub(r"\s+", " ", s)
    if not case_sensitive:
        s = s.lower()
    return s


def extract_last_number(text):
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_json(text):
    """Find the first balanced {...} or [...] block in text and parse it."""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def _keyword_hit(item, hay, case_sensitive):
    """item is a keyword string, or a list of alternative phrasings (any one counts)."""
    alts = item if isinstance(item, list) else [item]
    for alt in alts:
        needle = alt if case_sensitive else alt.lower()
        if needle in hay:
            return True
    return False


def grade(question, raw_response):
    """Returns (correct: bool, note: str|None, score: float in [0, 1])."""
    g = question["grading"]
    gtype = g["type"]
    response = clean_response(raw_response)

    if gtype == "numeric":
        val = extract_last_number(response)
        if val is None:
            return False, "no number found in response", 0.0
        tolerance = g.get("tolerance", 0)
        ok = abs(val - float(g["answer"])) <= tolerance
        return ok, None, 1.0 if ok else 0.0

    if gtype == "exact":
        case_sensitive = g.get("case_sensitive", False)
        got = normalize_text(response, case_sensitive)
        want = normalize_text(str(g["answer"]), case_sensitive)
        ok = got == want
        return ok, None, 1.0 if ok else 0.0

    if gtype == "contains":
        case_sensitive = g.get("case_sensitive", False)
        hay = response if case_sensitive else response.lower()
        needle = str(g["answer"]) if case_sensitive else str(g["answer"]).lower()
        ok = needle in hay
        return ok, None, 1.0 if ok else 0.0

    if gtype == "regex":
        flags = re.IGNORECASE if not g.get("case_sensitive", False) else 0
        ok = re.search(g["pattern"], response.strip(), flags) is not None
        return ok, None, 1.0 if ok else 0.0

    if gtype == "json":
        parsed = extract_json(response)
        if parsed is None:
            return False, "no valid JSON found in response", 0.0
        ok = parsed == g["expected"]
        return ok, None, 1.0 if ok else 0.0

    if gtype == "keywords":
        case_sensitive = g.get("case_sensitive", False)
        hay = response if case_sensitive else response.lower()
        required = g.get("required", [])
        optional = g.get("optional", [])
        req_hits = sum(1 for item in required if _keyword_hit(item, hay, case_sensitive))
        opt_hits = sum(1 for item in optional if _keyword_hit(item, hay, case_sensitive))
        total = len(required) + len(optional)
        score = (req_hits + opt_hits) / total if total else 0.0
        min_required = g.get("min_required", len(required))
        ok = req_hits >= min_required
        note = None if ok else f"matched {req_hits}/{len(required)} required keywords (need {min_required})"
        return ok, note, score

    raise ValueError(f"unknown grading type: {gtype}")


def validate_question(q, source):
    for field in ("id", "category", "prompt", "grading"):
        if field not in q:
            raise ValueError(f"question in {source} is missing required field '{field}': {q}")
    g = q["grading"]
    gtype = g.get("type")
    known = {"numeric", "exact", "contains", "regex", "json", "keywords"}
    if gtype not in known:
        raise ValueError(f"question '{q['id']}' in {source} has unknown grading type '{gtype}' "
                          f"(expected one of {sorted(known)})")
    if gtype in ("numeric",) and "answer" not in g:
        raise ValueError(f"question '{q['id']}' in {source}: numeric grading needs 'answer'")
    if gtype in ("exact", "contains") and "answer" not in g:
        raise ValueError(f"question '{q['id']}' in {source}: {gtype} grading needs 'answer'")
    if gtype == "regex" and "pattern" not in g:
        raise ValueError(f"question '{q['id']}' in {source}: regex grading needs 'pattern'")
    if gtype == "json" and "expected" not in g:
        raise ValueError(f"question '{q['id']}' in {source}: json grading needs 'expected'")
    if gtype == "keywords":
        if not g.get("required"):
            raise ValueError(f"question '{q['id']}' in {source}: keywords grading needs a non-empty "
                              f"'required' list")


# --------------------------------------------------------------------------
# API client (stdlib only)
# --------------------------------------------------------------------------

class ApiError(Exception):
    pass


def call_chat_api(base_url, api_key, model, prompt, max_tokens, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise ApiError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ApiError(f"connection error: {e.reason}") from e
    except TimeoutError:
        raise ApiError(f"timed out after {timeout}s") from None
    except json.JSONDecodeError as e:
        raise ApiError(f"invalid JSON response: {e}") from e
    elapsed = time.monotonic() - start

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ApiError(f"unexpected response shape: {json.dumps(body)[:500]}") from e

    usage = body.get("usage") or {}
    return content, elapsed, usage


def call_with_retries(base_url, api_key, model, prompt, max_tokens, temperature, timeout, retries):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return call_chat_api(base_url, api_key, model, prompt, max_tokens, temperature, timeout)
        except ApiError as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise last_err


# --------------------------------------------------------------------------
# Run command
# --------------------------------------------------------------------------

def resolve_questions_path(path):
    if path.lower() in ("built-in", "builtin", "default"):
        return DEFAULT_QUESTIONS_FILE
    return path


def load_questions(paths, category_filter=None, limit=None):
    """Load and merge one or more question-bank JSON files.

    Raises ValueError with a descriptive message on missing fields, unknown
    grading types, or duplicate question ids across files.
    """
    all_questions = []
    seen_ids = {}
    for raw_path in paths:
        path = resolve_questions_path(raw_path)
        with open(path) as f:
            data = json.load(f)
        for q in data["questions"]:
            validate_question(q, path)
            if q["id"] in seen_ids:
                raise ValueError(f"duplicate question id '{q['id']}' in {path} "
                                  f"(already defined in {seen_ids[q['id']]})")
            seen_ids[q["id"]] = path
            all_questions.append(q)

    canonical = json.dumps(sorted(all_questions, key=lambda q: q["id"]), sort_keys=True).encode("utf-8")
    questions_hash = hashlib.sha256(canonical).hexdigest()[:16]

    questions = all_questions
    if category_filter:
        questions = [q for q in questions if q["category"] == category_filter]
    if limit:
        questions = questions[:limit]
    return questions, questions_hash


def estimate_tokens(text):
    words = len(text.split())
    return max(1, round(words / 0.75))


def run_one(question, base_url, api_key, model, max_tokens, temperature, timeout, retries):
    entry = {"id": question["id"], "category": question["category"]}
    try:
        content, elapsed, usage = call_with_retries(
            base_url, api_key, model, question["prompt"], max_tokens, temperature, timeout, retries
        )
    except ApiError as e:
        entry.update({"correct": False, "score": 0.0, "error": str(e), "latency_s": None,
                       "tokens_per_sec": None, "response": None})
        return entry

    correct, grade_note, score = grade(question, content)
    completion_tokens = usage.get("completion_tokens")
    tokens_estimated = completion_tokens is None
    if completion_tokens is None:
        completion_tokens = estimate_tokens(content)
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else None

    entry.update({
        "correct": correct,
        "score": round(score, 4),
        "error": grade_note if not correct else None,
        "latency_s": round(elapsed, 4),
        "completion_tokens": completion_tokens,
        "tokens_estimated": tokens_estimated,
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
        "response": content,
    })
    return entry


def summarize(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r.get("error") and r.get("latency_s") is None)
    scores = [r["score"] for r in results if r.get("score") is not None]

    by_category = {}
    for r in results:
        c = by_category.setdefault(r["category"], {"correct": 0, "total": 0, "scores": []})
        c["total"] += 1
        if r["correct"]:
            c["correct"] += 1
        if r.get("score") is not None:
            c["scores"].append(r["score"])
    for c in by_category.values():
        c["pct"] = round(100 * c["correct"] / c["total"], 1) if c["total"] else 0.0
        c["avg_score_pct"] = round(100 * statistics.mean(c["scores"]), 1) if c["scores"] else None
        del c["scores"]

    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    tps_values = [r["tokens_per_sec"] for r in results if r.get("tokens_per_sec")]
    any_estimated = any(r.get("tokens_estimated") for r in results if r.get("latency_s") is not None)

    def pct95(vals):
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, int(round(0.95 * (len(s) - 1)))))
        return s[idx]

    summary = {
        "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
        "avg_score_pct": round(100 * statistics.mean(scores), 1) if scores else None,
        "correct": correct,
        "total": total,
        "errors": errors,
        "by_category": by_category,
        "latency_s": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "p95": round(pct95(latencies), 3) if latencies else None,
            "min": round(min(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "tokens_per_sec": {
            "mean": round(statistics.mean(tps_values), 2) if tps_values else None,
            "median": round(statistics.median(tps_values), 2) if tps_values else None,
            "estimated": any_estimated,
        },
    }
    return summary


def uses_partial_credit(summary):
    """True if avg_score_pct differs meaningfully from accuracy_pct, i.e. at least one
    partial-credit ('keywords') question was graded, so it's worth a separate line."""
    if summary.get("avg_score_pct") is None:
        return False
    return abs(summary["avg_score_pct"] - summary["accuracy_pct"]) >= 0.05


def sanitize_filename(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def cmd_run(args):
    try:
        questions, qhash = load_questions(args.questions, args.category, args.limit)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading questions: {e}", file=sys.stderr)
        return 1
    if not questions:
        print("No questions matched the given filters.", file=sys.stderr)
        return 1

    print(f"Running {len(questions)} questions against model '{args.model}' at {args.base_url}")
    print(f"(concurrency={args.concurrency}, timeout={args.timeout}s, retries={args.retries})\n")

    results = [None] * len(questions)
    t0 = time.monotonic()

    def work(i, q):
        return i, run_one(q, args.base_url, args.api_key, args.model,
                           args.max_tokens, args.temperature, args.timeout, args.retries)

    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(work, i, q) for i, q in enumerate(questions)]
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                i, entry = fut.result()
                results[i] = entry
                done += 1
                status = "OK " if entry["correct"] else ("ERR" if entry.get("latency_s") is None else "X  ")
                print(f"[{done}/{len(questions)}] {status} {entry['id']}")
    else:
        for i, q in enumerate(questions):
            _, entry = work(i, q)
            results[i] = entry
            status = "OK " if entry["correct"] else ("ERR" if entry.get("latency_s") is None else "X  ")
            print(f"[{i + 1}/{len(questions)}] {status} {entry['id']}")

    wall_s = time.monotonic() - t0
    summary = summarize(results)

    run_record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "questions_file": ",".join(os.path.basename(resolve_questions_path(p)) for p in args.questions),
        "questions_hash": qhash,
        "num_questions": len(questions),
        "concurrency": args.concurrency,
        "wall_time_s": round(wall_s, 2),
        "results": results,
        "summary": summary,
    }

    print_summary(run_record)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"{sanitize_filename(args.model)}_{ts}.json"
        outpath = os.path.join(args.output, fname)
        with open(outpath, "w") as f:
            json.dump(run_record, f, indent=2)
        print(f"\nSaved: {outpath}")

        prev = find_previous_run(args.output, args.model, qhash, exclude=outpath)
        if prev:
            print()
            print_comparison(prev, run_record)

    if args.fail_under is not None and summary["accuracy_pct"] < args.fail_under:
        print(f"\nFAIL: accuracy {summary['accuracy_pct']}% is below --fail-under {args.fail_under}%")
        return 2

    if args.fail_if_regression is not None and args.output:
        prev = find_previous_run(args.output, args.model, qhash, exclude=outpath if args.output else None)
        if prev:
            drop = prev["summary"]["accuracy_pct"] - summary["accuracy_pct"]
            if drop > args.fail_if_regression:
                print(f"\nFAIL: accuracy dropped {drop:.1f} points vs previous run "
                      f"(threshold {args.fail_if_regression})")
                return 3

    return 0


def print_summary(run_record):
    s = run_record["summary"]
    show_score = uses_partial_credit(s)
    print("\n" + "=" * 60)
    print(f"Model:     {run_record['model']}")
    print(f"Accuracy:  {s['accuracy_pct']}%  ({s['correct']}/{s['total']}, {s['errors']} errors)")
    if show_score:
        print(f"Avg score: {s['avg_score_pct']}%  (partial credit, from 'keywords'-graded questions)")
    print("-" * 60)
    if show_score:
        print(f"{'Category':<14}{'Correct':>10}{'Total':>8}{'Pct':>8}{'AvgScore':>10}")
        for cat, c in sorted(s["by_category"].items()):
            avg = f"{c['avg_score_pct']}%" if c["avg_score_pct"] is not None else "n/a"
            print(f"{cat:<14}{c['correct']:>10}{c['total']:>8}{c['pct']:>7}%{avg:>10}")
    else:
        print(f"{'Category':<14}{'Correct':>10}{'Total':>8}{'Pct':>8}")
        for cat, c in sorted(s["by_category"].items()):
            print(f"{cat:<14}{c['correct']:>10}{c['total']:>8}{c['pct']:>7}%")
    print("-" * 60)
    lat = s["latency_s"]
    tps = s["tokens_per_sec"]
    if lat["mean"] is not None:
        print(f"Latency (s): mean={lat['mean']} median={lat['median']} p95={lat['p95']} "
              f"min={lat['min']} max={lat['max']}")
    if tps["mean"] is not None:
        est = " (estimated)" if tps["estimated"] else ""
        print(f"Tokens/sec:  mean={tps['mean']} median={tps['median']}{est}")
    print(f"Wall time:   {run_record['wall_time_s']}s")
    print("=" * 60)


# --------------------------------------------------------------------------
# Compare / history
# --------------------------------------------------------------------------

def load_run(path):
    with open(path) as f:
        return json.load(f)


def find_previous_run(directory, model, qhash, exclude=None):
    candidates = []
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        if exclude and os.path.abspath(fpath) == os.path.abspath(exclude):
            continue
        try:
            record = load_run(fpath)
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("model") == model and record.get("questions_hash") == qhash:
            candidates.append(record)
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["timestamp"])
    return candidates[-1]


def fmt_delta(delta, unit="", higher_is_better=True):
    if delta is None:
        return "n/a"
    arrow = "-"
    if abs(delta) >= 0.05:
        if (delta > 0) == higher_is_better:
            arrow = "UP"
        else:
            arrow = "DOWN"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}{unit} [{arrow}]"


def print_comparison(old, new):
    print("Comparison vs previous run:")
    print(f"  previous: {old['timestamp']}  ({old['model']})")
    print(f"  current:  {new['timestamp']}  ({new['model']})")
    print("-" * 60)
    old_s, new_s = old["summary"], new["summary"]
    acc_delta = new_s["accuracy_pct"] - old_s["accuracy_pct"]
    print(f"Accuracy:   {old_s['accuracy_pct']}% -> {new_s['accuracy_pct']}%  "
          f"({fmt_delta(acc_delta, '%')})")

    old_avg, new_avg = old_s.get("avg_score_pct"), new_s.get("avg_score_pct")
    if old_avg is not None and new_avg is not None:
        print(f"Avg score:  {old_avg}% -> {new_avg}%  ({fmt_delta(new_avg - old_avg, '%')})")

    cats = sorted(set(old_s["by_category"]) | set(new_s["by_category"]))
    for cat in cats:
        o = old_s["by_category"].get(cat, {}).get("pct")
        n = new_s["by_category"].get(cat, {}).get("pct")
        if o is None or n is None:
            continue
        d = n - o
        print(f"  {cat:<12} {o}% -> {n}%  ({fmt_delta(d, '%')})")

    if old_s["latency_s"]["mean"] is not None and new_s["latency_s"]["mean"] is not None:
        d = new_s["latency_s"]["mean"] - old_s["latency_s"]["mean"]
        print(f"Latency:    {old_s['latency_s']['mean']}s -> {new_s['latency_s']['mean']}s  "
              f"({fmt_delta(d, 's', higher_is_better=False)})")
    if old_s["tokens_per_sec"]["mean"] is not None and new_s["tokens_per_sec"]["mean"] is not None:
        d = new_s["tokens_per_sec"]["mean"] - old_s["tokens_per_sec"]["mean"]
        print(f"Tokens/sec: {old_s['tokens_per_sec']['mean']} -> {new_s['tokens_per_sec']['mean']}  "
              f"({fmt_delta(d)})")
    print("-" * 60)


def cmd_compare(args):
    old = load_run(args.old)
    new = load_run(args.new)
    if old.get("questions_hash") != new.get("questions_hash"):
        print("WARNING: these runs used different question sets/versions; "
              "comparison may not be meaningful.", file=sys.stderr)
    print_comparison(old, new)
    return 0


def cmd_history(args):
    if not os.path.isdir(args.dir):
        print(f"No results directory at {args.dir}", file=sys.stderr)
        return 1
    records = []
    for fname in sorted(os.listdir(args.dir)):
        if not fname.endswith(".json"):
            continue
        try:
            r = load_run(os.path.join(args.dir, fname))
        except (json.JSONDecodeError, OSError):
            continue
        if args.model and r.get("model") != args.model:
            continue
        records.append(r)
    if not records:
        print("No matching runs found.")
        return 0
    records.sort(key=lambda r: r["timestamp"])

    print(f"{'Timestamp':<27}{'Model':<24}{'Accuracy':>10}{'Lat(mean)':>12}{'Tok/s':>10}")
    print("-" * 83)
    for r in records:
        s = r["summary"]
        lat = s["latency_s"]["mean"]
        tps = s["tokens_per_sec"]["mean"]
        ts = r["timestamp"].split(".")[0].replace("T", " ")
        print(f"{ts:<27}{r['model'][:23]:<24}{s['accuracy_pct']:>9}%"
              f"{(str(lat) + 's') if lat is not None else 'n/a':>12}{tps if tps is not None else 'n/a':>10}")
    return 0


# --------------------------------------------------------------------------
# Ping / categories
# --------------------------------------------------------------------------

def cmd_ping(args):
    print(f"Pinging {args.base_url} with model '{args.model}'...")
    try:
        content, elapsed, usage = call_chat_api(
            args.base_url, args.api_key, args.model,
            "Respond with only the word PONG.", max_tokens=16, temperature=0, timeout=args.timeout,
        )
    except ApiError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(f"OK ({elapsed:.2f}s): {content!r}")
    if usage:
        print(f"usage: {usage}")
    return 0


def cmd_categories(args):
    try:
        questions, _ = load_questions(args.questions)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading questions: {e}", file=sys.stderr)
        return 1
    from collections import Counter
    counts = Counter(q["category"] for q in questions)
    for cat, n in sorted(counts.items()):
        print(f"{cat:<14}{n}")
    print(f"{'total':<14}{len(questions)}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_connection_args(p):
    p.add_argument("--base-url", default=os.environ.get("LLM_BENCH_BASE_URL"),
                    help="OpenAI-compatible API base URL, e.g. http://localhost:11434/v1 "
                         "(env: LLM_BENCH_BASE_URL)")
    p.add_argument("--api-key", default=os.environ.get("LLM_BENCH_API_KEY", ""),
                    help="API key sent as Authorization: Bearer <key> (env: LLM_BENCH_API_KEY)")
    p.add_argument("--model", default=os.environ.get("LLM_BENCH_MODEL"),
                    help="model name as expected by the endpoint (env: LLM_BENCH_MODEL)")
    p.add_argument("--timeout", type=float, default=120, help="per-request timeout in seconds")


def build_parser():
    parser = argparse.ArgumentParser(description="Small self-contained offline LLM benchmark tool.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the benchmark against a model")
    add_connection_args(p_run)
    p_run.add_argument("--questions", nargs="+", default=[DEFAULT_QUESTIONS_FILE],
                        help="one or more question JSON files to merge and run (use 'built-in' as a "
                             "shorthand for the bundled questions.json, e.g. "
                             "--questions built-in my_questions.json)")
    p_run.add_argument("--category", default=None, help="only run questions in this category")
    p_run.add_argument("--limit", type=int, default=None, help="only run the first N matching questions")
    p_run.add_argument("--output", default=DEFAULT_RESULTS_DIR,
                        help="directory to save run results JSON (set to '' to disable saving)")
    p_run.add_argument("--concurrency", type=int, default=1, help="number of parallel requests")
    p_run.add_argument("--retries", type=int, default=2, help="retries per question on transport errors")
    p_run.add_argument("--max-tokens", type=int, default=512, help="max_tokens sent to the API")
    p_run.add_argument("--temperature", type=float, default=0, help="sampling temperature")
    p_run.add_argument("--fail-under", type=float, default=None,
                        help="exit non-zero if accuracy_pct is below this value")
    p_run.add_argument("--fail-if-regression", type=float, default=None,
                        help="exit non-zero if accuracy dropped more than this many points vs previous run")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="compare two saved run result files")
    p_cmp.add_argument("old", help="path to older run JSON")
    p_cmp.add_argument("new", help="path to newer run JSON")
    p_cmp.set_defaults(func=cmd_compare)

    p_hist = sub.add_parser("history", help="show accuracy/speed trend across saved runs")
    p_hist.add_argument("--dir", default=DEFAULT_RESULTS_DIR, help="results directory to scan")
    p_hist.add_argument("--model", default=None, help="filter to a specific model name")
    p_hist.set_defaults(func=cmd_history)

    p_ping = sub.add_parser("ping", help="send one test request to verify connectivity/auth")
    add_connection_args(p_ping)
    p_ping.set_defaults(func=cmd_ping)

    p_cat = sub.add_parser("categories", help="list question categories and counts")
    p_cat.add_argument("--questions", nargs="+", default=[DEFAULT_QUESTIONS_FILE],
                        help="one or more question JSON files to merge (use 'built-in' as a shorthand "
                             "for the bundled questions.json)")
    p_cat.set_defaults(func=cmd_categories)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("run", "ping"):
        if not args.base_url:
            parser.error("--base-url is required (or set LLM_BENCH_BASE_URL)")
        if not args.model:
            parser.error("--model is required (or set LLM_BENCH_MODEL)")

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
