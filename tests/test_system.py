"""System tests: run benchmark.py as a real subprocess, end to end.

The CLI is invoked exactly as a user would (argparse, subcommands, exit
codes, results files), pointed at a local mock server over loopback. Covers
ping, run (save + summary), category/limit filters, concurrency, the
--fail-under and --fail-if-regression exit codes, and the compare/history/
categories subcommands. Fully offline.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark  # noqa: E402
from mock_server import MockServer, make_answer_responder  # noqa: E402

BENCH = os.path.join(ROOT, "benchmark.py")

SYS_QUESTIONS = {
    "version": 1,
    "questions": [
        {"id": "s1", "category": "sys", "prompt": "Return the number 7.",
         "grading": {"type": "numeric", "answer": 7}},
        {"id": "s2", "category": "sys", "prompt": "Say the phrase hello world.",
         "grading": {"type": "exact", "answer": "hello world"}},
        {"id": "s3", "category": "sys", "prompt": "Respond with JSON status ok.",
         "grading": {"type": "json", "expected": {"status": "ok"}}},
    ],
}

CORRECT_ANSWERS = [
    ("number 7", "7"),
    ("hello world", "hello world"),
    ("JSON status ok", '{"status": "ok"}'),
    ("PONG", "PONG"),  # ping prompt
]


class SystemTestBase(unittest.TestCase):
    def setUp(self):
        self.server = MockServer(
            responder=make_answer_responder(CORRECT_ANSWERS)).start()
        self.addCleanup(self.server.stop)

        self.workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)
        self.qfile = os.path.join(self.workdir, "sys_questions.json")
        with open(self.qfile, "w") as f:
            json.dump(SYS_QUESTIONS, f)
        self.outdir = os.path.join(self.workdir, "results")

    def cli(self, *args):
        return subprocess.run([sys.executable, BENCH, *args],
                              capture_output=True, text=True, timeout=60)

    def run_args(self, *extra):
        return ("run", "--base-url", self.server.base_url, "--api-key", "k",
                "--model", "testmodel", "--questions", self.qfile,
                "--output", self.outdir, "--retries", "0", *extra)

    def saved_files(self):
        return sorted(glob.glob(os.path.join(self.outdir, "*.json")))

    def seed_baseline(self, model, accuracy):
        """Write a prior run result with an old timestamp, matching the current
        questions_hash, so find_previous_run picks it up as the baseline."""
        _, qhash = benchmark.load_questions([self.qfile])
        os.makedirs(self.outdir, exist_ok=True)
        record = {
            "schema_version": benchmark.SCHEMA_VERSION,
            "timestamp": "2020-01-01T00:00:00+00:00",
            "model": model,
            "base_url": self.server.base_url,
            "questions_file": "sys_questions.json",
            "questions_hash": qhash,
            "num_questions": 3,
            "summary": {
                "accuracy_pct": accuracy, "avg_score_pct": None,
                "correct": 3, "total": 3, "errors": 0, "by_category": {},
                "latency_s": {"mean": None, "median": None, "p95": None,
                              "min": None, "max": None},
                "tokens_per_sec": {"mean": None, "median": None, "estimated": False},
            },
            "results": [],
        }
        path = os.path.join(
            self.outdir, "{}_20200101_000000.json".format(benchmark.sanitize_filename(model)))
        with open(path, "w") as f:
            json.dump(record, f)
        return path


class PingTests(SystemTestBase):
    def test_ping_ok(self):
        res = self.cli("ping", "--base-url", self.server.base_url,
                       "--api-key", "k", "--model", "testmodel")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK", res.stdout)

    def test_ping_connection_failure_returns_1(self):
        dead = MockServer().start()
        url = dead.base_url
        dead.stop()
        res = self.cli("ping", "--base-url", url, "--api-key", "k", "--model", "m")
        self.assertEqual(res.returncode, 1)


class RunTests(SystemTestBase):
    def test_full_run_all_correct(self):
        res = self.cli(*self.run_args())
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Accuracy:  100.0%", res.stdout)

    def test_results_file_saved_and_wellformed(self):
        self.cli(*self.run_args())
        files = self.saved_files()
        self.assertEqual(len(files), 1)
        record = json.load(open(files[0]))
        self.assertEqual(record["model"], "testmodel")
        self.assertEqual(record["num_questions"], 3)
        self.assertEqual(record["summary"]["accuracy_pct"], 100.0)
        self.assertEqual(len(record["results"]), 3)

    def test_output_disabled_writes_nothing(self):
        res = self.cli("run", "--base-url", self.server.base_url, "--api-key", "k",
                       "--model", "testmodel", "--questions", self.qfile,
                       "--output", "", "--retries", "0")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse(os.path.isdir(self.outdir))

    def test_category_filter_and_limit(self):
        res = self.cli(*self.run_args("--limit", "1"))
        self.assertEqual(res.returncode, 0, res.stderr)
        record = json.load(open(self.saved_files()[0]))
        self.assertEqual(record["num_questions"], 1)

    def test_concurrency(self):
        res = self.cli(*self.run_args("--concurrency", "3"))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("100.0%", res.stdout)

    def test_bad_questions_file_returns_1(self):
        res = self.cli("run", "--base-url", self.server.base_url, "--api-key", "k",
                       "--model", "m", "--questions", "/no/such/file.json",
                       "--output", "")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error loading questions", res.stderr)


class ExitCodeTests(SystemTestBase):
    def test_fail_under_triggers_exit_2(self):
        # One wrong answer -> 2/3 = 66.7% accuracy, below the 90 floor.
        self.server.set_responder(make_answer_responder(
            [("number 7", "0")] + CORRECT_ANSWERS))
        res = self.cli(*self.run_args("--fail-under", "90"))
        self.assertEqual(res.returncode, 2)
        self.assertIn("FAIL", res.stdout)

    def test_fail_under_passes_when_above_floor(self):
        res = self.cli(*self.run_args("--fail-under", "90"))
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_fail_if_regression_triggers_exit_3(self):
        # Baseline: a prior 100% run for this model + question set.
        self.seed_baseline("testmodel", 100.0)
        # Now run degraded (one wrong answer -> 66.7%), which is a regression.
        self.server.set_responder(make_answer_responder(
            [("number 7", "0")] + CORRECT_ANSWERS))
        res = self.cli(*self.run_args("--fail-if-regression", "0"))
        self.assertEqual(res.returncode, 3)
        self.assertIn("dropped", res.stdout)


class CompareHistoryCategoriesTests(SystemTestBase):
    def test_categories_subcommand(self):
        res = self.cli("categories", "--questions", self.qfile)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("sys", res.stdout)
        self.assertIn("total", res.stdout)

    def test_compare_two_runs(self):
        # Two different model names -> two distinct result files (the filename
        # is <model>_<timestamp>, and same-second runs would otherwise collide).
        a = self.cli(*self.run_args("--model", "model-a"))
        self.assertEqual(a.returncode, 0, a.stderr)
        self.server.set_responder(make_answer_responder(
            [("number 7", "0")] + CORRECT_ANSWERS))
        b = self.cli(*self.run_args("--model", "model-b"))
        self.assertEqual(b.returncode, 0, b.stderr)
        files = self.saved_files()
        self.assertGreaterEqual(len(files), 2)
        res = self.cli("compare", files[0], files[1])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Comparison", res.stdout)

    def test_history_subcommand(self):
        self.cli(*self.run_args())
        res = self.cli("history", "--dir", self.outdir, "--model", "testmodel")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("testmodel", res.stdout)


if __name__ == "__main__":
    unittest.main()
