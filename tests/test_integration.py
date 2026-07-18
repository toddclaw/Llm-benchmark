"""Integration tests: benchmark.py's API layer against a live local mock server.

Exercises the real HTTP round-trip (urllib -> stdlib http.server on loopback)
including auth headers, error handling, retry/backoff, and the run_one wiring
that turns an API response into a graded result entry. The retry-backoff
sleeps are patched out so the suite stays fast.
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark  # noqa: E402
from mock_server import MockServer, chat_body, make_flaky_responder  # noqa: E402


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        self.server = MockServer(default_content="PONG", default_tokens=7).start()
        self.addCleanup(self.server.stop)

    def test_success_returns_content_and_usage(self):
        content, elapsed, usage = benchmark.call_chat_api(
            self.server.base_url, "", "m", "hi", 16, 0, 5)
        self.assertEqual(content, "PONG")
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(usage.get("completion_tokens"), 7)

    def test_posts_to_chat_completions_path(self):
        benchmark.call_chat_api(self.server.base_url, "", "m", "hi", 16, 0, 5)
        self.assertTrue(self.server.requests[-1]["path"].endswith("/chat/completions"))

    def test_auth_header_sent_when_key_present(self):
        benchmark.call_chat_api(self.server.base_url, "secret-key", "m", "hi", 16, 0, 5)
        self.assertEqual(self.server.requests[-1]["headers"].get("Authorization"),
                         "Bearer secret-key")

    def test_no_auth_header_when_key_blank(self):
        benchmark.call_chat_api(self.server.base_url, "", "m", "hi", 16, 0, 5)
        self.assertIsNone(self.server.requests[-1]["headers"].get("Authorization"))

    def test_payload_shape(self):
        benchmark.call_chat_api(self.server.base_url, "", "mymodel", "hello", 32, 0.5, 5)
        body = self.server.requests[-1]["json"]
        self.assertEqual(body["model"], "mymodel")
        self.assertEqual(body["messages"][0]["content"], "hello")
        self.assertFalse(body["stream"])

    def test_http_error_becomes_apierror(self):
        self.server.set_responder(lambda req, srv: (500, {"error": "boom"}, 0))
        with self.assertRaises(benchmark.ApiError) as ctx:
            benchmark.call_chat_api(self.server.base_url, "", "m", "hi", 16, 0, 5)
        self.assertIn("HTTP 500", str(ctx.exception))

    def test_unexpected_shape_becomes_apierror(self):
        self.server.set_responder(lambda req, srv: (200, {"no_choices": True}, 0))
        with self.assertRaises(benchmark.ApiError):
            benchmark.call_chat_api(self.server.base_url, "", "m", "hi", 16, 0, 5)

    def test_connection_error(self):
        # Grab a port, then close the server so the connection is refused.
        dead = MockServer().start()
        url = dead.base_url
        dead.stop()
        with self.assertRaises(benchmark.ApiError) as ctx:
            benchmark.call_chat_api(url, "", "m", "hi", 16, 0, 3)
        self.assertIn("connection error", str(ctx.exception))


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.server = MockServer().start()
        self.addCleanup(self.server.stop)

    def test_retries_then_succeeds(self):
        self.server.set_responder(make_flaky_responder(2, content="OK"))
        with mock.patch("benchmark.time.sleep"):  # skip backoff delays
            content, _, _ = benchmark.call_with_retries(
                self.server.base_url, "", "m", "hi", 16, 0, 5, retries=2)
        self.assertEqual(content, "OK")
        self.assertEqual(len(self.server.requests), 3)  # 2 failures + 1 success

    def test_exhausts_retries_and_raises(self):
        self.server.set_responder(lambda req, srv: (503, {"error": "down"}, 0))
        with mock.patch("benchmark.time.sleep"):
            with self.assertRaises(benchmark.ApiError):
                benchmark.call_with_retries(
                    self.server.base_url, "", "m", "hi", 16, 0, 5, retries=1)
        self.assertEqual(len(self.server.requests), 2)  # retries + 1 attempt


class RunOneTests(unittest.TestCase):
    def setUp(self):
        self.server = MockServer().start()
        self.addCleanup(self.server.stop)
        self.question = {"id": "t1", "category": "sys", "prompt": "Give 42",
                         "grading": {"type": "numeric", "answer": 42}}

    def _run(self):
        return benchmark.run_one(self.question, self.server.base_url, "", "m",
                                 16, 0, 5, retries=0)

    def test_correct_answer(self):
        self.server.set_responder(lambda req, srv: (200, chat_body("42", 7), 0))
        entry = self._run()
        self.assertTrue(entry["correct"])
        self.assertEqual(entry["score"], 1.0)
        self.assertIsNotNone(entry["latency_s"])
        self.assertEqual(entry["completion_tokens"], 7)
        self.assertFalse(entry["tokens_estimated"])
        self.assertEqual(entry["response"], "42")

    def test_wrong_answer(self):
        self.server.set_responder(lambda req, srv: (200, chat_body("99", 7), 0))
        entry = self._run()
        self.assertFalse(entry["correct"])
        self.assertEqual(entry["score"], 0.0)

    def test_token_estimation_when_usage_absent(self):
        # No usage block -> tokens are estimated and flagged.
        self.server.set_responder(lambda req, srv: (200, chat_body("42", None), 0))
        entry = self._run()
        self.assertTrue(entry["tokens_estimated"])
        self.assertIsNotNone(entry["completion_tokens"])

    def test_api_error_recorded_not_raised(self):
        self.server.set_responder(lambda req, srv: (500, {"error": "x"}, 0))
        entry = self._run()
        self.assertFalse(entry["correct"])
        self.assertIsNone(entry["latency_s"])
        self.assertIsNone(entry["response"])
        self.assertIn("HTTP 500", entry["error"])


if __name__ == "__main__":
    unittest.main()
