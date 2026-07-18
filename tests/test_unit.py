"""Unit tests: pure functions in benchmark.py (no network, no subprocess).

Covers response cleaning, text/number/JSON extraction, every grading type,
question validation, multi-file loading (dedupe/hash/filter), and summary
statistics.
"""
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark  # noqa: E402


def q(gtype, **grading):
    grading["type"] = gtype
    return {"id": "q", "category": "c", "prompt": "p", "grading": grading}


class CleanResponseTests(unittest.TestCase):
    def test_strips_think_block(self):
        self.assertEqual(benchmark.clean_response("<think>reasoning here</think>42"), "42")

    def test_strips_multiline_think_case_insensitive(self):
        self.assertEqual(benchmark.clean_response("<THINK>\na\nb\n</THINK>\nhi"), "hi")

    def test_strips_plain_code_fence(self):
        self.assertEqual(benchmark.clean_response("```\nhello\n```"), "hello")

    def test_strips_language_fence(self):
        self.assertEqual(benchmark.clean_response("```python\nx = 1\n```"), "x = 1")

    def test_plain_text_trimmed(self):
        self.assertEqual(benchmark.clean_response("  hi  "), "hi")

    def test_none_is_empty(self):
        self.assertEqual(benchmark.clean_response(None), "")


class NormalizeTextTests(unittest.TestCase):
    def test_lowercases_and_trims_punctuation(self):
        self.assertEqual(benchmark.normalize_text('  "Hello". '), "hello")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(benchmark.normalize_text("a\t  b\nc"), "a b c")

    def test_case_sensitive_preserves_case(self):
        self.assertEqual(benchmark.normalize_text("Hello", case_sensitive=True), "Hello")


class ExtractNumberTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(benchmark.extract_last_number("the answer is 42"), 42.0)

    def test_returns_last(self):
        self.assertEqual(benchmark.extract_last_number("1 then 2 then 3"), 3.0)

    def test_thousands_separator(self):
        self.assertEqual(benchmark.extract_last_number("about 1,024 tokens"), 1024.0)

    def test_negative_and_decimal(self):
        self.assertEqual(benchmark.extract_last_number("value -3.5"), -3.5)

    def test_none_when_absent(self):
        self.assertIsNone(benchmark.extract_last_number("no digits here"))


class ExtractJsonTests(unittest.TestCase):
    def test_object(self):
        self.assertEqual(benchmark.extract_json('{"a": 1}'), {"a": 1})

    def test_object_with_surrounding_text(self):
        self.assertEqual(benchmark.extract_json('here: {"a": 1} done'), {"a": 1})

    def test_array(self):
        self.assertEqual(benchmark.extract_json("[1, 2, 3]"), [1, 2, 3])

    def test_nested(self):
        self.assertEqual(benchmark.extract_json('{"a": {"b": 2}} trailing'), {"a": {"b": 2}})

    def test_none_when_not_json(self):
        self.assertIsNone(benchmark.extract_json("definitely not json"))


class GradeNumericTests(unittest.TestCase):
    def test_correct(self):
        ok, note, score = benchmark.grade(q("numeric", answer=42), "42")
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_wrong(self):
        ok, note, score = benchmark.grade(q("numeric", answer=42), "43")
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_tolerance(self):
        self.assertTrue(benchmark.grade(q("numeric", answer=100, tolerance=5), "103")[0])
        self.assertFalse(benchmark.grade(q("numeric", answer=100, tolerance=5), "110")[0])

    def test_no_number_found(self):
        ok, note, score = benchmark.grade(q("numeric", answer=42), "no number")
        self.assertFalse(ok)
        self.assertIn("no number", note)


class GradeExactContainsTests(unittest.TestCase):
    def test_exact_case_insensitive_default(self):
        self.assertTrue(benchmark.grade(q("exact", answer="Paris"), "paris")[0])

    def test_exact_case_sensitive(self):
        self.assertFalse(benchmark.grade(q("exact", answer="ABC", case_sensitive=True), "abc")[0])
        self.assertTrue(benchmark.grade(q("exact", answer="ABC", case_sensitive=True), "ABC")[0])

    def test_contains(self):
        self.assertTrue(benchmark.grade(q("contains", answer="fox"), "the quick brown fox")[0])
        self.assertFalse(benchmark.grade(q("contains", answer="cat"), "the quick brown fox")[0])


class GradeRegexTests(unittest.TestCase):
    def test_match_case_insensitive_default(self):
        self.assertTrue(benchmark.grade(q("regex", pattern="^hola$"), "Hola")[0])

    def test_case_sensitive(self):
        self.assertFalse(benchmark.grade(q("regex", pattern="^hola$", case_sensitive=True), "Hola")[0])


class GradeJsonTests(unittest.TestCase):
    def test_value_equality_ignores_key_order(self):
        self.assertTrue(benchmark.grade(q("json", expected={"a": 1, "b": 2}), '{"b": 2, "a": 1}')[0])

    def test_mismatch(self):
        self.assertFalse(benchmark.grade(q("json", expected={"a": 1}), '{"a": 2}')[0])

    def test_no_json(self):
        ok, note, score = benchmark.grade(q("json", expected={"a": 1}), "plain text")
        self.assertFalse(ok)
        self.assertIn("no valid JSON", note)


class GradeKeywordsTests(unittest.TestCase):
    def setUp(self):
        self.question = q(
            "keywords",
            required=["rollback", ["pagerduty", "on-call"]],
            optional=["logs"],
            min_required=2,
        )

    def test_full_credit(self):
        ok, note, score = benchmark.grade(
            self.question, "we rollback and page on-call and check logs")
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_partial_credit_below_min_required_fails(self):
        ok, note, score = benchmark.grade(self.question, "just rollback")
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 1 / 3, places=4)
        self.assertIn("1/2", note)

    def test_alternative_phrasing_counts(self):
        # 'pagerduty' alternative satisfies the same required slot as 'on-call'
        ok, note, score = benchmark.grade(self.question, "rollback then alert pagerduty")
        self.assertTrue(ok)


class GradeUnknownTypeTests(unittest.TestCase):
    def test_raises(self):
        with self.assertRaises(ValueError):
            benchmark.grade({"grading": {"type": "bogus"}}, "x")


class ValidateQuestionTests(unittest.TestCase):
    def test_ok(self):
        benchmark.validate_question(q("numeric", answer=1), "src")  # no raise

    def test_missing_field(self):
        with self.assertRaises(ValueError):
            benchmark.validate_question({"id": "x", "category": "c", "prompt": "p"}, "src")

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            benchmark.validate_question(q("mystery"), "src")

    def test_numeric_needs_answer(self):
        with self.assertRaises(ValueError):
            benchmark.validate_question(q("numeric"), "src")

    def test_regex_needs_pattern(self):
        with self.assertRaises(ValueError):
            benchmark.validate_question(q("regex"), "src")

    def test_json_needs_expected(self):
        with self.assertRaises(ValueError):
            benchmark.validate_question(q("json"), "src")

    def test_keywords_needs_required(self):
        with self.assertRaises(ValueError):
            benchmark.validate_question(q("keywords"), "src")


class LoadQuestionsTests(unittest.TestCase):
    def _write(self, questions):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump({"questions": questions}, f)
        self.addCleanup(os.remove, path)
        return path

    def test_merge_two_files(self):
        a = self._write([q("numeric", answer=1)])
        b = self._write([dict(q("exact", answer="x"), id="q2")])
        questions, qhash = benchmark.load_questions([a, b])
        self.assertEqual(len(questions), 2)
        self.assertEqual(len(qhash), 16)

    def test_duplicate_id_across_files_raises(self):
        a = self._write([q("numeric", answer=1)])
        b = self._write([q("numeric", answer=2)])  # same id 'q'
        with self.assertRaises(ValueError) as ctx:
            benchmark.load_questions([a, b])
        self.assertIn("duplicate question id", str(ctx.exception))

    def test_hash_is_order_independent(self):
        q1 = q("numeric", answer=1)
        q2 = dict(q("exact", answer="x"), id="q2")
        a = self._write([q1, q2])
        b = self._write([q2, q1])
        _, ha = benchmark.load_questions([a])
        _, hb = benchmark.load_questions([b])
        self.assertEqual(ha, hb)

    def test_category_filter_and_limit(self):
        items = [
            dict(q("numeric", answer=1), id="m1", category="math"),
            dict(q("numeric", answer=2), id="m2", category="math"),
            dict(q("numeric", answer=3), id="l1", category="logic"),
        ]
        path = self._write(items)
        math_only, _ = benchmark.load_questions([path], category_filter="math")
        self.assertEqual([x["id"] for x in math_only], ["m1", "m2"])
        limited, _ = benchmark.load_questions([path], limit=1)
        self.assertEqual(len(limited), 1)

    def test_invalid_question_raises_on_load(self):
        path = self._write([q("regex")])  # regex without pattern
        with self.assertRaises(ValueError):
            benchmark.load_questions([path])

    def test_builtin_alias_resolves(self):
        self.assertEqual(
            benchmark.resolve_questions_path("built-in"), benchmark.DEFAULT_QUESTIONS_FILE)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"id": "a", "category": "x", "correct": True, "score": 1.0,
             "latency_s": 1.0, "tokens_per_sec": 10.0, "tokens_estimated": False},
            {"id": "b", "category": "x", "correct": False, "score": 0.0,
             "latency_s": 2.0, "tokens_per_sec": 20.0, "tokens_estimated": False},
            {"id": "c", "category": "y", "correct": True, "score": 0.5,
             "latency_s": 3.0, "tokens_per_sec": 30.0, "tokens_estimated": True},
            {"id": "d", "category": "y", "correct": False, "score": 0.0,
             "error": "connection error", "latency_s": None, "tokens_per_sec": None},
        ]
        self.summary = benchmark.summarize(self.results)

    def test_accuracy_and_counts(self):
        self.assertEqual(self.summary["accuracy_pct"], 50.0)
        self.assertEqual(self.summary["correct"], 2)
        self.assertEqual(self.summary["total"], 4)

    def test_error_count(self):
        self.assertEqual(self.summary["errors"], 1)

    def test_by_category(self):
        self.assertEqual(self.summary["by_category"]["x"]["pct"], 50.0)
        self.assertEqual(self.summary["by_category"]["y"]["total"], 2)

    def test_avg_score(self):
        self.assertEqual(self.summary["avg_score_pct"], 37.5)

    def test_latency_stats(self):
        lat = self.summary["latency_s"]
        self.assertEqual(lat["mean"], 2.0)
        self.assertEqual(lat["min"], 1.0)
        self.assertEqual(lat["max"], 3.0)

    def test_tokens_per_sec_and_estimated_flag(self):
        self.assertEqual(self.summary["tokens_per_sec"]["mean"], 20.0)
        self.assertTrue(self.summary["tokens_per_sec"]["estimated"])


class MiscHelperTests(unittest.TestCase):
    def test_estimate_tokens_positive(self):
        self.assertGreaterEqual(benchmark.estimate_tokens("one two three"), 1)

    def test_sanitize_filename(self):
        self.assertEqual(benchmark.sanitize_filename("qwen2.5:14b/x"), "qwen2.5_14b_x")

    def test_uses_partial_credit(self):
        self.assertTrue(benchmark.uses_partial_credit({"accuracy_pct": 50.0, "avg_score_pct": 37.5}))
        self.assertFalse(benchmark.uses_partial_credit({"accuracy_pct": 50.0, "avg_score_pct": 50.0}))
        self.assertFalse(benchmark.uses_partial_credit({"avg_score_pct": None}))

    def test_fmt_delta_direction(self):
        self.assertIn("UP", benchmark.fmt_delta(3.0, "%"))
        self.assertIn("DOWN", benchmark.fmt_delta(-3.0, "%"))
        self.assertIn("DOWN", benchmark.fmt_delta(3.0, "s", higher_is_better=False))


if __name__ == "__main__":
    unittest.main()
