"""Data-integrity tests for the shipped questions.json bank.

These guard the question bank itself, which is the file most likely to be
edited on the airgapped network. They catch malformed JSON, missing/invalid
grading fields, duplicate ids, uncompilable regexes, and -- most usefully --
questions whose own canonical answer does not grade as correct (a sign the
expected value or grading config is wrong).
"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark  # noqa: E402


class BankIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions, cls.qhash = benchmark.load_questions(
            [benchmark.DEFAULT_QUESTIONS_FILE])

    def test_bank_is_valid_json(self):
        with open(benchmark.DEFAULT_QUESTIONS_FILE) as f:
            json.load(f)  # raises on malformed JSON

    def test_loads_and_validates(self):
        # load_questions runs validate_question on every entry; reaching here
        # means all questions have the required fields and known grading types.
        self.assertGreater(len(self.questions), 0)

    def test_ids_unique(self):
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_regex_patterns_compile(self):
        for q in self.questions:
            if q["grading"]["type"] == "regex":
                try:
                    re.compile(q["grading"]["pattern"])
                except re.error as e:
                    self.fail("bad regex in {}: {}".format(q["id"], e))

    def test_deterministic_answers_self_grade(self):
        """Every numeric/exact/json question must grade its own answer correct.

        This is the key regression guard for the bank: if someone changes a
        prompt or an expected value inconsistently, the canonical answer stops
        grading as correct and this test fails, naming the question.
        """
        for q in self.questions:
            g = q["grading"]
            gtype = g["type"]
            if gtype in ("numeric", "exact", "contains"):
                resp = str(g["answer"])
            elif gtype == "json":
                resp = json.dumps(g["expected"])
            else:
                continue  # regex/keywords have no single canonical answer
            ok, note, score = benchmark.grade(q, resp)
            self.assertTrue(ok, "{} did not self-grade correct: {}".format(q["id"], note))

    def test_expected_categories_present(self):
        cats = {q["category"] for q in self.questions}
        for expected in ("math", "logic", "code", "python", "c", "cpp",
                         "vulnre", "ghidra", "factual", "instruction"):
            self.assertIn(expected, cats)


if __name__ == "__main__":
    unittest.main()
