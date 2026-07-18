#!/usr/bin/env python3
"""Run the llm-benchmark test suite. Standard library only, fully offline.

    python3 run_tests.py            # run everything
    python3 run_tests.py -v         # verbose
    python3 run_tests.py test_unit  # run a single test module by name

Equivalent to `python3 -m unittest discover -s tests -t .`, but self-contained
and exit-code friendly for CI. Requires no network access: the integration and
system tests spin up a loopback-only mock LLM server.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(ROOT, "tests")
for _p in (ROOT, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv):
    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    names = [a for a in argv if not a.startswith("-")]

    loader = unittest.TestLoader()
    if names:
        suite = loader.loadTestsFromNames(names)
    else:
        suite = loader.discover(TESTS, pattern="test_*.py", top_level_dir=ROOT)

    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
