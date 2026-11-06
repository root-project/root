import os
import sys
import unittest
from pathlib import Path

hs3_root_dir = os.environ.get("HS3TESTSUITE_ROOT")
hs3_code_dir = os.environ.get("HS3TESTSUITECODE_ROOT")
if hs3_code_dir:
    sys.path.insert(0, hs3_code_dir)


def load_expected_failures(path):
    """Read the list of expected failures.

    Every non-empty line that is not a comment has the form
    ``<test_id>::<check_id>  # <reason>``.
    """
    expected = {}
    if not path:
        return expected
    for line in Path(path).read_text().splitlines():
        entry, _, reason = line.partition("#")
        entry = entry.strip()
        if entry:
            expected[entry] = reason.strip()
    return expected


try:
    from hs3suite.runner import run_suite
except ImportError as e:
    TestHS3Suite = type(
        "TestHS3Suite",
        (unittest.TestCase,),
        dict(test_dependencies=lambda self, e=e: self.fail(f"Missing dependency: {e}")),
    )
else:
    expected_failures = load_expected_failures(os.environ.get("HS3TESTSUITE_EXPECTED_FAILURES"))
    try:
        results = run_suite(Path(hs3_root_dir), "hs3testsuite_roofit_backend:RooFitBackend")
    except Exception as e:

        def test(self, e=e):
            self.fail(f"HS3TestSuite failed to run: {e}")

        test.__name__ = "test_hs3testsuite_execution"
        TestHS3Suite = type("TestHS3Suite", (unittest.TestCase,), {test.__name__: test})
    else:
        namespace = dict(__module__=__name__)
        for r in results:

            def _make(result):
                def test(self):
                    if result.status in ("failed", "xfail"):
                        self.fail(result.message)
                    return

                key = f"{result.test_id}::{result.check_id}"
                test.__name__ = f"test_{result.test_id}__{result.check_id}"
                test.__doc__ = key
                # Known failures are listed in hs3testsuite_expected_failures.txt, or
                # marked as "xfail" for RooFit in the manifest of the HS3TestSuite. An
                # unexpected success fails the test run, so the list has to be shrunk as
                # soon as a fixture starts to pass.
                if result.status == "xfail" or key in expected_failures:
                    test = unittest.expectedFailure(test)
                return test

            func = _make(r)
            namespace[func.__name__] = func

        # Entries that don't correspond to any check (e.g. after a fixture was renamed
        # upstream) would silently hide nothing, so they are reported as failures.
        stale = sorted(set(expected_failures).difference(f"{r.test_id}::{r.check_id}" for r in results))

        def test_expected_failures_list_is_current(self):
            self.assertEqual(stale, [], "entries in the list of expected failures that match no check")

        namespace[test_expected_failures_list_is_current.__name__] = test_expected_failures_list_is_current
        TestHS3Suite = type("TestHS3Suite", (unittest.TestCase,), namespace)

if __name__ == "__main__":
    unittest.main()
