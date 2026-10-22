#!/usr/bin/env python3

"""Unit tests for CTest failure annotation logic in build_utils.py.

Usage:

    python3 test_annotate_failures.py
"""

import os
import tempfile
import unittest

import build_utils


class AnnotateCTestFailuresTest(unittest.TestCase):
    def test_extract_annotations_gtest(self):
        failed_log = "1:gtest-core-foundation\n"
        test_log = """
1/1 Testing: gtest-core-foundation
----------------------------------------------------------
/ROOT-CI/src/core/foundation/test/TStringTest.cxx:88: Failure
Value of: str.Length()
Actual: 0
Expected: 5
----------------------------------------------------------
Test Failed.
"""
        annotations = build_utils.extract_ctest_annotations(failed_log, test_log, "/ROOT-CI/src")
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["file"], "core/foundation/test/TStringTest.cxx")
        self.assertEqual(annotations[0]["line"], "88")
        self.assertEqual(annotations[0]["title"], "gtest-core-foundation")
        self.assertIn("Value of: str.Length()", annotations[0]["message"])

    def test_extract_annotations_compiler_macro(self):
        failed_log = "2:roottest-root-meta-test\n"
        test_log = """
1/1 Testing: roottest-root-meta-test
----------------------------------------------------------
/ROOT-CI/src/roottest/root/meta/test.C:42:15: error: use of undeclared identifier 'foo'
----------------------------------------------------------
Test Failed.
"""
        annotations = build_utils.extract_ctest_annotations(failed_log, test_log, "/ROOT-CI/src")
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["file"], "roottest/root/meta/test.C")
        self.assertEqual(annotations[0]["line"], "42")
        self.assertEqual(annotations[0]["title"], "roottest-root-meta-test")
        self.assertIn("use of undeclared identifier 'foo'", annotations[0]["message"])

    def test_extract_annotations_python_traceback(self):
        failed_log = "3:roottest-python-import\n"
        test_log = """
1/1 Testing: roottest-python-import
----------------------------------------------------------
Traceback (most recent call last):
  File "/ROOT-CI/src/bindings/pyroot/test/test_import.py", line 15, in test_something
    self.assertEqual(a, b)
AssertionError: 1 != 2
----------------------------------------------------------
Test Failed.
"""
        annotations = build_utils.extract_ctest_annotations(failed_log, test_log, "/ROOT-CI/src")
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["file"], "bindings/pyroot/test/test_import.py")
        self.assertEqual(annotations[0]["line"], "15")
        self.assertEqual(annotations[0]["title"], "roottest-python-import")
        self.assertIn("AssertionError: 1 != 2", annotations[0]["message"])

    def test_extract_annotations_unknown_failure(self):
        failed_log = "4:roottest-mystery-crash\n"
        test_log = """
1/1 Testing: roottest-mystery-crash
----------------------------------------------------------
Process killed by signal 11
----------------------------------------------------------
Test Failed.
"""
        annotations = build_utils.extract_ctest_annotations(failed_log, test_log, "/ROOT-CI/src")
        self.assertEqual(len(annotations), 1)
        self.assertIsNone(annotations[0]["file"])
        self.assertIsNone(annotations[0]["line"])
        self.assertEqual(annotations[0]["title"], "roottest-mystery-crash")

    def test_annotate_ctest_failures_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_build = os.path.join(tmpdir, "build", "Testing", "Temporary")
            os.makedirs(temp_build)
            with open(os.path.join(temp_build, "LastTestsFailed.log"), "w", encoding="utf-8") as f:
                f.write("1:test-sample\n")
            with open(os.path.join(temp_build, "LastTest.log"), "w", encoding="utf-8") as f:
                f.write("""
1/1 Testing: test-sample
----------------------------------------------------------
/dummy/src/roottest/sample.C:10: error: failed assertion
----------------------------------------------------------
Test Failed.
""")
            # Should run without error
            build_utils.annotate_ctest_failures(tmpdir)


if __name__ == "__main__":
    unittest.main()
