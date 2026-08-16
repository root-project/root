#!/usr/bin/env python3

# pylint: disable=missing-function-docstring,line-too-long

"""Unit tests for null_build_check.py, guarding that unwanted rebuilds keep
being detected and reported.

Usage:

    python3 test_null_build_check.py
"""

import contextlib
import io
import os
import tempfile
import unittest

import null_build_check

# Arbitrary but deterministic timestamps: tests set file mtimes explicitly so
# they do not depend on filesystem timestamp granularity.
MTIME_FIRST_BUILD = 1_000_000_000_000_000_000
MTIME_SECOND_BUILD = 2_000_000_000_000_000_000


class NullBuildCheckTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self._tmpdir.cleanup)
        self.builddir = self._tmpdir.name

    def write(self, relpath, content="", mtime_ns=MTIME_FIRST_BUILD):
        path = os.path.join(self.builddir, *relpath.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        os.utime(path, ns=(mtime_ns, mtime_ns))

    def remove(self, relpath):
        os.remove(os.path.join(self.builddir, *relpath.split("/")))

    def check(self, rebuild):
        with contextlib.redirect_stdout(io.StringIO()):
            return null_build_check.check_null_build(self.builddir, rebuild)

    def test_untouched_tree_reports_nothing(self):
        self.write("lib/libCore.so")
        self.write("include/TObject.h")

        self.assertEqual(self.check(lambda: 0), [])

    def test_created_modified_and_deleted_files_are_reported(self):
        self.write("lib/libCore.so")
        self.write("lib/stale.o")

        def rebuild():
            self.write("lib/fresh.o", mtime_ns=MTIME_SECOND_BUILD)
            self.write("lib/libCore.so", mtime_ns=MTIME_SECOND_BUILD)
            self.remove("lib/stale.o")
            return 0

        self.assertEqual(
            self.check(rebuild),
            [
                ("lib/fresh.o", "created"),
                ("lib/libCore.so", "modified"),
                ("lib/stale.o", "deleted"),
            ],
        )

    def test_rewrite_with_identical_size_is_still_a_modification(self):
        self.write("lib/module.pcm", content="same size")

        def rebuild():
            self.write("lib/module.pcm", content="same size", mtime_ns=MTIME_SECOND_BUILD)
            return 0

        self.assertEqual(self.check(rebuild), [("lib/module.pcm", "modified")])

    def test_allowed_files_are_not_reported(self):
        allowed = (
            "etc/gitinfo.txt",
            ".ninja_log",
            ".cmake/api/v1/reply/index.json",
            "core/base/CMakeFiles/base.dir/compiler_depend.ts",
            "CMakeFiles/progress.marks",
            "tree/tree.vcxproj.tlog",
            "tree/x64/tree.tlog/link.write.1u.tlog",
            "core/Core.dir/RelWithDebInfo/Core.vcxproj.recipe",
            "CMakeFiles/generate.stamp",
            "math/mathcore/CMakeFiles/generate.stamp",
            "Testing/Temporary/LastTest.log",
        )
        for relpath in allowed:
            self.write(relpath)

        def rebuild():
            for relpath in allowed:
                self.write(relpath, mtime_ns=MTIME_SECOND_BUILD)
            return 0

        self.assertEqual(self.check(rebuild), [])

    def test_allowed_patterns_without_wildcard_are_anchored(self):
        def rebuild():
            self.write("subproject/etc/gitinfo.txt", mtime_ns=MTIME_SECOND_BUILD)
            return 0

        self.assertEqual(self.check(rebuild), [("subproject/etc/gitinfo.txt", "created")])

    def test_pruned_directories_are_ignored(self):
        self.write("builtins/openssl/.git/index")

        def rebuild():
            self.write("builtins/openssl/.git/index", mtime_ns=MTIME_SECOND_BUILD)
            self.write("builtins/openssl/.git/FETCH_HEAD", mtime_ns=MTIME_SECOND_BUILD)
            return 0

        self.assertEqual(self.check(rebuild), [])

    def test_failing_rebuild_raises(self):
        with self.assertRaises(RuntimeError):
            self.check(lambda: 1)

    def test_report_lists_the_offending_files(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            null_build_check.report(self.builddir, [("lib/libCore.so", "modified")])

        self.assertIn("1 file(s) were written", output.getvalue())
        self.assertIn("lib/libCore.so", output.getvalue())

    def test_report_on_clean_tree(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            null_build_check.report(self.builddir, [])

        self.assertIn("No spurious rebuilds", output.getvalue())

    def test_summarize_counts_per_file_type(self):
        touched = [
            ("core/base/src/TObject.cxx.o", "modified"),
            ("core/cont/src/TList.cxx.o", "modified"),
            ("lib/module.pcm", "created"),
            ("Makefile", "modified"),
        ]

        self.assertEqual(
            null_build_check.summarize(touched),
            [("*.o", 2), ("*.pcm", 1), ("Makefile", 1)],
        )


if __name__ == "__main__":
    unittest.main()
