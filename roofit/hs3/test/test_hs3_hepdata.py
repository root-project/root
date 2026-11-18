"""Test RooFit with the HS3 documents that are published on HEPData.

The documents are run through the runner of the HS3 test suite, with the
fixtures in roofit/hs3/test/hs3hepdata: for every document, the suite checks
that RooFit imports it and that its likelihood has the expected values at
fixed parameter points. On top of that, this test checks that RooFit can export
the imported workspace and import the result again and, if asked to, fits the
likelihood and profiles it over its parameters of interest. The fits are slow,
and make the test usable as a benchmark: the time of every step is written to
timings.json in the cache directory.

The documents are not part of the fixtures. They are downloaded from HEPData
into a cache directory when the test runs, and not downloaded again. The test
is therefore only enabled with the CMake option -Dtest_roofit_hs3hepdata=ON,
and is not meant to run in the CI: HEPData should not be queried on every build.

Environment variables:

HS3HEPDATA_CACHE
    The cache directory (required).
HS3TESTSUITECODE_ROOT
    A checkout of the runner of the HS3 test suite (HS3TestSuiteCode).
HS3HEPDATA_FIXTURES
    The fixtures directory, by default the directory hs3hepdata next to this file.
HS3HEPDATA_EXPECTED_FAILURES
    The list of checks that are known to fail.
HS3HEPDATA_FITS
    If set to 1, the fits are run.
HS3HEPDATA_EVAL_BACKEND
    The evaluation backend of RooFit for the fits, e.g. "codegen".
HS3HEPDATA_OFFLINE
    If set to 1, HEPData is not contacted: documents that are not in the cache are skipped.
HS3HEPDATA_SELECT
    A regular expression. Only the documents whose test_id matches it are tested.
"""

import os
import re
import sys
import unittest
from pathlib import Path

if os.environ.get("HS3TESTSUITECODE_ROOT"):
    sys.path.insert(0, os.environ["HS3TESTSUITECODE_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def flag(name):
    return os.environ.get(name, "") not in ("", "0")


def make_test_case():
    namespace = dict(__module__=__name__)

    def add(name, function):
        function.__name__ = name
        namespace[name] = function

    def finish():
        return type("TestHS3HEPData", (unittest.TestCase,), namespace)

    def failing(name, message):
        add(name, lambda self: self.fail(message))
        return finish()

    try:
        import hs3hepdata_common as common
        from hs3hepdata_roofit import HEPDataBackend
        from hs3suite.runner import run_suite
    except ImportError as e:
        return failing("test_dependencies", f"Missing dependency: {e}")

    if not os.environ.get("HS3HEPDATA_CACHE"):
        return failing("test_configuration", "HS3HEPDATA_CACHE is not set")
    cache = Path(os.environ["HS3HEPDATA_CACHE"])
    fixtures = Path(os.environ.get("HS3HEPDATA_FIXTURES") or Path(__file__).resolve().parent / "hs3hepdata")
    offline = flag("HS3HEPDATA_OFFLINE")
    select = os.environ.get("HS3HEPDATA_SELECT")
    expected_failures = load_expected_failures(os.environ.get("HS3HEPDATA_EXPECTED_FAILURES"))

    documents = common.load_json(fixtures / common.DOCUMENTS_FILE)["documents"]
    manifest = common.load_json(fixtures / "manifest.json")
    selected = [d for d in documents if not select or re.search(select, d["test_id"])]

    # The fits of the fixtures, and all checks that the fixtures define, whether they are run or not.
    all_fits = {}
    known_checks = set()
    for entry in manifest["fixtures"]:
        test_id = entry["test_id"]
        known_checks.update(f"{test_id}::{check}" for check in entry["checks"] + ["round_trip"])
        fit_file = fixtures / entry["path"] / "expected_fit.json"
        if fit_file.is_file():
            all_fits[test_id] = common.load_json(fit_file)["fits"]
            known_checks.update(f"{test_id}::{fit['id']}" for fit in all_fits[test_id])
    fits = all_fits if flag("HS3HEPDATA_FITS") else {}

    try:
        available = common.prepare_documents(cache, selected, offline)
    except Exception as e:
        return failing("test_hepdata_download", f"could not get the HS3 documents from HEPData: {e}")
    available_ids = {document["test_id"] for document in available}
    for document in selected:
        if document["test_id"] not in available_ids:

            def missing(self, offline=offline):
                if offline:
                    self.skipTest("the document is not in the cache")
                self.fail("the document is not on HEPData any more, or not in its resource")

            add("test_" + re.sub(r"[^A-Za-z0-9]", "_", document["test_id"]) + "__download", missing)

    backend = HEPDataBackend(fits)
    try:
        root = cache / "suite-root"
        common.assemble_suite_root(root, fixtures, cache, available)
        results = [(r.test_id, r.check_id, r.status, r.message) for r in run_suite(root, backend=backend)]
    except Exception as e:
        return failing("test_hs3testsuite_execution", f"the HS3 test suite failed to run: {e}")
    results += backend.extra
    common.write_json(cache / "timings.json", backend.seconds)

    for test_id, check_id, status, message in results:

        def make(status=status, message=message):
            def test(self):
                if status in ("failed", "xfail"):
                    self.fail(message)

            return test

        key = f"{test_id}::{check_id}"
        test = make()
        test.__doc__ = key
        # Known failures are listed in hs3hepdata_expected_failures.txt. An unexpected success
        # fails the test run, so the list has to be shrunk as soon as a check starts to pass.
        if status == "xfail" or key in expected_failures:
            test = unittest.expectedFailure(test)
        name = "test_" + re.sub(r"[^A-Za-z0-9]", "_", test_id) + "__" + re.sub(r"[^A-Za-z0-9]", "_", check_id)
        test.__name__ = name
        namespace[name] = test

    if not select:
        # Entries that don't correspond to any check would silently hide nothing, so they are
        # reported as failures.
        stale = sorted(set(expected_failures).difference(known_checks))

        def test_expected_failures_list_is_current(self):
            self.assertEqual(stale, [], "entries in the list of expected failures that match no check")

        add(test_expected_failures_list_is_current.__name__, test_expected_failures_list_is_current)

    if not select and not offline:
        # HS3 documents on HEPData without fixtures are not tested, which should not go unnoticed.
        try:
            known = {document["resource_id"] for document in documents}
            new = [r for r in common.query_resources() if r["resource_id"] not in known]
        except Exception as e:
            return failing("test_hepdata_query", f"could not ask HEPData for its HS3 documents: {e}")

        def test_fixtures_cover_hepdata(self):
            self.assertEqual(
                [f"INSPIRE {r['inspire_id']}: {r['filename']}" for r in new],
                [],
                "HS3 documents on HEPData without fixtures: run hs3hepdata_generate_fixtures.py",
            )

        add(test_fixtures_cover_hepdata.__name__, test_fixtures_cover_hepdata)

    return finish()


TestHS3HEPData = make_test_case()

if __name__ == "__main__":
    unittest.main()
