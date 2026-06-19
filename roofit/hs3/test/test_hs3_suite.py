import os
import sys
import unittest
from pathlib import Path

hs3_root_dir = os.environ.get("HS3TESTSUITE_ROOT")
if hs3_root_dir:
    sys.path.insert(0, hs3_root_dir)

try:
    from hs3suite.backends import _BACKENDS
    from hs3suite.runner import run_suite
    
    _BACKENDS['roofit'] = ("hs3testsuite_roofit_backend","RooFitBackend")

except ImportError as e:
    TestHS3Suite = type(
        "TestHS3Suite",
        (unittest.TestCase,),
        dict(test_dependencies=lambda self, e=e: self.fail(f"Missing dependency: {e}")),
    )
else:
    hs3_root_path = Path(hs3_root_dir)
    try:
        results = run_suite(hs3_root_path, "roofit")
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
                    if result.status == "failed":
                        self.fail(result.message)
                    return

                test.__name__ = f"test_{result.test_id}__{result.check_id}"
                test.__doc__ = f"{result.test_id}::{result.check_id}"
                return test

            func = _make(r)
            namespace[func.__name__] = func
        TestHS3Suite = type("TestHS3Suite", (unittest.TestCase,), namespace)

if __name__ == "__main__":
    unittest.main()
