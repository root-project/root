import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest


class RegressionPickleNoCppyy(unittest.TestCase):
    """
    Reading back a pickled ROOT object in a fresh interpreter, where the cppyy
    backend has not been imported yet, must not crash.

    Unpickling a ROOT object first imports the module of the function that
    reads it back. When that was a CPython extension module, which did not
    initialize the cppyy backend, entering CPyCppyy through the public API
    used to leave ``gThisModule`` null, so that ``CreateScopeProxy`` crashed
    with a segmentation violation when using it as a fake scope.

    See the ROOT forum report:
    https://root-forum.cern.ch/t/issue-with-new-root-version-on-lxplus
    """

    def test_load_without_prior_cppyy_import(self):
        import ROOT

        # Create a pickle of a ROOT object. Touching a class here imports the
        # cppyy backend, but only in this (throw-away) process.
        h = ROOT.TH1F("h", "h", 10, 0, 1)
        h.Fill(0.5)

        fname = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name
        try:
            with open(fname, "wb") as f:
                pickle.dump(h, f)

            # Read it back in a *fresh* interpreter that only imports pickle, so
            # that the cppyy backend is not initialized before the object is
            # expanded. This reproduces the exact scenario from the bug report.
            code = textwrap.dedent(
                """
                import pickle
                with open({fname!r}, "rb") as f:
                    h = pickle.load(f)
                assert h.GetName() == "h", h.GetName()
                assert h.GetEntries() == 1.0, h.GetEntries()
                """
            ).format(fname=fname)

            # Do not use check=True so we can give a helpful message on crash
            # (a segfault shows up as a negative return code on POSIX).
            proc = subprocess.run([sys.executable, "-c", code])
            self.assertEqual(
                proc.returncode,
                0,
                "Unpickling a ROOT object in a fresh interpreter crashed (return code {}).".format(proc.returncode),
            )
        finally:
            os.remove(fname)


if __name__ == "__main__":
    unittest.main()
