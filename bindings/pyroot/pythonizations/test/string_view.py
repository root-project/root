import os
import unittest

import ROOT


class StringView(unittest.TestCase):
    """
    Test the availability of std::string_view
    """

    def test_interpreter(self):
        ROOT.gInterpreter.Declare("std::string TestStringView(std::string_view x) { return std::string(x); }")
        x = ROOT.TestStringView("foo")
        self.assertEqual(str(x), "foo")

    def _check_rdf_constructor_taking_string_view(self, treename, filename):
        # Create file.root to avoid errors in the RDF constructor
        with ROOT.TFile(filename, "recreate") as f:
            _ = ROOT.TTree(treename, treename)
            f.Write()
        df = ROOT.RDataFrame(treename, filename)
        self.assertEqual(df.GetNRuns(), 0)

    def test_rdataframe(self):
        try:
            treename = "tree"
            filename = "string_view_backport_test_file.root"
            self._check_rdf_constructor_taking_string_view(treename, filename)
        finally:
            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
