import os
import unittest
import ROOT


class StringView(unittest.TestCase):
    """
    Test the availability of std::string_view
    """

    def test_interpreter(self):
        ROOT.gInterpreter.Declare(
            "std::string TestStringView(std::string_view x) { return std::string(x); }")
        x = ROOT.TestStringView("foo")
        self.assertEqual(str(x), "foo")

    def test_rdataframe(self):
        # Create file.root to avoid errors in the RDF constructor
        treename = "tree"
        filename = "string_view_backport_test_file.root"

        with ROOT.TFile(filename, "recreate") as f:
            t = ROOT.TTree(treename, treename)
            f.WriteObject(t, treename)

        df = ROOT.RDataFrame(treename, filename)
        self.assertEqual(
            str(df), "A data frame built on top of the tree dataset.")
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
