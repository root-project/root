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

    def test_rdataframe(self):
        df = ROOT.ROOT.RDataFrame("tree", "file.root")
        self.assertEqual(str(df), "A data frame built on top of the tree dataset.")


if __name__ == '__main__':
    unittest.main()
