import os
import unittest
import ROOT
import cppyy
import platform


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
        self.assertEqual(df.GetNRuns(), 0)
        os.remove(filename)

    def test_17497(self):
        """Regression test for https://github.com/root-project/root/issues/17497"""
        # See https://github.com/root-project/root/issues/7541 and
        # https://bugs.llvm.org/show_bug.cgi?id=49692 :
        # llvm JIT fails to catch exceptions on MacOS ARM, so we disable their testing
        # Also fails on Windows 64bit for the same reason
        if (platform.processor() != "arm" or platform.mac_ver()[0] == '') and not (platform.system() == "Windows" and platform.architecture()[0] == "64bit"):
            ROOT.gInterpreter.Declare(r"""
            void fun(std::string_view, std::string_view){throw std::runtime_error("std::string_view overload");}
            void fun(std::string_view, const std::vector<std::string> &){throw std::runtime_error("const std::vector<std::string> & overload");}
            """)
            with self.assertRaises(cppyy.gbl.std.runtime_error):
                ROOT.fun("", [])
            with self.assertRaises(cppyy.gbl.std.runtime_error):
                ROOT.fun(ROOT.std.string_view("hello world"),
                         ROOT.std.vector[ROOT.std.string]())
            with self.assertRaises(cppyy.gbl.std.runtime_error):
                ROOT.fun("", ROOT.std.vector[ROOT.std.string]())
            with self.assertRaises(cppyy.gbl.std.runtime_error):
                ROOT.fun(ROOT.std.string_view("hello world"), [])

if __name__ == '__main__':
    unittest.main()
