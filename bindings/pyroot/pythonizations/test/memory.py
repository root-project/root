import ROOT
import gc
import os
import unittest


def _leak(obj):
    gc.collect()
    for _ in range(1000000):
        __ = obj.leak(2048)
    gc.collect()


class MemoryStlString(unittest.TestCase):

    def test_15703(self):
        """Regression test for https://github.com/root-project/root/issues/15703"""

        ROOT.gInterpreter.Declare("""
        #ifndef TEST_15703
        #define TEST_15703
        #include <string>
        class foo {
            public:
            const std::string leak (std::size_t size) const {
                std::string result;
                result.reserve(size);
                return result;
            }
        };

        auto get_rss_KB() {
            ProcInfo_t info;
            gSystem->GetProcInfo(&info);
            return info.fMemResident;
        }
        #endif // TEST_15703
        """)
        obj = ROOT.foo()
        _leak(obj)
        before = ROOT.get_rss_KB()
        _leak(obj)
        after = ROOT.get_rss_KB()
        delta = after - before
        self.assertLess(delta, 16)

    def test_tstyle_memory_management(self):
        """Regression test for https://github.com/root-project/root/issues/16918"""

        h1 = ROOT.TH1F("h1", "", 100, 0, 10)

        style = ROOT.TStyle("NewSTYLE", "")
        groot = ROOT.ROOT.GetROOT()
        groot.SetStyle(style.GetName())
        groot.ForceStyle()

    def test_tf2_memory_regulation(self):
        """Regression test for https://github.com/root-project/root/issues/16942"""
        # The test is just that the memory regulation works correctly and the
        # application does not segfault
        f2 = ROOT.TF2("f2", "sin(x)*sin(y)/x/y")

    def test_tf3_memory_regulation(self):
        """Make sure TF3 is properly managed by the memory regulation logic"""
        # The test is just that the memory regulation works correctly and the
        # application does not segfault
        f3 = ROOT.TF3("f3","[0] * sin(x) + [1] * cos(y) + [2] * z",0,10,0,10,0,10)

    def test_tcolor_memory_regulation(self):
        """Make sure TColor is properly managed by the memory regulation logic"""
        # The test is just that the memory regulation works correctly and the
        # application does not segfault
        c = ROOT.TColor(42, 42, 42)

    def test_ttree_clone_in_file_context(self):
        """Test that CloneTree() doesn't give the ownership to Python when
        TFile is opened."""

        filename = "test_ttree_clone_in_file_context"

        ttree = ROOT.TTree("tree", "tree")

        with ROOT.TFile(filename, "RECREATE") as infile:
            ttree_clone = ttree.CloneTree()

        os.remove(filename)

    def _check_object_in_subdir(self, klass, args):
        """
        Test that an object which automatically registers with a subdirectory
        does not give ownership to Python
        """
        filename = "test_object_in_subdir.root"
        try:
            with ROOT.TFile(filename, "recreate") as f:
                f.mkdir("subdir")
                f.cd("subdir")
                x = klass(*args)
                x.Write()
        finally:
            os.remove(filename)

    def test_objects_ownership_with_subdir(self):
        """
        Test interaction of various types of objects with automatic directory
        registration with a subdirectory of a TFile.
        """

        objs = {
            "TH1D": ("h", "h", 10, 0, 10),
            "TH1C": ("h", "h", 10, 0, 10),
            "TH1S": ("h", "h", 10, 0, 10),
            "TH1I": ("h", "h", 10, 0, 10),
            "TH1L": ("h", "h", 10, 0, 10),
            "TH1F": ("h", "h", 10, 0, 10),
            "TH1D": ("h", "h", 10, 0, 10),
            "TH1K": ("h", "h", 10, 0, 10),
            "TProfile": ("h", "h", 10, 0, 10),
            "TH2C": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2S": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2I": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2L": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2F": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2D": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2Poly": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH2PolyBin": tuple(),
            "TProfile2D": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TProfile2PolyBin": tuple(),
            "TProfile2Poly": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH3C": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TH3S": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TH3I": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TH3L": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TH3F": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TH3D": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TProfile3D": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TGraph2D": (100,),
            "TEntryList": ("name", "title"),
            "TEventList": ("name", "title"),
            "TTree": ("name", "title"),
            "TNtuple": ("name", "title", "x:y:z"),
        }
        for klass, args in objs.items():
            with self.subTest(klass=klass):
                self._check_object_in_subdir(getattr(ROOT, klass), args)

    def _check_object_setdirectory(self, klass, classname, args):
        """
        Test that registering manually an object with a directory also triggers
        a release of ownership from Python to C++.
        """
        f1 = ROOT.TMemFile(
            "_check_object_setdirectory_in_memory_file_begin", "recreate")

        x = klass(*args)
        # TEfficiency does not automatically register with the directory
        if not classname == "TEfficiency":
            self.assertIs(x.GetDirectory(), f1)
            x.SetDirectory(ROOT.nullptr)
        self.assertFalse(x.GetDirectory())
        # Make sure that at this point the ownership of the object is with Python
        ROOT.SetOwnership(x, True)

        f1.Close()

        f2 = ROOT.TMemFile("_check_object_setdirectory_in_memory_file_end", "recreate")

        # The pythonization should trigger the release of ownership to C++
        x.SetDirectory(f2)
        self.assertIs(x.GetDirectory(), f2)

        f2.Close()

    def test_objects_interaction_with_setdirectory(self):
        """
        Test interaction of various types of objects with manual registration
        to a directory.
        """

        objs = {
            "TH1D": ("h", "h", 10, 0, 10),
            "TH2D": ("h", "h", 10, 0, 10, 10, 0, 10),
            "TH3D": ("h", "h", 10, 0, 10, 10, 0, 10, 10, 0, 10),
            "TGraph2D": (100,),
            "TEfficiency": (ROOT.TH1D("h1", "h1", 10, 0, 10), ROOT.TH1D("h2", "h2", 10, 0, 10)),
            "TEntryList": ("name", "title"),
            "TEventList": ("name", "title"),
            "TTree": ("name", "title"),
            "TNtuple": ("name", "title", "x:y:z"),
        }
        for classname, args in objs.items():
            with self.subTest(classname=classname):
                self._check_object_setdirectory(getattr(ROOT, classname), classname, args)


if __name__ == '__main__':
    unittest.main()
