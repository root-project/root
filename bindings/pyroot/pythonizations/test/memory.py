import gc
import ROOT
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

if __name__ == '__main__':
    unittest.main()
