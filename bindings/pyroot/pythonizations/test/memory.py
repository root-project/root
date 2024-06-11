import gc
import ROOT
import unittest


class MemoryStlString(unittest.TestCase):

    def test_15703(self):
        """Regression test for https://github.com/root-project/root/issues/15703"""

        ROOT.gInterpreter.Declare("""
        #include <string>
        class foo {
            public:
            const std::string leak (std::size_t size) const {
                std::string result;
                result.reserve(size);
                return result;
            }
        };
                                  
        auto get_rss_MB() {
            ProcInfo_t info;
            gSystem->GetProcInfo(&info);
            return info.fMemResident;
        }
        """)
        obj = ROOT.foo()
        for _ in range(1000000):
            __ = obj.leak(2048)

        gc.collect()

        before = ROOT.get_rss_MB()

        for _ in range(1000000):
            __ = obj.leak(2048)

        gc.collect()
        after = ROOT.get_rss_MB()
        delta = after - before
        self.assertLess(delta, 16)


if __name__ == '__main__':
    unittest.main()
