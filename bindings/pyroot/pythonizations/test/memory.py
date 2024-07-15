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


if __name__ == '__main__':
    unittest.main()
