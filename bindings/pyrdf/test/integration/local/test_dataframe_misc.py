import unittest
import ROOT
import PyRDF


class MiscTests(unittest.TestCase):
    """Misc tests"""

    @classmethod
    def setUpClass(cls):
        """Load all libs and autoparse"""
        ROOT.ROOT.RDataFrame(1)

    def test_lazy_define(self):
        """Check laziness"""
        df = PyRDF.RDataFrame(10)
        ROOT.gInterpreter.ProcessLine('int myCount = 0;')
        cppcode = 'cout << "This should not be triggered!!" << endl; ' \
                  'myCount++; return 1;'
        h = df.Define('a', 'static int i = 0; return i++;')\
              .Filter('a > 100')\
              .Define('xx', cppcode)\
              .Histo1D('xx')  # this is to check if the define is triggered!
        h.GetMean()
        self.assertEqual(0, ROOT.myCount)


if __name__ == '__main__':
    unittest.main()
