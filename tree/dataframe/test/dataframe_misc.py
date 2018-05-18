import unittest
import ROOT

class MiscTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load all libs and autoparse
        df=ROOT.ROOT.RDataFrame(1)

    def test_lazy_define(self):
        df=ROOT.ROOT.RDataFrame(10)
        ROOT.gInterpreter.ProcessLine('int myCount = 0;')
        h = df.Define('a', 'static int i = 0; return i++;')\
              .Filter('a > 100')\
              .Define('xx', ' cout << "This should not be triggered!!" << endl; myCount++; return 1;')\
              .Histo1D('xx') # this is to check if the define is triggered!
        h.GetMean()
        self.assertEqual(0, ROOT.myCount)

if __name__ == '__main__':
    unittest.main()
