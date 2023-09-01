import unittest
import ROOT

class HistogramsFromRDF(unittest.TestCase):
    @classmethod
    def setUp(cls):
        ROOT.gRandom.SetSeed(1)

    def test_histo1D(self):
        ROOT.gRandom.SetSeed(1)
        tdf = ROOT.ROOT.RDataFrame(64)
        g = tdf.Define("r","gRandom->Gaus(0,1)")
        h1Proxy = g.Histo1D(("h1","h1",64, -2., 2.),"r")
        h1 = h1Proxy.GetValue()

        cppCode = 'gRandom->SetSeed(1);' + \
                  'ROOT::RDataFrame tdf(64);' + \
                  'auto g = tdf.Define("r","gRandom->Gaus(0,1)");' + \
                  'auto h2Proxy = g.Histo1D({"h1","h1",64, -2., 2.},"r");'
        ROOT.gInterpreter.ProcessLine(cppCode)
        h2 = ROOT.h2Proxy.GetValue()

        self.assertEqual(h1.GetEntries(), h2.GetEntries())
        self.assertEqual(h1.GetMean(), h2.GetMean())
        self.assertEqual(h1.GetStdDev(), h2.GetStdDev())

if __name__ == '__main__':
    unittest.main()
