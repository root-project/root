import unittest
import ROOT
import PyRDF


class HistogramsFromRDF(unittest.TestCase):
    """
    Same histogram operations in PyRDF with Local backend and RDF has to produce
    the same results

    """

    @classmethod
    def setUp(cls):
        """Fix the seed"""
        ROOT.gRandom.SetSeed(1)

    def test_histo1D(self):
        """Run the same code with PyRDF and RDF using the cpp interpreter"""
        ROOT.gRandom.SetSeed(1)
        rdf = PyRDF.RDataFrame(64)
        g = rdf.Define("r", "gRandom->Gaus(0,1)")
        h1Proxy = g.Histo1D(("h1", "h1", 64, -2., 2.), "r")
        h1 = h1Proxy.GetValue()

        cppCode = 'gRandom->SetSeed(1);' + \
                  'ROOT::RDataFrame rdf(64);' + \
                  'auto g = rdf.Define("r","gRandom->Gaus(0,1)");' + \
                  'auto h2Proxy = g.Histo1D({"h1","h1",64, -2., 2.},"r");'
        ROOT.gInterpreter.ProcessLine(cppCode)
        h2 = ROOT.h2Proxy.GetValue()

        self.assertEqual(h1.GetEntries(), h2.GetEntries())
        self.assertEqual(h1.GetMean(), h2.GetMean())
        self.assertEqual(h1.GetStdDev(), h2.GetStdDev())


if __name__ == '__main__':
    unittest.main()
