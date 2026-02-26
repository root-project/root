import unittest

import ROOT


class RDFHist(unittest.TestCase):
    def test_Regular(self):
        df = ROOT.RDataFrame(10)
        dfX = df.Define("x", "rdfentry_ + 5.5")
        hist = dfX.Hist(10, (5.0, 15.0), "x")
        self.assertEqual(hist.GetNEntries(), 10)

    def test_Weight(self):
        df = ROOT.RDataFrame(10)
        dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03")
        hist = dfXW.Hist(10, (5.0, 15.0), "x", "w")
        self.assertEqual(hist.GetNEntries(), 10)
