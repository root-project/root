import unittest

import ROOT

RRegularAxis = ROOT.Experimental.RRegularAxis
RVariableBinAxis = ROOT.Experimental.RVariableBinAxis


class RDFHist(unittest.TestCase):
    def test_Regular(self):
        df = ROOT.RDataFrame(10)
        dfX = df.Define("x", "rdfentry_ + 5.5")
        hist = dfX.Hist(10, (5.0, 15.0), "x")
        self.assertEqual(hist.GetNEntries(), 10)

    def test_MultiDim(self):
        df = ROOT.RDataFrame(10)
        dfXY = df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5")

        regularAxis = RRegularAxis(10, (5.0, 15.0))
        bins = [i for i in range(0, 21)]
        variableBinAxis = RVariableBinAxis(bins)
        hist = dfXY.Hist([regularAxis, variableBinAxis], ["x", "y"])
        self.assertEqual(hist.GetNEntries(), 10)

    def test_Weight(self):
        df = ROOT.RDataFrame(10)
        dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03")
        hist = dfXW.Hist(10, (5.0, 15.0), "x", "w")
        self.assertEqual(hist.GetNEntries(), 10)

        regularAxis = RRegularAxis(10, (5.0, 15.0))
        hist = dfXW.Hist([regularAxis], ["x"], "w")
        self.assertEqual(hist.GetNEntries(), 10)
