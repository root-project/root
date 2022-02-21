import unittest
import warnings

import pyspark

import ROOT

import DistRDF
from DistRDF.Backends import Spark


class VariationsTest(unittest.TestCase):
    """Tests usage of systematic variations with Spark backend"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Ignore `ResourceWarning: unclosed socket` warning triggered by Spark.
          this is ignored by default in any application, but Python's unittest
          library overrides the default warning filters thus exposing this
          warning
        - Initialize a SparkContext for the tests in this class
        """
        warnings.simplefilter("ignore", ResourceWarning)

        sparkconf = pyspark.SparkConf().setMaster("local[2]")
        cls.sc = pyspark.SparkContext(conf=sparkconf)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        warnings.simplefilter("default", ResourceWarning)

        cls.sc.stop()

    def test_histo(self):
        df = Spark.RDataFrame(10, sparkcontext=self.sc, npartitions=2).Define("x", "1")
        df1 = df.Vary("x", "ROOT::RVecI{-2,2}", ["down", "up"])
        h = df1.Histo1D("x")
        histos = DistRDF.VariationsFor(h)

        expectednames = ["nominal", "x:up", "x:down"]
        expectedmeans = [1, 2, -2]
        for varname, mean in zip(expectednames, expectedmeans):
            histo = histos[varname]
            self.assertIsInstance(histo, ROOT.TH1D)
            self.assertEqual(histo.GetEntries(), 10)
            self.assertAlmostEqual(histo.GetMean(), mean)

    def test_graph(self):
        df = Spark.RDataFrame(10, sparkcontext=self.sc, npartitions=2).Define("x", "1")
        g = df.Vary("x", "ROOT::RVecI{-1, 2}", nVariations=2).Graph("x", "x")
        gs = DistRDF.VariationsFor(g)

        self.assertAlmostEqual(g.GetMean(), 1)

        expectednames = ["nominal", "x:0", "x:1"]
        expectedmeans = [1, -1, 2]
        for varname, mean in zip(expectednames, expectedmeans):
            graph = gs[varname]
            self.assertIsInstance(graph, ROOT.TGraph)
            self.assertAlmostEqual(graph.GetMean(), mean)

    def test_mixed(self):
        df = Spark.RDataFrame(10, sparkcontext=self.sc, npartitions=2).Define("x", "1").Define("y", "42")
        h = df.Vary("x", "ROOT::RVecI{-1, 2}", variationTags=["down", "up"]).Histo1D("x", "y")
        histos = DistRDF.VariationsFor(h)

        expectednames = ["nominal", "x:down", "x:up"]
        expectedmeans = [1, -1, 2]
        expectedmax = 420
        for varname, mean in zip(expectednames, expectedmeans):
            histo = histos[varname]
            self.assertIsInstance(histo, ROOT.TH1D)
            self.assertAlmostEqual(histo.GetMaximum(), expectedmax)
            self.assertAlmostEqual(histo.GetMean(), mean)

    def test_simultaneous(self):
        df = Spark.RDataFrame(10, sparkcontext=self.sc, npartitions=2).Define("x", "1").Define("y", "42")
        h = df.Vary(["x", "y"],
                    "ROOT::RVec<ROOT::RVecI>{{-1, 2, 3}, {41, 43, 44}}",
                    ["down", "up", "other"], "xy").Histo1D("x", "y")
        histos = DistRDF.VariationsFor(h)

        expectednames = ["nominal", "xy:down", "xy:up", "xy:other"]
        expectedmeans = [1, -1, 2, 3]
        expectedmax = [420, 410, 430, 440]
        for varname, mean, maxval in zip(expectednames, expectedmeans, expectedmax):
            graph = histos[varname]
            self.assertIsInstance(graph, ROOT.TH1D)
            self.assertAlmostEqual(graph.GetMaximum(), maxval)
            self.assertAlmostEqual(graph.GetMean(), mean)

    def test_varyfiltersum(self):
        df = Spark.RDataFrame(10, sparkcontext=self.sc, npartitions=2).Define("x", "1")
        df_sum = df.Vary("x", "ROOT::RVecI{-1*x, 2*x}", ("down", "up"), "myvariation").Filter("x > 0").Sum("x")

        self.assertAlmostEqual(df_sum.GetValue(), 10)

        sums = DistRDF.VariationsFor(df_sum)

        expectednames = ["nominal", "myvariation:down", "myvariation:up"]
        expectedsums = [10, 0, 20]
        for varname, val in zip(expectednames, expectedsums):
            self.assertAlmostEqual(sums[varname], val)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
