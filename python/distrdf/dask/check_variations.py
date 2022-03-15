import pytest

import ROOT

import DistRDF
from DistRDF.Backends import Dask


class TestVariations:
    """Tests usage of systematic variations with Dask backend"""

    def test_histo(self, connection):
        df = Dask.RDataFrame(10, daskclient=connection, npartitions=2).Define("x", "1")
        df1 = df.Vary("x", "ROOT::RVecI{-2,2}", ["down", "up"])
        h = df1.Histo1D("x")
        histos = DistRDF.VariationsFor(h)

        expectednames = ["nominal", "x:up", "x:down"]
        expectedmeans = [1, 2, -2]
        for varname, mean in zip(expectednames, expectedmeans):
            histo = histos[varname]
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == 10
            assert histo.GetMean() == mean

    def test_graph(self, connection):
        df = Dask.RDataFrame(10, daskclient=connection, npartitions=2).Define("x", "1")
        g = df.Vary("x", "ROOT::RVecI{-1, 2}", nVariations=2).Graph("x", "x")
        gs = DistRDF.VariationsFor(g)

        assert g.GetMean() == 1

        expectednames = ["nominal", "x:0", "x:1"]
        expectedmeans = [1, -1, 2]
        for varname, mean in zip(expectednames, expectedmeans):
            graph = gs[varname]
            assert isinstance(graph, ROOT.TGraph)
            assert graph.GetMean() == mean

    def test_mixed(self, connection):
        df = Dask.RDataFrame(10, daskclient=connection, npartitions=2).Define("x", "1").Define("y", "42")
        h = df.Vary("x", "ROOT::RVecI{-1, 2}", variationTags=["down", "up"]).Histo1D("x", "y")
        histos = DistRDF.VariationsFor(h)

        expectednames = ["nominal", "x:down", "x:up"]
        expectedmeans = [1, -1, 2]
        expectedmax = 420
        for varname, mean in zip(expectednames, expectedmeans):
            histo = histos[varname]
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetMaximum() == expectedmax
            assert histo.GetMean() == mean

    def test_simultaneous(self, connection):
        df = Dask.RDataFrame(10, daskclient=connection, npartitions=2).Define("x", "1").Define("y", "42")
        h = df.Vary(["x", "y"],
                    "ROOT::RVec<ROOT::RVecI>{{-1, 2, 3}, {41, 43, 44}}",
                    ["down", "up", "other"], "xy").Histo1D("x", "y")
        histos = DistRDF.VariationsFor(h)

        expectednames = ["nominal", "xy:down", "xy:up", "xy:other"]
        expectedmeans = [1, -1, 2, 3]
        expectedmax = [420, 410, 430, 440]
        for varname, mean, maxval in zip(expectednames, expectedmeans, expectedmax):
            graph = histos[varname]
            assert isinstance(graph, ROOT.TH1D)
            assert graph.GetMaximum() == maxval
            assert graph.GetMean() == mean

    def test_varyfiltersum(self, connection):
        df = Dask.RDataFrame(10, daskclient=connection, npartitions=2).Define("x", "1")
        df_sum = df.Vary("x", "ROOT::RVecI{-1*x, 2*x}", ("down", "up"), "myvariation").Filter("x > 0").Sum("x")

        assert df_sum.GetValue() == 10

        sums = DistRDF.VariationsFor(df_sum)

        expectednames = ["nominal", "myvariation:down", "myvariation:up"]
        expectedsums = [10, 0, 20]
        for varname, val in zip(expectednames, expectedsums):
            assert sums[varname] == val


if __name__ == "__main__":
    pytest.main(args=[__file__])
