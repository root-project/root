import os

import pytest

import ROOT

import DistRDF
from DistRDF.Backends import Dask


class TestRunGraphs:
    """Tests usage of RunGraphs function with Dask backend"""

    def test_rungraphs_dask_3histos(self, connection):
        """
        Submit three different Dask RDF graphs concurrently
        """
        # Create a test file for processing
        treename = "myTree"
        filename = "2clusters.root"
        nentries = 10000
        opts = ROOT.RDF.RSnapshotOptions()
        opts.fAutoFlush = 5000
        ROOT.RDataFrame(nentries).Define("b1", "42")\
                                 .Define("b2", "42")\
                                 .Define("b3", "42")\
                                 .Snapshot(treename, filename, ["b1", "b2", "b3"], opts)

        histoproxies = [
            Dask.RDataFrame(treename, filename, daskclient=connection, npartitions=2)
                .Histo1D((col, col, 1, 40, 45), col)
            for col in ["b1", "b2", "b3"]
        ]

        # Before triggering the computation graphs values are None
        for proxy in histoproxies:
            assert proxy.proxied_node.value is None

        DistRDF.RunGraphs(histoproxies)

        # After RunGraphs all histograms are correctly assigned to the
        # node objects
        for proxy in histoproxies:
            histo = proxy.proxied_node.value
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == nentries
            assert histo.GetMean() == 42

        os.remove(filename)


if __name__ == "__main__":
    pytest.main(args=[__file__])
