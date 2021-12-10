import os
import unittest

import ROOT

import DistRDF
from DistRDF.Backends import Dask

from dask.distributed import Client, LocalCluster


class RunGraphsTests(unittest.TestCase):
    """Tests usage of RunGraphs function with Dask backend"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

    def test_rungraphs_dask_3histos(self):
        """
        Submit three different Dask RDF graphs concurrently
        """
        # Create a test file for processing
        treename = "myTree"
        filename = "2clusters.root"
        nentries = 10000
        opts = ROOT.RDF.RSnapshotOptions()
        opts.fAutoFlush = 5000
        ROOT.RDataFrame(nentries).Define("b1", "gRandom->Gaus(10, 1)")\
                                 .Define("b2", "gRandom->Gaus(10, 1)")\
                                 .Define("b3", "gRandom->Gaus(10, 1)")\
                                 .Snapshot(treename, filename, ["b1", "b2", "b3"], opts)

        histoproxies = [
            Dask.RDataFrame(treename, filename, daskclient=self.client, npartitions=2)
                .Histo1D((col, col, 100, 0, 20), col)
            for col in ["b1", "b2", "b3"]
        ]

        # Before triggering the computation graphs values are None
        for proxy in histoproxies:
            self.assertIsNone(proxy.proxied_node.value)

        DistRDF.RunGraphs(histoproxies)

        # After RunGraphs all histograms are correctly assigned to the
        # node objects
        for proxy in histoproxies:
            histo = proxy.proxied_node.value
            self.assertIsInstance(histo, ROOT.TH1D)
            self.assertEqual(histo.GetEntries(), nentries)
            self.assertAlmostEqual(histo.GetMean(), 10)

        os.remove(filename)
