import os
import sys
import unittest
import warnings

import pyspark

import ROOT

import DistRDF
from DistRDF.Backends import Spark


class RunGraphsTests(unittest.TestCase):
    """Tests usage of RunGraphs function with Spark backend"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Synchronize PYSPARK_PYTHON variable to the current Python executable.
          Needed to avoid mismatch between python versions on driver and on the
          fake executor on the same machine.
        - Ignore `ResourceWarning: unclosed socket` warning triggered by Spark.
          this is ignored by default in any application, but Python's unittest
          library overrides the default warning filters thus exposing this
          warning
        - Initialize a SparkContext for the tests in this class
        """
        os.environ["PYSPARK_PYTHON"] = sys.executable

        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

        sparkconf = pyspark.SparkConf().setMaster("local[2]")
        cls.sc = pyspark.SparkContext(conf=sparkconf)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        os.environ["PYSPARK_PYTHON"] = ""

        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

        cls.sc.stop()

    def test_rungraphs_spark_3histos(self):
        """
        Submit three different Spark RDF graphs concurrently
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
            Spark.RDataFrame(treename, filename, sparkcontext=self.sc, npartitions=2)
                 .Histo1D((col, col, 1, 40, 45), col)
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
            self.assertAlmostEqual(histo.GetMean(), 42)

        os.remove(filename)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
