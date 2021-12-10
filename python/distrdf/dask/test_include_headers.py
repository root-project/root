import math
import os
import unittest

import ROOT

from DistRDF.Backends import Dask

from dask.distributed import Client, LocalCluster

class IncludesDaskTest(unittest.TestCase):
    """
    Check that the required header files are properly included in Dask
    environment.
    """

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

    def _includes_function_with_filter_and_histo(self):
        """
        Check that the filter operation is able to use C++ functions that
        were included using header files.
        """

        rdf = Dask.RDataFrame(10, daskclient=self.client)

        rdf._headnode.backend.distribute_headers("../test_headers/header1.hxx")

        # This filters out all numbers less than 5
        rdf_filtered = rdf.Filter("check_number_less_than_5(tdfentry_)")
        histo = rdf_filtered.Histo1D("tdfentry_")

        # The expected results after filtering
        # The actual set of numbers required after filtering
        required_numbers = range(5)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(
            sum((x - required_mean)**2 for x in required_numbers) /
            required_size)

        # Compare the sizes of equivalent set of numbers
        self.assertEqual(histo.GetEntries(), float(required_size))

        # Compare the means of equivalent set of numbers
        self.assertEqual(histo.GetMean(), required_mean)

        # Compare the standard deviations of equivalent set of numbers
        self.assertEqual(histo.GetStdDev(), required_stdDev)

    def _extend_ROOT_include_path(self):
        """
        Check that the include path of ROOT is extended with the directories
        specified in `DistRDF.include_headers()` so references between headers
        are correctly solved.
        """

        # Create an RDataFrame with 100 integers from 0 to 99
        rdf = Dask.RDataFrame(100, daskclient=self.client)

        # Distribute headers to the workers
        header_folder = "../test_headers/headers_folder"
        rdf._headnode.backend.distribute_headers(header_folder)

        # Get list of include paths seen by ROOT
        ROOT_include_path = ROOT.gInterpreter.GetIncludePath().split(" ")

        # Create new include folder token
        new_folder_include = "-I\"{}\"".format(header_folder)

        # Check that new folder is in ROOT include paths
        self.assertTrue(new_folder_include in ROOT_include_path)

        # Filter numbers less than 10 and create an histogram
        rdf_less_than_10 = rdf.Filter("check_number_less_than_10(tdfentry_)")
        histo1 = rdf_less_than_10.Histo1D("tdfentry_")

        # Check that histogram has 10 entries and mean 4.5
        self.assertEqual(histo1.GetEntries(), 10)
        self.assertAlmostEqual(histo1.GetMean(), 4.5)

    def test_header_distribution_and_inclusion(self):
        """
        Tests for the distribution of headers to the workers and their
        corresponding inclusion.
        """

        self._includes_function_with_filter_and_histo()
        self._extend_ROOT_include_path()

if __name__ == "__main__":
    unittest.main(argv=[__file__])
