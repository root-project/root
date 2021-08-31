import math
import os
import sys
import unittest

import pyspark
from DistRDF.Backends import Spark


class IncludesSparkTest(unittest.TestCase):
    """
    Check that the required header files are properly included in Spark
    environment.
    """

    @classmethod
    def setUpClass(cls):
        """
        Synchronize PYSPARK_PYTHON variable to the current Python executable.

        Needed to avoid mismatch between python versions on driver and on
        the fake executor on the same machine.
        """
        os.environ["PYSPARK_PYTHON"] = sys.executable

    def tearDown(self):
        """Stop any created SparkContext"""
        pyspark.SparkContext.getOrCreate().stop()

    @classmethod
    def tearDownClass(cls):
        """
        Stop the SparkContext and reset environment variable.
        """
        os.environ["PYSPARK_PYTHON"] = ""

    def test_header_distribution_and_inclusion(self):
        """
        Tests for the distribution of headers to the workers and their
        corresponding inclusion.
        """

        self._includes_function_with_filter_and_histo()
        self._extend_ROOT_include_path()

    def _includes_function_with_filter_and_histo(self):
        """
        Check that the filter operation is able to use C++ functions that
        were included using header files.
        """

        rdf = Spark.RDataFrame(10)

        rdf._headnode.backend.distribute_headers("test_headers/header1.hxx")

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
        import ROOT

        # Create an RDataFrame with 100 integers from 0 to 99
        rdf = Spark.RDataFrame(100)

        # Distribute headers to the workers
        header_folder = "test_headers/headers_folder"
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


if __name__ == "__main__":
    unittest.main(argv=[__file__])
