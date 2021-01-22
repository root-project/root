import unittest
import PyRDF
import math


class IncludesSparkTest(unittest.TestCase):
    """
    Check that the required header files are properly included in Spark
    environment.
    """
    def tearDown(self):
        """
        Clean up the `SparkContext` objects that were created and remove
        included headers from the global set.
        """
        from PyRDF import current_backend
        current_backend.sparkContext.stop()
        PyRDF.includes_headers.clear()

    def test_includes_function_with_filter_and_histo(self):
        """
        Check that the filter operation is able to use C++ functions that
        were included using header files.
        """
        PyRDF.use("spark")
        PyRDF.include_headers("../local/test_headers/header1.hxx")

        rdf = PyRDF.RDataFrame(10)

        # This filters out all numbers less than 5
        rdf_filtered = rdf.Filter("check_number_less_than_5(tdfentry_)")
        histo = rdf_filtered.Histo1D("tdfentry_")

        # The expected results after filtering
        # The actual set of numbers required after filtering
        required_numbers = range(5)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(sum((x - required_mean)**2
                                    for x in required_numbers) / required_size)

        # Compare the sizes of equivalent set of numbers
        self.assertEqual(histo.GetEntries(), float(required_size))

        # Compare the means of equivalent set of numbers
        self.assertEqual(histo.GetMean(), required_mean)

        # Compare the standard deviations of equivalent set of numbers
        self.assertEqual(histo.GetStdDev(), required_stdDev)

    def test_extend_ROOT_include_path(self):
        """
        Check that the include path of ROOT is extended with the directories
        specified in `PyRDF.include_headers()` so references between headers
        are correctly solved.
        """
        import ROOT

        header_folder = "../local/test_headers/headers_folder"

        PyRDF.use("spark")
        PyRDF.include_headers(header_folder)

        # Get list of include paths seen by ROOT
        ROOT_include_path = ROOT.gInterpreter.GetIncludePath().split(" ")

        # Create new include folder token
        new_folder_include = "-I\"{}\"".format(header_folder)

        # Check that new folder is in ROOT include paths
        self.assertTrue(new_folder_include in ROOT_include_path)

        # Create an RDataFrame with 100 integers from 0 to 99
        rdf = PyRDF.RDataFrame(100)

        # Filter numbers less than 10 and create an histogram
        rdf_less_than_10 = rdf.Filter("check_number_less_than_10(tdfentry_)")
        histo1 = rdf_less_than_10.Histo1D("tdfentry_")

        # Check that histogram has 10 entries and mean 4.5
        self.assertEqual(histo1.GetEntries(), 10)
        self.assertAlmostEqual(histo1.GetMean(), 4.5)


if __name__ == "__main__":
    unittest.main()
