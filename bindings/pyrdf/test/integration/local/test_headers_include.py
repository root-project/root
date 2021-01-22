import os
import unittest

import PyRDF


class IncludesLocalTest(unittest.TestCase):
    """Check that the required header files are properly included."""

    def tearDown(self):
        """remove included headers after analysis"""
        PyRDF.includes_headers.clear()

    def test_include_dir_and_headers(self):
        """
        Check that the filter operation is able to use C++ functions included
        from a list with a directory and a single header file.
        """
        PyRDF.include_headers([
            "test_headers/headers_folder",
            "test_headers/header1.hxx"
        ])
        # creates and RDataFrame with 10 integers [0...9]
        rdf = PyRDF.RDataFrame(10)

        # This filters out all numbers less than 5
        filter1 = rdf.Filter("check_number_less_than_5(tdfentry_)")
        # This filters out all numbers greater than 5
        filter2 = rdf.Filter("check_number_greater_than_5(tdfentry_)")
        # This filters out all numbers less than 10
        filter3 = rdf.Filter("check_number_less_than_10(tdfentry_)")

        count1 = filter1.Count()
        count2 = filter2.Count()
        count3 = filter3.Count()

        # The final answer should respectively 5 integers less than 5,
        # 4 integers greater than 5 and 10 integers less than 10.
        self.assertEqual(count1.GetValue(), 5)
        self.assertEqual(count2.GetValue(), 4)
        self.assertEqual(count3.GetValue(), 10)
