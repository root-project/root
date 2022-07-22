import unittest
import ROOT
import numpy as np

import os
from rdf_filter_pyz_helper import CreateData, TYPE_TO_SYMBOL, filter_dict

class PyFilter(unittest.TestCase):
    """
    Testing Pythonized Filters of RDF
    """

    def test_with_dtypes(self):
        """
        Tests the pythonized filter with all the tree datatypes and 
        """
        CreateData()
        rdf = ROOT.RDataFrame("TestData", "./RDF_Filter_Pyz_TestData.root")
        test_cols =[str(c) for c in rdf.GetColumnNames()]
        for col_name in test_cols:
            func = filter_dict[TYPE_TO_SYMBOL[col_name]] # filter function
            x = rdf.Mean(col_name).GetValue()
            if col_name == 'Bool_t': x = True
            filtered = rdf.Filter(func, extra_args = {'x':x})
            res_root = filtered.AsNumpy()[col_name]
            if not isinstance(x, bool):
                filtered2 = rdf.Filter(f"{col_name} > {x}")
            else:
                if x:
                    filtered2 = rdf.Filter(f"{col_name} == true")
                else:
                    filtered2 = rdf.Filter(f"{col_name} == false")
            res_root2 = filtered2.AsNumpy()[col_name]
            self.assertTrue(np.array_equal(res_root,res_root2))

        os.remove("./RDF_Filter_Pyz_TestData.root")

    # CPP Overload 1: Filter(callable, col_list = [], name = "") => 3 Possibilities 
    def test_filter_overload1_a(self):
        """
        Test to verify the first overload (1.a) of filter
        Filter(callable, col_list, name)
        """
        rdf = ROOT.RDataFrame(5).Define("x", "(double) rdfentry_")
        def x_greater_than_2(x):
            return x>2
        fil1 = rdf.Filter(x_greater_than_2, ["x"], "x is more than 2")
        self.assertTrue(np.array_equal(fil1.AsNumpy()["x"], np.array([3, 4])))

    def test_filter_overload1_b(self):
        """
        Test to verify the first overload (1.b) of filter
        Filter(callable, col_list)
        """
        rdf = ROOT.RDataFrame(5).Define("x", "(double) rdfentry_")
        fil1 = rdf.Filter(lambda x: x>2, ["x"])
        self.assertTrue(np.array_equal(fil1.AsNumpy()["x"], np.array([3, 4])))
    
    def test_filter_overload1_c(self):
        """
        Test to verify the first overload (1.c) of filter
        Filter(callable)
        """
        rdf = ROOT.RDataFrame(5).Define("x", "(double) rdfentry_")
        def x_greater_than_2(x):
            return x>2
        fil1 = rdf.Filter(x_greater_than_2)
        self.assertTrue(np.array_equal(fil1.AsNumpy()["x"], np.array([3, 4])))
    
    # CPP Overload 3: Filter(callable, name)
    def test_filter_overload3(self):
        """
        Test to verify the third overload of filter
        Filter(callable, name)
        """
        rdf = ROOT.RDataFrame(5).Define("x", "(double) rdfentry_")
        def x_greater_than_2(x):
            return x>2
        fil1 = rdf.Filter(x_greater_than_2, "x is greater than 2")
        self.assertTrue(np.array_equal(fil1.AsNumpy()["x"], np.array([3, 4])))
    
    def test_capture_from_scope(self):
        rdf = ROOT.RDataFrame(5).Define("x", "(double) rdfentry_")
        y = 2
        def x_greater_than_y(x):
            return x > y
        fil1 = rdf.Filter(x_greater_than_y, "x is greater than 2")
        self.assertTrue(np.array_equal(fil1.AsNumpy()["x"], np.array([3, 4])))
    
if __name__ == '__main__':
    unittest.main()
