import unittest
import ROOT
import numpy as np

class PyDefine(unittest.TestCase):
    """
    Testing Pythonized Define of RDF
    """

    def test_with_dtypes(self):
        """
        Tests the pythonized define with all the numba declare datatypes and 
        """
        numba_declare_dtypes = ['float', 'double', 'int', 'unsigned int', 'long', 'unsigned long', 'bool']
        rdf = ROOT.RDataFrame(10)
        for type in numba_declare_dtypes:
            col_name = "col_" + type.replace(" ","") 
            rdf = rdf.Define(col_name, f"({type}) rdfentry_")
            rdf = rdf.Define(col_name + "_arr", lambda col: np.array([col,col]), [col_name])
            arr = np.arange(0, 10)
            if type == 'bool':
                arr = np.array(arr, dtype='bool')
            flag1 = np.array_equal(rdf.AsNumpy()[col_name], arr)
            flag2 = True
            for idx, entry in enumerate(rdf.AsNumpy()[col_name + "_arr"]):
                if not (entry[0] == arr[idx] and entry[1] == arr[idx]):
                    flag2 = False
            self.assertTrue(flag1 and flag2)
    
    def test_define_overload1(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")
        rdf = rdf.Define("x2", lambda y: y*y, ["x"])
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x2"], arr*arr)
        self.assertTrue(flag)

    def test_define_overload2(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")
        rdf = rdf.Define("x2", lambda x: x*x)
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x2"], arr*arr)
        self.assertTrue(flag)

    def test_define_extra_args(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")
        def x_y(x, y):
            return x*y
        rdf = rdf.Define("x_y", x_y , extra_args = {"y": 0.5})
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x_y"], arr*0.5)
        self.assertTrue(flag)

    def test_capture_from_scope(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")
        y = 0.5
        def x_times_y(x):
            return x*y
        rdf = rdf.Define("x_y", x_times_y )
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x_y"], arr*0.5)
        self.assertTrue(flag)
        
    def test_arrays(self):
        rdf = ROOT.RDataFrame(5).Define("x", "rdfentry_")
        rdf = rdf.Define("x_arr", lambda x:  np.array([x, x]))
        def norm(x_arr):
            return np.sqrt(x_arr[0]**2 + x_arr[1]**2)
        rdf = rdf.Define("mag", norm)
        arr = np.arange(0, 5)
        arr = np.sqrt(arr*arr + arr*arr )
        flag = np.array_equal(rdf.AsNumpy()["mag"], arr)
        self.assertTrue(flag)
    
if __name__ == '__main__':
    unittest.main()
