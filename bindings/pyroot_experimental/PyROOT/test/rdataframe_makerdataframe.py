import unittest
import ROOT
import numpy as np
import sys


class MakeRDataFrame(unittest.TestCase):
    """
    Tests for the MakeRDataFrame feature enabling to read numpy arrays
    with RDataFrame.
    """

    dtypes = [
        "int32", "int64", "uint32", "uint64", "float32", "float64"
    ]

    def test_dtypes(self):
        """
        Test reading different datatypes
        """
        for dtype in self.dtypes:
            data = {"x": np.array([1, 2, 3], dtype=dtype)}
            df = ROOT.ROOT.RDF.MakeRDataFrame(data)
            self.assertEqual(df.Mean("x").GetValue(), 2)

    def test_multiple_columns(self):
        """
        Test reading multiple columns
        """
        data = {}
        for dtype in self.dtypes:
            data[dtype] = np.array([1, 2, 3], dtype=dtype)
        df = ROOT.ROOT.RDF.MakeRDataFrame(data)
        colnames = df.GetColumnNames()
        # Test column names
        for dtype in colnames:
            self.assertIn(dtype, self.dtypes)
        # Test mean
        for dtype in self.dtypes:
            self.assertEqual(df.Mean(dtype).GetValue(), 2)

    def test_refcount(self):
        """
        Check refcounts of associated PyObjects
        """
        data = {"x": np.array([1, 2, 3], dtype="float32")}
        self.assertEqual(sys.getrefcount(data), 2)
        self.assertEqual(sys.getrefcount(data["x"]), 2)
        df = ROOT.ROOT.RDF.MakeRDataFrame(data)
        self.assertTrue(hasattr(df, "__data__"))
        self.assertEqual(sys.getrefcount(data["x"]), 3)

    def test_transformations(self):
        """
        Test the use of transformations
        """
        data = {"x": np.array([1, 2, 3], dtype="float32")}
        df = ROOT.ROOT.RDF.MakeRDataFrame(data)
        df2 = df.Filter("x>1").Define("y", "2*x")
        self.assertEqual(df2.Mean("x").GetValue(), 2.5)
        self.assertEqual(df2.Mean("y").GetValue(), 5)


if __name__ == '__main__':
    unittest.main()
