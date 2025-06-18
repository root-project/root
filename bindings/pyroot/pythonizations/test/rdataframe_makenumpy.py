import gc
import sys
import unittest

import numpy as np
import ROOT


class DataFrameFromNumpy(unittest.TestCase):
    """
    Tests for the FromNumpy feature enabling to read numpy arrays
    with RDataFrame.
    """

    dtypes = ["int32", "int64", "uint32", "uint64", "float32", "float64"]

    def test_dtypes(self):
        """
        Test reading different datatypes
        """
        for dtype in self.dtypes:
            data = {"x": np.array([1, 2, 3], dtype=dtype)}
            df = ROOT.RDF.FromNumpy(data)
            self.assertEqual(df.Mean("x").GetValue(), 2)

    def test_object_dtype_strings(self):
        """
        Test reading numpy arrays with dtype=object containing strings
        """
        data = {"x": np.array(["test_string_1", "test_string_2"], dtype=object)}
        df = ROOT.RDF.FromNumpy(data)
        colnames = df.GetColumnNames()

        self.assertIn("x", colnames)

        entries = df.AsNumpy()["x"]
        self.assertTrue(np.array_equal(entries, np.array(["test_string_1", "test_string_2"], dtype=object)))

    def test_empty_arrays(self):
        """
        Test creating an RDataFrame from an empty numpy array with different data types
        """
        for dtype in self.dtypes:
            data = {"x": np.array([], dtype=dtype)}
            df = ROOT.RDF.FromNumpy(data)
            colnames = df.GetColumnNames()
            self.assertIn("x", colnames)
            self.assertEqual(df.Count().GetValue(), 0)

        data_obj = {"x": np.array([], dtype="object")}
        with self.assertRaises(RuntimeError) as context:
            df = ROOT.RDF.FromNumpy(data_obj)

        self.assertIn(
            "Failure in creating column 'x' for RDataFrame: the input column type is 'object', which is not supported. Make sure your column type is supported",
            str(context.exception),
        )

    def test_multiple_columns(self):
        """
        Test reading multiple columns
        """
        data = {}
        for dtype in self.dtypes:
            data[dtype] = np.array([1, 2, 3], dtype=dtype)
        df = ROOT.RDF.FromNumpy(data)
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

        In case of Python <=3.14, we expect a refcount of 2 for the data dict,
        because the call to sys.getrefcount creates a second reference by
        itself. Starting from Python 3.14, we expect a refcount of 1 because
        there were changes to the interpreter to avoid some unnecessary ref
        counts. See also:
        https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-refcount
        """
        extra_ref_count = int(sys.version_info < (3, 14))
        data = {"x": np.array([1, 2, 3], dtype="float32")}
        gc.collect()
        self.assertEqual(sys.getrefcount(data), 1 + extra_ref_count)
        self.assertEqual(sys.getrefcount(data["x"]), 2)

        df = ROOT.RDF.FromNumpy(data)
        gc.collect()
        self.assertEqual(sys.getrefcount(df), 1 + extra_ref_count)

        self.assertEqual(sys.getrefcount(data["x"]), 3)

    def test_transformations(self):
        """
        Test the use of transformations
        """
        data = {"x": np.array([1, 2, 3], dtype="float32")}
        df = ROOT.RDF.FromNumpy(data)
        df2 = df.Filter("x>1").Define("y", "2*x")
        self.assertEqual(df2.Mean("x").GetValue(), 2.5)
        self.assertEqual(df2.Mean("y").GetValue(), 5)

    def test_delete_dict(self):
        """
        Test behaviour with data dictionary going out of scope
        """
        data = {"x": np.array([1, 2, 3], dtype="float32")}
        df = ROOT.RDF.FromNumpy(data)
        del data
        self.assertEqual(df.Mean("x").GetValue(), 2)

    def test_delete_numpy_array(self):
        """
        Test behaviour with numpy array going out of scope
        """
        x = np.array([1, 2, 3], dtype="float32")
        data = {"x": x}
        df = ROOT.RDF.FromNumpy(data)
        del x
        self.assertEqual(df.Mean("x").GetValue(), 2)

    def test_inplace_dict(self):
        """
        Test behaviour with inplace dictionary
        """
        df = ROOT.RDF.FromNumpy({"x": np.array([1, 2, 3], dtype="float32")})
        self.assertEqual(df.Mean("x").GetValue(), 2)

    def test_lifetime_numpy_array(self):
        """
        Test lifetime of numpy array
        """
        x = np.array([1, 2, 3], dtype="float32")
        gc.collect()
        ref1 = sys.getrefcount(x)

        df = ROOT.RDF.FromNumpy({"x": x})
        gc.collect()
        ref2 = sys.getrefcount(x)
        self.assertEqual(ref2, ref1 + 1)

        del df
        gc.collect()
        ref3 = sys.getrefcount(x)
        self.assertEqual(ref1, ref3)

    def test_lifetime_datasource(self):
        """
        Test lifetime of datasource

        Datasource survives until last node of the graph goes out of scope
        """
        x = np.array([1, 2, 3], dtype="float32")
        gc.collect()
        ref1 = sys.getrefcount(x)

        # Data source has dictionary with RVecs attached, which take a reference
        # to the numpy array
        df = ROOT.RDF.FromNumpy({"x": x})
        m = df.Mean("x")
        gc.collect()
        ref2 = sys.getrefcount(x)
        self.assertEqual(ref1 + 1, ref2)

        # Deleting the root node does not change anything since the datasource
        # owns the RVecs
        del df
        self.assertEqual(m.GetValue(), 2)
        gc.collect()
        ref3 = sys.getrefcount(x)
        self.assertEqual(ref1 + 1, ref3)

        # Deleting the last node releases the RVecs and releases the reference
        # to the numpy array
        del m
        gc.collect()
        ref4 = sys.getrefcount(x)
        self.assertEqual(ref1, ref4)

    def test_sliced_array(self):
        """
        Test correct reading of a sliced numpy array (#13690)
        """
        table = np.array([[1, 2], [3, 4]], dtype="int64")
        columns = {"x": table[:, 0], "y": table[:, 1]}
        df = ROOT.RDF.FromNumpy(columns)
        x_col = df.Take["Long64_t"]("x")
        y_col = df.Take["Long64_t"]("y")
        self.assertEqual(list(x_col.GetValue()), [1, 3])
        self.assertEqual(list(y_col.GetValue()), [2, 4])


if __name__ == "__main__":
    unittest.main()
