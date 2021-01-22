import PyRDF
import unittest


class RDataFrameAsNumpy(unittest.TestCase):
    """Test `AsNumpy` functionality for RDataFrame"""

    def test_asnumpy_return_arrays(self):
        """Test support for `AsNumpy` pythonization in local backend"""
        import numpy

        # Let's create a simple dataframe with ten rows and two columns
        df = PyRDF.RDataFrame(10).Define("x", "(int)rdfentry_")\
                                 .Define("y", "1.f/(1.f+rdfentry_)")

        # Build a dictionary of numpy arrays.
        npy = df.AsNumpy()
        self.assertIsInstance(npy, dict)

        # Retrieve the two numpy arrays with the column names of the original
        # RDataFrame as dictionary keys.
        npy_x = npy["x"]
        npy_y = npy["y"]
        self.assertIsInstance(npy_x, numpy.ndarray)
        self.assertIsInstance(npy_y, numpy.ndarray)

        # Check the two arrays are of the same length as the original columns.
        self.assertEqual(len(npy_x), 10)
        self.assertEqual(len(npy_y), 10)

        # Check the types correspond to the ones of the original columns.
        int_32_dtype = numpy.dtype("int32")
        float_32_dtype = numpy.dtype("float32")
        self.assertEqual(npy_x.dtype, int_32_dtype)
        self.assertEqual(npy_y.dtype, float_32_dtype)
