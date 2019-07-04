import unittest
import ROOT
RTensor = ROOT.TMVA.Experimental.RTensor
import numpy as np


class ArrayInterface(unittest.TestCase):
    """
    Test memory adoption of RTensor array interface.
    """

    # Helpers
    dtypes = [
        "int", "unsigned int", "long", "long long", "Long64_t", "unsigned long",
        "unsigned long long", "ULong64_t", "float", "double"
    ]

    def get_maximum_for_dtype(self, dtype):
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        if np.issubdtype(dtype, np.floating):
            return np.finfo(dtype).max

    def get_minimum_for_dtype(self, dtype):
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).min
        if np.issubdtype(dtype, np.floating):
            return np.finfo(dtype).min

    def check_memory_adoption(self, root_obj, np_obj):
        # TODO
        pass
        """
        np_obj[0,0] = self.get_maximum_for_dtype(np_obj.dtype)
        np_obj[0,1] = self.get_minimum_for_dtype(np_obj.dtype)
        self.assertEqual(root_obj[0], np_obj[0])
        self.assertEqual(root_obj[1], np_obj[1])
        """

    def check_shape(self, expected_shape, np_obj):
        self.assertEqual(len(expected_shape), len(np_obj.shape))
        for a, b in zip(expected_shape, np_obj.shape):
            self.assertEqual(a, b)


    # Tests
    def test_memoryAdoption(self):
        """
        Test correct adoption of different datatypes
        """
        shape = ROOT.std.vector("size_t")((2, 2))
        for dtype in self.dtypes:
            root_obj = RTensor(dtype)(shape)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption(root_obj, np_obj)
            self.check_shape((2, 2), np_obj)

    def test_memoryLayout(self):
        """
        Test adoption of the memory layout
        """
        shape = ROOT.std.vector("size_t")((2, 2))
        x = RTensor("float")(shape)
        y = np.asarray(x)
        self.assertTrue(y.flags.c_contiguous)

        x = x.Transpose()
        y = np.asarray(x)
        self.assertTrue(y.flags.f_contiguous)

    def test_ownData(self):
        """
        Test ownership of adopted numpy array
        """
        shape = ROOT.std.vector("size_t")((2, 2))
        x = RTensor("float")(shape)
        y = np.asarray(x)
        self.assertFalse(y.flags.owndata)

        y = np.transpose(y)
        self.assertFalse(y.flags.owndata)

        y = np.copy(y)
        self.assertTrue(y.flags.owndata)
