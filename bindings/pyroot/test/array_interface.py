import unittest
import ROOT
import numpy as np


class ArrayInterface(unittest.TestCase):
    # Helpers
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

    def check_memory_adoption_vector(self, root_obj, np_obj):
        root_obj[0] = self.get_maximum_for_dtype(np_obj.dtype)
        root_obj[1] = self.get_minimum_for_dtype(np_obj.dtype)
        self.assertEqual(root_obj[0], np_obj[0])
        self.assertEqual(root_obj[1], np_obj[1])

    def check_memory_adoption_matrix(self, root_obj, np_obj):
        root_obj[0][0] = self.get_maximum_for_dtype(np_obj.dtype)
        root_obj[1][0] = self.get_minimum_for_dtype(np_obj.dtype)
        self.assertEqual(root_obj[0][0], np_obj[0, 0])
        self.assertEqual(root_obj[1][0], np_obj[1, 0])

    def check_shape_vector(self, expected_shape, np_obj):
        self.assertEqual(expected_shape, np_obj.shape)

    def check_shape_matrix(self, expected_shape, np_obj):
        self.assertEqual(expected_shape, np_obj.shape)

    # TVector
    def test_TVector(self):
        for dtype in ["float", "double"]:
            root_obj = ROOT.TVectorT(dtype)(2)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption_vector(root_obj, np_obj)
            self.check_shape_vector((2, ), np_obj)

    # TVec
    def test_TVec(self):
        for dtype in [
                "int", "unsigned int", "long", "unsigned long", "float",
                "double"
        ]:
            root_obj = ROOT.Experimental.VecOps.TVec(dtype)(2)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption_vector(root_obj, np_obj)
            self.check_shape_vector((2, ), np_obj)

    # STL vector
    def test_STLVector(self):
        for dtype in [
                "int", "unsigned int", "long", "unsigned long", "float",
                "double"
        ]:
            root_obj = ROOT.std.vector(dtype)(2)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption_vector(root_obj, np_obj)
            self.check_shape_vector((2, ), np_obj)

    # TMatrix
    def test_TMatrix(self):
        for dtype in ["float", "double"]:
            root_obj = ROOT.TMatrixT(dtype)(2, 1)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption_matrix(root_obj, np_obj)
            self.check_shape_matrix((2, 1), np_obj)


if __name__ == '__main__':
    unittest.main()
