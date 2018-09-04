import unittest
import ROOT
import numpy as np


class ArrayInterface(unittest.TestCase):
    # Helpers
    dtypes = [
        "int", "unsigned int", "long", "unsigned long", "float", "double"
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
        root_obj[0] = self.get_maximum_for_dtype(np_obj.dtype)
        root_obj[1] = self.get_minimum_for_dtype(np_obj.dtype)
        self.assertEqual(root_obj[0], np_obj[0])
        self.assertEqual(root_obj[1], np_obj[1])

    def check_shape(self, expected_shape, np_obj):
        self.assertEqual(expected_shape, np_obj.shape)

    # Tests
    def test_RVec(self):
        for dtype in self.dtypes:
            root_obj = ROOT.ROOT.VecOps.RVec(dtype)(2)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption(root_obj, np_obj)
            self.check_shape((2, ), np_obj)

    def test_STLVector(self):
        for dtype in self.dtypes:
            root_obj = ROOT.std.vector(dtype)(2)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption(root_obj, np_obj)
            self.check_shape((2, ), np_obj)

    def test_RTensor(self):
        shape = ROOT.std.vector("size_t")()
        shape.push_back(2)
        for dtype in self.dtypes:
            root_obj = ROOT.TMVA.Experimental.RTensor(dtype)(shape)
            np_obj = np.asarray(root_obj)
            self.check_memory_adoption(root_obj, np_obj)
            self.check_shape((2, ), np_obj)

    def test_RTensor_dim2(self):
        shape = ROOT.std.vector("size_t")((2, 2))
        root_obj = ROOT.TMVA.Experimental.RTensor("float")(shape)
        count = 0
        np_ref = np.empty((2, 2), dtype="float32")
        for i in range(2):
            for j in range(2):
                root_obj[i, j] = count
                np_ref[i, j] = count
                count += 1
        np_obj = np.asarray(root_obj)
        self.assertTrue((np_obj == np_ref).all())

    def test_RTensor_dim3(self):
        shape = ROOT.std.vector("size_t")((2, 2, 2))
        root_obj = ROOT.TMVA.Experimental.RTensor("float")(shape)
        count = 0
        np_ref = np.empty((2, 2, 2), dtype="float32")
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    root_obj[i, j, k] = count
                    np_ref[i, j, k] = count
                    count += 1
        np_obj = np.asarray(root_obj)
        self.assertTrue((np_obj == np_ref).all())


if __name__ == '__main__':
    unittest.main()
